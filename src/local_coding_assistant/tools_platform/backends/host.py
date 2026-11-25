"""Host execution backend implementation.

Operates inside the resolved workspace root, uses subprocess argument lists,
applies environment and resource policy, and supports Windows and POSIX command
resolution. Receives WorkspaceManager and RepositoryContextService as injected
capabilities.
"""

from __future__ import annotations

import difflib
import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_coding_assistant.tools_platform.backends.base import ExecutionBackend
from local_coding_assistant.tools_platform.backends.match_paths_impl import (
    match_paths,
)
from local_coding_assistant.tools_platform.backends.search_text_impl import (
    search_with_python,
    search_with_ripgrep,
)
from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    FileObservation,
    ObservationSource,
    ToolOutputChunk,
)
from local_coding_assistant.tools_platform.exceptions import (
    ToolBackendError,
    ToolPathViolationError,
    ToolProcessError,
    ToolValidationError,
)
from local_coding_assistant.tools_platform.supervisor import (
    ExecutionConfig,
    OutputLimits,
    ProcessSupervisor,
    TimeoutPolicy,
)
from local_coding_assistant.tools_platform.supervisor.host_transport import (
    HostProcessTransport,
)
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.core.protocols import IConfigManager
    from local_coding_assistant.tools_platform.workspace import WorkspaceManager

logger = get_logger("tools_platform.backend.host")


@dataclass
class _EditResult:
    """Outcome of resolving an exact-text edit with fallback passes."""

    new_content_norm: str
    applied: bool
    actual_count: int


class HostBackend(ExecutionBackend):
    """Host execution backend.

    Executes tools directly on the host system within the workspace root.
    Receives RepositoryContextService for symbol search operations.
    """

    def __init__(
        self,
        workspace_root: Path,
        workspace_manager: WorkspaceManager,
        config_manager: IConfigManager,
    ) -> None:
        """Initialize the host backend.

        Args:
            workspace_root: Path to the workspace root directory
            workspace_manager: WorkspaceManager for path validation and symbol operations
            config_manager: Config manager for dynamic configuration resolution
        """
        self._workspace_root = Path(workspace_root)
        self._workspace_manager = workspace_manager
        self._config_manager = config_manager
        # Create host-specific transport and pass to supervisor
        transport = HostProcessTransport()
        self._supervisor = ProcessSupervisor(transport=transport)
        self._initialized = False

    @property
    def _timeout_policy(self) -> TimeoutPolicy:
        """Get timeout policy from config."""
        if self._config_manager is None:
            return TimeoutPolicy()
        tc = self._config_manager.global_config.tools_platform
        return TimeoutPolicy(
            soft_timeout=timedelta(seconds=tc.timeout.soft_seconds),
            hard_timeout=timedelta(seconds=tc.timeout.hard_seconds),
            idle_timeout=timedelta(seconds=tc.timeout.idle_seconds),
            heartbeat_interval=timedelta(seconds=tc.timeout.heartbeat_seconds),
            cancellation_grace=timedelta(seconds=tc.timeout.cancellation_grace_seconds),
        )

    @property
    def _output_limits(self) -> OutputLimits:
        """Get output limits from config."""
        if self._config_manager is None:
            return OutputLimits()
        tc = self._config_manager.global_config.tools_platform
        return OutputLimits(
            max_stdout_bytes=tc.output_limits.max_stdout_bytes,
            max_stderr_bytes=tc.output_limits.max_stderr_bytes,
            max_lines=tc.output_limits.max_lines,
        )

    @property
    def _env_allowlist(self) -> list[str] | None:
        """Get env allowlist from config."""
        if self._config_manager is None:
            return None
        return self._config_manager.global_config.tools_platform.env_allowlist

    def _filter_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        """Filter requested environment variables through the allowlist.

        Args:
            env: Requested environment variables

        Returns:
            Filtered environment, or None when no filtering is configured
        """
        if env is None or self._env_allowlist is None:
            return env
        return {key: value for key, value in env.items() if key in self._env_allowlist}

    async def read_file(
        self,
        path: Path,
        start_line: int | None = None,
        end_line: int | None = None,
        byte_limit: int | None = None,
    ) -> str:
        """Read file content with optional line range and byte limit.

        Args:
            path: Path to file (relative to workspace root)
            start_line: Optional start line (1-indexed)
            end_line: Optional end line (1-indexed)
            byte_limit: Maximum bytes to read

        Returns:
            File content

        Raises:
            ToolValidationError: If path is invalid or file not found
            ToolBackendError: If read fails
        """
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise ToolValidationError(
                message=f"File not found: {path}",
                details={"path": str(path)},
            )

        if not full_path.is_file():
            raise ToolValidationError(
                message=f"Path is not a file: {path}",
                details={"path": str(path)},
            )

        try:
            # Normalize line endings for LLM-facing content when enabled.
            # Baseline observation (via observe_file) stays on raw disk bytes.
            content = self._workspace_manager.read_file_normalized(full_path)

            # Apply line range if specified
            if start_line is not None or end_line is not None:
                lines = content.splitlines(keepends=True)
                start = (start_line - 1) if start_line else 0
                end = end_line if end_line else len(lines)
                content = "".join(lines[start:end])

            # Apply byte limit if specified
            if byte_limit is not None and len(content.encode("utf-8")) > byte_limit:
                content = content.encode("utf-8")[:byte_limit].decode(
                    "utf-8", errors="ignore"
                )

            return content
        except UnicodeDecodeError as e:
            raise ToolValidationError(
                message=f"Failed to decode file as UTF-8: {path}",
                details={"path": str(path), "error": str(e)},
            ) from e
        except Exception as e:
            raise ToolBackendError(
                message=f"Failed to read file: {path}",
                details={"path": str(path), "error": str(e)},
            ) from e

    async def observe_file(
        self, path: Path, session_id: str | None = None
    ) -> FileObservation:
        """Observe file state for conflict detection.

        Args:
            path: Path to file (relative to workspace root)
            session_id: Optional session ID to scope the file-version observation.

        Returns:
            FileObservation with current file state

        Raises:
            ToolBackendError: If observation fails
        """
        if self._workspace_manager is None:
            raise ToolBackendError(
                message="WorkspaceManager not available for file observation",
                details={"path": str(path)},
            )

        resolved_path = self._resolve_path(path)
        rel_path = self._workspace_manager.get_relative_path(resolved_path)
        return self._workspace_manager.observe_file(
            rel_path, source=ObservationSource.READ, session_id=session_id
        )

    async def execute_command(
        self,
        argv: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ToolOutputChunk]:
        """Execute a command with streaming output.

        Args:
            argv: Command and arguments
            cwd: Working directory (relative to workspace root)
            env: Environment variables
            timeout_seconds: Timeout in seconds (overrides default)

        Yields:
            ToolOutputChunk with stdout/stderr/status

        Raises:
            ToolBackendError: If execution fails
        """
        # Resolve working directory to absolute path under workspace root
        if cwd is None:
            cwd = self.get_workspace_root()

        resolved_cwd = None
        if cwd is not None:
            resolved_cwd = self._resolve_path(cwd)

        # Build timeout policy (config defaults, overridable per call)
        timeout_policy = TimeoutPolicy(
            soft_timeout=self._timeout_policy.soft_timeout,
            hard_timeout=self._timeout_policy.hard_timeout,
            idle_timeout=self._timeout_policy.idle_timeout,
            heartbeat_interval=self._timeout_policy.heartbeat_interval,
            cancellation_grace=self._timeout_policy.cancellation_grace,
        )
        if timeout_seconds is not None:
            timeout_policy.hard_timeout = timedelta(seconds=timeout_seconds)
            timeout_policy.soft_timeout = timedelta(
                seconds=min(timeout_seconds * 0.5, 10)
            )

        # Build execution config
        config = ExecutionConfig(
            argv=argv,
            cwd=resolved_cwd,
            env=self._filter_env(env),
            input=None,  # stdin is disabled
            timeout_policy=timeout_policy,
            output_limits=self._output_limits,
            backend=BackendType.HOST,
        )

        # Execute with supervisor
        async for chunk in self._supervisor.execute(config):
            yield chunk

    async def cancel_execution(self, execution_id: str) -> None:
        """Cancel a running execution.

        Args:
            execution_id: Execution identifier (process handle ID)

        Raises:
            ToolValidationError: If execution_id is invalid
            ToolProcessError: If cancellation fails
        """
        try:
            await self._supervisor.cancel(execution_id)
        except ValueError as e:
            raise ToolValidationError(
                message=str(e),
                details={"execution_id": execution_id},
            ) from e
        except Exception as e:
            raise ToolProcessError(
                message=f"Failed to cancel execution: {e}",
                details={"execution_id": execution_id, "error": str(e)},
            ) from e

    async def check_process_status(self, handle_id: str) -> dict[str, Any]:
        """Check status of a LOCCA-owned process by handle.

        Args:
            handle_id: Process handle ID

        Returns:
            Dict with process status information

        Raises:
            ToolValidationError: If handle not found
        """
        from datetime import UTC, datetime

        handle = await self._supervisor.get_handle(handle_id)
        if handle is None:
            raise ToolValidationError(
                message=f"Process handle not found: {handle_id}",
                details={"handle_id": handle_id},
            )

        # Calculate elapsed time
        elapsed = None
        if handle.start_time:
            elapsed = (datetime.now(UTC) - handle.start_time).total_seconds()

        # Calculate idle time
        idle = None
        if handle.last_activity:
            idle = (datetime.now(UTC) - handle.last_activity).total_seconds()

        return {
            "handle_id": handle.handle_id,
            "state": handle.state.value,
            "return_code": handle.return_code,
            "exit_signal": handle.exit_signal,
            "os_pid": handle.os_pid,
            "backend": handle.backend.value,
            "start_time": handle.start_time.isoformat() if handle.start_time else None,
            "last_activity": handle.last_activity.isoformat()
            if handle.last_activity
            else None,
            "elapsed_seconds": elapsed,
            "idle_seconds": idle,
            "resource_usage": handle.resource_usage,
            "metadata": handle.metadata,
        }

    async def query_repository(
        self,
        query_type: str,
        params: dict[str, Any],
    ) -> Any:
        """Query repository index (symbols, etc.).

        Args:
            query_type: Type of query (e.g., "search_symbols")
            params: Query parameters

        Returns:
            Query results

        Raises:
            ToolBackendError: If workspace manager unavailable
            ToolValidationError: If query type unknown
        """
        if self._workspace_manager is None:
            raise ToolBackendError(
                message="Workspace manager not available",
            )

        if query_type == "search_symbols":
            return await self._workspace_manager.search_symbols(params)
        elif query_type == "get_symbol":
            return await self._workspace_manager.get_symbol(params)
        else:
            raise ToolValidationError(
                message=f"Unknown query type: {query_type}",
                details={"query_type": query_type},
            )

    async def search_text(
        self,
        query: str,
        path: Path,
        pattern: str = "literal",
        includes: list[str] | None = None,
        excludes: list[str] | None = None,
        case_sensitive: bool = False,
        context_lines: int = 2,
        max_matches: int = 200,
        max_bytes: int = 262144,
        include_ignored: bool = False,
    ) -> Any:
        """Search for text in files.

        Args:
            query: Search query (literal or regex)
            path: Search path relative to workspace root
            pattern: Pattern type (literal or regex)
            includes: Glob patterns to include
            excludes: Glob patterns to exclude
            case_sensitive: Case-sensitive search
            context_lines: Number of context lines before/after match
            max_matches: Maximum number of matches to return
            max_bytes: Maximum output bytes
            include_ignored: Include ignored files

        Returns:
            Search results

        Raises:
            ToolBackendError: If search fails
        """
        workspace_root = self.get_workspace_root()
        search_path = self._resolve_path(path)

        ignore_files: list[str] | None = None
        if self._config_manager is not None:
            ignore_files = (
                self._config_manager.global_config.tools_platform.ignore_policy.files
            )

        # Try ripgrep first, fall back to Python
        try:
            return await search_with_ripgrep(
                query=query,
                path=search_path,
                pattern=pattern,
                includes=includes or [],
                excludes=excludes or [],
                case_sensitive=case_sensitive,
                context_lines=context_lines,
                max_matches=max_matches,
                max_bytes=max_bytes,
                workspace_root=workspace_root,
                include_ignored=include_ignored,
                ignore_files=ignore_files,
            )
        except (FileNotFoundError, Exception):
            # Fall back to Python implementation
            return await search_with_python(
                query=query,
                path=search_path,
                pattern=pattern,
                includes=includes or [],
                excludes=excludes or [],
                case_sensitive=case_sensitive,
                context_lines=context_lines,
                max_matches=max_matches,
                max_bytes=max_bytes,
                workspace_root=workspace_root,
                include_ignored=include_ignored,
                ignore_files=ignore_files,
            )

    async def match_paths(
        self,
        pattern: str,
        path: Path,
        include_hidden: bool = False,
        include_ignored: bool = False,
        file_type: str | None = None,
        max_results: int = 200,
    ) -> Any:
        """Match files using glob patterns.

        Args:
            pattern: Glob pattern to match files
            path: Search path relative to workspace root
            include_hidden: Include hidden files/directories
            include_ignored: Include git-ignored files
            file_type: Filter by file extension
            max_results: Maximum number of results to return

        Returns:
            Matched paths

        Raises:
            ToolBackendError: If matching fails
            ToolPathViolationError: If path is outside workspace
        """
        workspace_root = self.get_workspace_root()
        search_path = self._resolve_path(path)

        ignore_files: list[str] | None = None
        if self._config_manager is not None:
            ignore_files = (
                self._config_manager.global_config.tools_platform.ignore_policy.files
            )

        return await match_paths(
            pattern=pattern,
            search_path=search_path,
            workspace_root=workspace_root,
            include_hidden=include_hidden,
            include_ignored=include_ignored,
            file_type=file_type,
            max_results=max_results,
            ignore_files=ignore_files,
        )

    async def replace_exact(
        self,
        path: str,
        old_text: str,
        new_text: str,
        session_id: str,
        expected_match_count: int = 1,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Implementation of replace_exact tool.

        Args:
            path: Path to file relative to workspace root
            old_text: Exact text to replace (must match exactly)
            new_text: New text to replace with
            expected_match_count: Expected number of old_text matches
            session_id: Session ID for tracking
            turn_id: Turn ID for tracking

        Returns:
            Dict with change details
        """
        # Resolve and validate path
        # Let ToolPathViolationError propagate - gateway will handle it
        resolved_path = self._workspace_manager.resolve_path(Path(path))
        rel_path = self._workspace_manager.get_relative_path(resolved_path)

        # Check if file exists
        if not resolved_path.exists():
            raise ToolPathViolationError(
                f"File not found: {path}",
                details={"path": path},
            )

        # Read current content (raw disk bytes for hash; normalized for matching).
        try:
            current_content = self._workspace_manager.read_file_raw(resolved_path)
        except OSError as e:
            return {
                "success": False,
                "match_count": 0,
                "before_hash": "",
                "after_hash": "",
                "before_size": 0,
                "after_size": 0,
                "before_lines": 0,
                "after_lines": 0,
                "error": f"Failed to read file: {e}",
            }

        # Detect original line endings so disk preserves them after the edit.
        original_ending = self._workspace_manager.detect_line_ending(current_content)

        # Normalize disk content and LLM-provided text to LF so the edit matches
        # regardless of what line endings the LLM emitted.
        current_content_norm = self._workspace_manager.normalize_for_match(
            current_content
        )
        old_text_norm = self._workspace_manager.normalize_for_match(old_text)
        new_text_norm = self._workspace_manager.normalize_for_match(new_text)

        # Compute current hash on raw disk bytes (baseline stays raw).
        current_hash = hashlib.sha256(current_content.encode("utf-8")).hexdigest()

        # Resolve edit match with trailing-newline fallback passes.
        edit = self._resolve_edit(
            current_content_norm,
            old_text_norm,
            new_text_norm,
            expected_match_count,
        )
        new_content_norm = edit.new_content_norm
        match_count = expected_match_count if edit.applied else edit.actual_count

        # Restore original line endings before writing to disk so the file's
        # native style is preserved. Hash computed on disk-bound content.
        new_content = self._workspace_manager.restore_line_endings(
            new_content_norm, original_ending
        )
        new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()

        # Validate against the workspace file_versions baseline (populated by reads).
        # The LLM never passes hashes; conflict detection lives in WorkspaceManager.
        conflict_report = self._workspace_manager.validate_write_version(
            file_path=rel_path,
            proposed_hash=new_hash,
            proposed_content=new_content,
            session_id=session_id,
        )
        if conflict_report.conflict_detected:
            # Update baseline to current disk state so retry works
            try:
                self._workspace_manager.observe_file(
                    rel_path, source=ObservationSource.READ, session_id=session_id
                )
            except Exception as e:
                logger.debug("Failed to observe file", error=str(e))

            return {
                "success": False,
                "match_count": 0,
                "before_hash": conflict_report.base_hash,
                "after_hash": "",
                "before_size": len(current_content.encode("utf-8")),
                "after_size": 0,
                "before_lines": len(current_content.splitlines()),
                "after_lines": 0,
                "error": conflict_report.message,
                "conflict": conflict_report.model_dump(),
                "external_diff": conflict_report.external_diff,
                "proposed_diff": conflict_report.proposed_diff,
                "recommendation": conflict_report.recommendation,
                "changed": False,
                "diff": None,
            }

        # If the edit could not be applied cleanly and there was no external
        # conflict, report a plain match-count mismatch.
        if not edit.applied:
            hint = self._workspace_manager.generate_failure_hint(
                content_lf=current_content_norm,
                old_text_lf=old_text_norm,
                expected=expected_match_count,
                actual=edit.actual_count,
            )
            return {
                "success": False,
                "match_count": edit.actual_count,
                "before_hash": current_hash,
                "after_hash": "",
                "before_size": len(current_content.encode("utf-8")),
                "after_size": 0,
                "before_lines": len(current_content.splitlines()),
                "after_lines": 0,
                "error": hint,
                "changed": False,
                "diff": None,
            }

        # Capture checkpoint before write (first write in this turn)
        if turn_id and session_id:
            self._workspace_manager.capture_checkpoint(
                session_id=session_id,
                turn_id=turn_id,
                file_path=rel_path,
                content=current_content,  # Store current content as checkpoint
            )

        self._workspace_manager.begin_transaction()

        try:
            # Perform atomic write (baseline derived from file_versions internally)
            # Baseline will be updated to new content after successful write
            _before_obs, after_obs = self._workspace_manager.atomic_write(
                file_path=rel_path,
                content=new_content,
                session_id=session_id,
            )
            self._workspace_manager.commit_transaction()

            # Compute diff
            diff = "".join(
                difflib.unified_diff(
                    current_content.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile="before",
                    tofile="after",
                )
            )
            normalized_diff = self._workspace_manager.normalize_for_match(diff)
            changed = current_hash != after_obs.hash

            return {
                "success": True,
                "match_count": match_count,
                "before_hash": current_hash,
                "after_hash": after_obs.hash,
                "before_size": len(current_content.encode("utf-8")),
                "after_size": len(new_content.encode("utf-8")),
                "before_lines": len(current_content.splitlines()),
                "after_lines": len(new_content.splitlines()),
                "error": None,
                "changed": changed,
                "diff": normalized_diff if changed else None,
            }

        except ToolPathViolationError:
            self._workspace_manager.rollback_transaction()
            raise
        except ValueError as e:
            self._workspace_manager.rollback_transaction()
            return {
                "success": False,
                "match_count": match_count,
                "before_hash": current_hash,
                "after_hash": "",
                "before_size": len(current_content.encode("utf-8")),
                "after_size": 0,
                "before_lines": len(current_content.splitlines()),
                "after_lines": 0,
                "error": str(e),
                "changed": False,
                "diff": None,
            }

    def _resolve_edit(
        self,
        current_content_norm: str,
        old_text_norm: str,
        new_text_norm: str,
        expected_match_count: int,
    ) -> _EditResult:
        """Resolve an exact-text edit with trailing-newline fallback passes.

        Pass 1 attempts a strict LF-normalized match. Pass 2 tries two trailing
        newline fallback cases when the strict count does not match expectations.
        """
        actual_count = current_content_norm.count(old_text_norm)

        # Pass 1: Strict match in LF space.
        if actual_count == expected_match_count:
            return _EditResult(
                new_content_norm=current_content_norm.replace(
                    old_text_norm, new_text_norm, expected_match_count
                ),
                applied=True,
                actual_count=actual_count,
            )

        # Pass 2: Trailing newline mismatch fallback.
        case_a = self._match_fallback_case_a(
            current_content_norm, old_text_norm, new_text_norm, expected_match_count
        )
        if case_a is not None:
            return case_a

        case_b = self._match_fallback_case_b(
            current_content_norm, old_text_norm, new_text_norm, expected_match_count
        )
        if case_b is not None:
            return case_b

        return _EditResult(
            new_content_norm=current_content_norm,
            applied=False,
            actual_count=actual_count,
        )

    def _match_fallback_case_a(
        self,
        current_content_norm: str,
        old_text_norm: str,
        new_text_norm: str,
        expected_match_count: int,
    ) -> _EditResult | None:
        # Case A: old_text missing trailing \n but target has \n.
        if (
            not old_text_norm.endswith("\n")
            and current_content_norm.count(old_text_norm + "\n") == expected_match_count
        ):
            adjusted_old_a = old_text_norm + "\n"
            adjusted_new_a = (
                new_text_norm if new_text_norm.endswith("\n") else new_text_norm + "\n"
            )
            if current_content_norm.count(adjusted_old_a) == expected_match_count:
                return _EditResult(
                    new_content_norm=current_content_norm.replace(
                        adjusted_old_a, adjusted_new_a, expected_match_count
                    ),
                    applied=True,
                    actual_count=current_content_norm.count(old_text_norm),
                )
        return None

    def _match_fallback_case_b(
        self,
        current_content_norm: str,
        old_text_norm: str,
        new_text_norm: str,
        expected_match_count: int,
    ) -> _EditResult | None:
        # Case B: old_text has extraneous trailing \n.
        if (
            old_text_norm.endswith("\n")
            and current_content_norm.count(old_text_norm[:-1]) == expected_match_count
        ):
            adjusted_old_b = old_text_norm[:-1]
            adjusted_new_b = (
                new_text_norm[:-1] if new_text_norm.endswith("\n") else new_text_norm
            )
            if current_content_norm.count(adjusted_old_b) == expected_match_count:
                return _EditResult(
                    new_content_norm=current_content_norm.replace(
                        adjusted_old_b, adjusted_new_b, expected_match_count
                    ),
                    applied=True,
                    actual_count=current_content_norm.count(old_text_norm),
                )
        return None

    async def create_file(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
        session_id: str | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Implementation of create_file tool.

        Args:
            path: Path to file relative to workspace root
            content: File content to write
            overwrite: Overwrite the file if it already exists
            session_id: Optional session ID for tracking
            turn_id: Optional turn ID for checkpoint tracking

        Returns:
            Dict with change details
        """
        # Resolve and validate path
        # Let ToolPathViolationError propagate - gateway will handle it
        resolved_path = self._workspace_manager.resolve_path(Path(path))
        rel_path = self._workspace_manager.get_relative_path(resolved_path)

        # Capture checkpoint before create (first create in this turn)
        if turn_id and session_id:
            # For create, checkpoint is None (deletion marker for undo)
            self._workspace_manager.capture_checkpoint(
                session_id=session_id,
                turn_id=turn_id,
                file_path=rel_path,
                content=None,  # None = file didn't exist
            )

        self._workspace_manager.begin_transaction()

        try:
            if resolved_path.exists():
                if not overwrite:
                    raise ToolPathViolationError(
                        f"File already exists: {path}",
                        details={"path": path},
                    )
                # Overwrite: remove the existing file within this transaction so
                # atomic_create can replace it.
                self._workspace_manager.transaction_manager.atomic_delete(
                    resolved_path=resolved_path,
                    create_snapshot=False,
                )

            after_observation = self._workspace_manager.atomic_create(
                file_path=rel_path,
                content=content,
            )

            self._workspace_manager.commit_transaction()

            # Verify the result
            verification = self._read_file_verification(resolved_path)
            if verification["hash"] != after_observation.hash:
                return {
                    "success": False,
                    "path": path,
                    "after_hash": verification["hash"],
                    "after_size": verification["size"],
                    "after_lines": verification["lines"],
                    "error": "Verification failed: written content doesn't match expected hash",
                }

            return {
                "success": True,
                "path": path,
                "after_hash": after_observation.hash,
                "after_size": after_observation.size,
                "after_lines": len(content.splitlines()),
                "error": None,
            }

        except ToolPathViolationError:
            self._workspace_manager.rollback_transaction()
            raise
        except FileExistsError as e:
            self._workspace_manager.rollback_transaction()
            return {
                "success": False,
                "path": path,
                "after_hash": "",
                "after_size": 0,
                "after_lines": 0,
                "error": str(e),
            }

        except Exception as e:
            self._workspace_manager.rollback_transaction()
            return {
                "success": False,
                "path": path,
                "after_hash": "",
                "after_size": 0,
                "after_lines": 0,
                "error": f"Unexpected error: {e}",
            }

    def _read_file_verification(self, resolved_path: Path) -> dict[str, Any]:
        verification_content = self._workspace_manager.read_file_raw(resolved_path)
        return {
            "hash": hashlib.sha256(verification_content.encode("utf-8")).hexdigest(),
            "size": len(verification_content.encode("utf-8")),
            "lines": len(verification_content.splitlines()),
        }

    async def delete_file(
        self,
        path: str,
        recursive: bool = False,
        dry_run: bool = False,
        session_id: str | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Implementation of delete_file tool.

        Args:
            path: Path to file relative to workspace root
            recursive: Delete directories recursively
            dry_run: Preview deletion without executing
            session_id: Optional session ID for tracking
            turn_id: Optional turn ID for checkpoint tracking

        Returns:
            Dict with change details
        """
        # Resolve and validate path
        # Let ToolPathViolationError propagate - gateway will handle it
        resolved_path = self._workspace_manager.resolve_path(Path(path))
        rel_path = self._workspace_manager.get_relative_path(resolved_path)

        # Check if file exists
        if not resolved_path.exists():
            raise ToolPathViolationError(
                f"File not found: {path}",
                details={"path": path},
            )

        # Capture checkpoint before delete (first delete in this turn)
        if turn_id and session_id:
            # Read current content for checkpoint
            try:
                content = resolved_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = None
            self._workspace_manager.capture_checkpoint(
                session_id=session_id,
                turn_id=turn_id,
                file_path=rel_path,
                content=content,
            )

        # Dry run: preview what would be deleted without modifying anything.
        if dry_run:
            would_delete: list[str] = [rel_path]
            if recursive and resolved_path.is_dir():
                would_delete = [
                    str(p.relative_to(self._workspace_root)).replace("\\", "/")
                    for p in resolved_path.rglob("*")
                ]
            return {
                "success": True,
                "path": path,
                "error": None,
                "would_delete": would_delete,
            }

        self._workspace_manager.begin_transaction()

        try:
            if recursive and resolved_path.is_dir():
                import shutil

                shutil.rmtree(resolved_path)
            else:
                # Perform atomic delete (snapshot created for rollback)
                self._workspace_manager.atomic_delete(file_path=rel_path)

            self._workspace_manager.commit_transaction()

        except ToolPathViolationError:
            self._workspace_manager.rollback_transaction()
            raise
        except Exception as e:
            # Rollback if we started the transaction
            self._workspace_manager.rollback_transaction()

            return {
                "success": False,
                "path": path,
                "error": str(e),
                "would_delete": None,
            }

        # Verify the result
        if resolved_path.exists():
            return {
                "success": False,
                "path": path,
                "error": "Verification failed: file still exists after deletion",
                "would_delete": None,
            }

        return {
            "success": True,
            "path": path,
            "error": None,
            "would_delete": None,
        }

    def get_workspace_root(self) -> Path:
        """Get the workspace root path.

        Returns:
            Path to workspace root
        """
        return self._workspace_root

    def resolve_path(self, path: Path) -> Path:
        """Resolve a path relative to workspace root.

        Args:
            path: Path to resolve (can be relative or absolute)

        Returns:
            Absolute path within workspace

        Raises:
            ToolPathViolationError: If path is outside workspace
        """
        # Use WorkspaceManager if available, otherwise use local logic
        if self._workspace_manager is not None:
            return self._workspace_manager.resolve_path(path)

        # Fallback to local resolution
        return self._resolve_path_local(path)

    def _resolve_path_local(self, path: Path) -> Path:
        """Resolve a path relative to workspace root (local implementation).

        Args:
            path: Path to resolve (can be relative or absolute)

        Returns:
            Absolute path within workspace

        Raises:
            ToolPathViolationError: If path is outside workspace
        """
        # Convert to absolute path
        if path.is_absolute():
            resolved = path
        else:
            resolved = self._workspace_root / path

        # Resolve to normalize
        resolved = resolved.resolve()

        # Check if within workspace
        try:
            resolved.relative_to(self._workspace_root)
        except ValueError as e:
            raise ToolPathViolationError(
                message=f"Path is outside workspace: {path}",
                details={
                    "path": str(path),
                    "resolved": str(resolved),
                    "workspace_root": str(self._workspace_root),
                },
            ) from e

        return resolved

    def get_backend_type(self) -> str:
        """Get the backend type identifier."""
        return BackendType.HOST

    async def initialize(self) -> None:
        """Initialize the backend."""
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the backend."""
        self._initialized = False

    def is_available(self) -> bool:
        """Check if backend is available."""
        return self._initialized and self._workspace_root.exists()

    def _resolve_path(self, path: Path) -> Path:
        """Resolve a path relative to workspace root.

        Args:
            path: Path to resolve (can be relative or absolute)

        Returns:
            Absolute path within workspace

        Raises:
            ToolPathViolationError: If path is outside workspace
        """
        # Convert to absolute path
        if path.is_absolute():
            resolved = path
        else:
            resolved = self._workspace_root / path

        # Resolve to normalize
        resolved = resolved.resolve()

        # Check if within workspace
        try:
            resolved.relative_to(self._workspace_root)
        except ValueError as e:
            raise ToolPathViolationError(
                message=f"Path is outside workspace: {path}",
                details={
                    "path": str(path),
                    "resolved": str(resolved),
                    "workspace_root": str(self._workspace_root),
                },
            ) from e

        return resolved
