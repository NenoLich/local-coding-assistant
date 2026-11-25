# pyrefly: ignore-errors
"""Docker execution backend implementation.

Operates inside a Docker container, maps container paths to workspace-relative
paths, preserves streamed process output, and uses the same ToolOutputChunk
format as host. Receives WorkspaceManager and RepositoryContextService as
injected capabilities.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_coding_assistant.tools_platform.backends.base import ExecutionBackend
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
from local_coding_assistant.tools_platform.supervisor.docker_transport import (
    DockerProcessTransport,
)

if TYPE_CHECKING:
    from local_coding_assistant.core.protocols import IConfigManager
    from local_coding_assistant.tools_platform.workspace import WorkspaceManager

try:
    from docker.models.containers import Container
except ImportError:
    Container = object


class DockerBackend(ExecutionBackend):
    """Docker execution backend.

    Executes tools inside a Docker container with workspace mapping.
    Receives RepositoryContextService for symbol search operations.
    """

    def __init__(
        self,
        workspace_root: Path,
        container: Container | None = None,
        workspace_manager: WorkspaceManager | None = None,
        config_manager: IConfigManager | None = None,
    ) -> None:
        """Initialize the Docker backend.

        Args:
            workspace_root: Path to the workspace root directory
            container: Docker container instance (optional, will be created if None)
            workspace_manager: WorkspaceManager for path validation and symbol operations
            config_manager: Config manager for dynamic configuration resolution
        """
        self._workspace_root = Path(workspace_root)
        self._container = container
        self._workspace_manager = workspace_manager
        self._config_manager = config_manager

        # Create Docker-specific transport and pass to supervisor
        transport = DockerProcessTransport(
            container=container,
            workspace_root=workspace_root,
        )
        self._supervisor = ProcessSupervisor(transport=transport)
        self._initialized = False

    @property
    def _timeout_policy(self) -> TimeoutPolicy:
        """Get timeout policy from config."""
        if self._config_manager is None:
            return TimeoutPolicy()
        tc = self._config_manager.global_config.tools_platform
        return TimeoutPolicy(
            soft_timeout=tc.timeout.soft_seconds,
            hard_timeout=tc.timeout.hard_seconds,
            idle_timeout=tc.timeout.idle_seconds,
            heartbeat_interval=tc.timeout.heartbeat_seconds,
            cancellation_grace=tc.timeout.cancellation_grace_seconds,
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

    async def read_file(
        self,
        path: Path,
        start_line: int | None = None,
        end_line: int | None = None,
        byte_limit: int | None = None,
    ) -> str:
        """Read file content from container with optional line range and byte limit.

        Args:
            path: Path to file (relative to workspace root)
            start_line: Optional start line (1-indexed)
            end_line: Optional end line (1-indexed)
            byte_limit: Maximum bytes to read

        Returns:
            File content

        Raises:
            ToolExecutionError: If read fails
        """
        container_path = self._map_to_container_path(path)

        # Use cat command to read file
        command = ["cat", str(container_path)]
        if start_line is not None or end_line is not None:
            # Use sed for line ranges
            if start_line is not None and end_line is not None:
                command = [
                    "sed",
                    "-n",
                    f"{start_line},{end_line}p",
                    str(container_path),
                ]
            elif start_line is not None:
                command = ["sed", "-n", f"{start_line},$p", str(container_path)]
            elif end_line is not None:
                command = ["sed", "-n", f"1,{end_line}p", str(container_path)]

        # Collect output from streaming execution
        stdout = ""
        async for chunk in self.execute_command(
            argv=command,
            cwd=Path("/workspace"),
        ):
            if chunk.channel == "stdout":
                stdout += chunk.payload

        # Apply byte limit if specified
        if byte_limit is not None and len(stdout.encode("utf-8")) > byte_limit:
            stdout = stdout.encode("utf-8")[:byte_limit].decode(
                "utf-8", errors="ignore"
            )

        return stdout

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

        resolved_path = self._workspace_manager.resolve_path(path)
        rel_path = self._workspace_manager.get_relative_path(resolved_path)
        return self._workspace_manager.observe_file(
            rel_path, source=ObservationSource.READ, session_id=session_id
        )

    async def create_file(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Implementation of create_file tool in container.

        Args:
            path: Path to file relative to workspace root
            content: File content to write
            overwrite: Overwrite the file if it already exists

        Returns:
            Dict with change details
        """
        if self._workspace_manager is None:
            return {
                "success": False,
                "path": path,
                "after_hash": "",
                "after_size": 0,
                "after_lines": 0,
                "error": "Workspace manager not available",
            }

        # Resolve and validate path
        # Let ToolPathViolationError propagate - gateway will handle it
        resolved_path = self._workspace_manager.resolve_path(Path(path))
        _rel_path = self._workspace_manager.get_relative_path(resolved_path)

        # Check if file already exists in container
        container_path = self._map_to_container_path(Path(path))
        check_command = ["test", "-f", str(container_path)]
        try:
            async for chunk in self.execute_command(
                argv=check_command,
                cwd=Path("/workspace"),
            ):
                if chunk.channel == "status":
                    status_data = chunk.payload
                    if status_data.get("return_code") == 0:
                        return {
                            "success": False,
                            "path": path,
                            "after_hash": "",
                            "after_size": 0,
                            "after_lines": 0,
                            "error": f"File already exists: {path}",
                        }
        except Exception:  # noqa S110
            # File doesn't exist, proceed with creation
            pass

        # Write file using tee
        write_command = ["tee", str(container_path)]
        stdout = ""
        try:
            async for chunk in self.execute_command(
                argv=write_command,
                cwd=Path("/workspace"),
            ):
                if chunk.channel == "stdout":
                    stdout += chunk.payload

            # Calculate hash
            after_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            after_size = len(content.encode("utf-8"))
            after_lines = len(content.splitlines())

            return {
                "success": True,
                "path": path,
                "after_hash": after_hash,
                "after_size": after_size,
                "after_lines": after_lines,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "path": path,
                "after_hash": "",
                "after_size": 0,
                "after_lines": 0,
                "error": f"Failed to create file: {e}",
            }

    async def delete_file(
        self,
        path: str,
    ) -> dict[str, Any]:
        """Implementation of delete_file tool in container.

        Args:
            path: Path to file relative to workspace root

        Returns:
            Dict with change details
        """
        if self._workspace_manager is None:
            return {
                "success": False,
                "path": path,
                "before_hash": "",
                "before_size": 0,
                "before_lines": 0,
                "error": "Workspace manager not available",
            }

        # Resolve and validate path
        # Let ToolPathViolationError propagate - gateway will handle it
        resolved_path = self._workspace_manager.resolve_path(Path(path))
        _rel_path = self._workspace_manager.get_relative_path(resolved_path)

        # Read current content to get hash and line count
        container_path = self._map_to_container_path(Path(path))
        try:
            current_content = await self.read_file(Path(path))
            current_hash = hashlib.sha256(current_content.encode("utf-8")).hexdigest()
        except Exception as e:
            return {
                "success": False,
                "path": path,
                "before_hash": "",
                "before_size": 0,
                "before_lines": 0,
                "error": f"Failed to read file: {e}",
            }

        # Delete file
        delete_command = ["rm", str(container_path)]
        try:
            async for _chunk in self.execute_command(
                argv=delete_command,
                cwd=Path("/workspace"),
            ):
                pass  # Wait for completion

            return {
                "success": True,
                "path": path,
                "before_hash": current_hash,
                "before_size": len(current_content.encode("utf-8")),
                "before_lines": len(current_content.splitlines()),
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "path": path,
                "before_hash": current_hash,
                "before_size": len(current_content.encode("utf-8")),
                "before_lines": len(current_content.splitlines()),
                "error": f"Failed to delete file: {e}",
            }

    async def replace_exact(
        self,
        path: str,
        old_text: str,
        new_text: str,
        expected_match_count: int = 1,
    ) -> dict[str, Any]:
        """Implementation of replace_exact tool in container.

        Args:
            path: Path to file relative to workspace root
            old_text: Exact text to replace (must match exactly)
            new_text: New text to replace with
            expected_match_count: Expected number of old_text matches

        Returns:
            Dict with change details
        """
        if self._workspace_manager is None:
            return {
                "success": False,
                "match_count": 0,
                "before_hash": "",
                "after_hash": "",
                "before_size": 0,
                "after_size": 0,
                "before_lines": 0,
                "after_lines": 0,
                "error": "Workspace manager not available",
            }

        # Resolve and validate path
        # Let ToolPathViolationError propagate - gateway will handle it
        resolved_path = self._workspace_manager.resolve_path(Path(path))
        _rel_path = self._workspace_manager.get_relative_path(resolved_path)

        # Read current content
        try:
            current_content = await self.read_file(Path(path))
        except Exception as e:
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
                "changed": False,
                "diff": None,
            }

        # Compute current hash
        current_hash = hashlib.sha256(current_content.encode("utf-8")).hexdigest()
        base_hash = current_hash

        # Count matches
        match_count = current_content.count(old_text)

        # Verify match count
        if match_count != expected_match_count:
            return {
                "success": False,
                "match_count": match_count,
                "before_hash": base_hash,
                "after_hash": "",
                "before_size": len(current_content.encode("utf-8")),
                "after_size": 0,
                "before_lines": len(current_content.splitlines()),
                "after_lines": 0,
                "error": f"Match count mismatch: expected {expected_match_count}, found {match_count}",
                "changed": False,
                "diff": None,
            }

        # Perform replacement
        new_content = current_content.replace(old_text, new_text)
        _new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()

        # Write new content
        container_path = self._map_to_container_path(Path(path))
        write_command = ["tee", str(container_path)]
        try:
            async for chunk in self.execute_command(
                argv=write_command,
                cwd=Path("/workspace"),
            ):
                if chunk.channel == "stdout":
                    # Feed the new content to tee
                    pass  # This needs stdin support, will be enhanced later

            # For now, use a simpler approach with cat
            # TODO: Implement proper stdin support
            return {
                "success": False,
                "match_count": match_count,
                "before_hash": base_hash,
                "after_hash": "",
                "before_size": len(current_content.encode("utf-8")),
                "after_size": 0,
                "before_lines": len(current_content.splitlines()),
                "after_lines": 0,
                "error": "stdin support not yet implemented for Docker backend",
                "changed": False,
                "diff": None,
            }

        except Exception as e:
            return {
                "success": False,
                "match_count": match_count,
                "before_hash": base_hash,
                "after_hash": "",
                "before_size": len(current_content.encode("utf-8")),
                "after_size": 0,
                "before_lines": len(current_content.splitlines()),
                "after_lines": 0,
                "error": f"Failed to write file: {e}",
                "changed": False,
                "diff": None,
            }

    async def execute_command(
        self,
        argv: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ToolOutputChunk]:
        """Execute a command in container with streaming output.

        Args:
            argv: Command and arguments
            cwd: Working directory (relative to workspace root)
            env: Environment variables
            timeout_seconds: Timeout in seconds (overrides default)

        Yields:
            ToolOutputChunk with stdout/stderr/status

        Raises:
            ToolExecutionError: If execution fails
        """
        # Resolve working directory to container path
        resolved_cwd = None
        if cwd is not None:
            resolved_cwd = self._map_to_container_path(cwd)

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
            env=env,
            input=None,  # stdin is disabled
            timeout_policy=timeout_policy,
            output_limits=self._output_limits,
            backend=BackendType.DOCKER,
        )

        # Execute with supervisor
        async for chunk in self._supervisor.execute(config):
            yield chunk

    async def cancel_execution(self, execution_id: str) -> None:
        """Cancel a running execution in container.

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
        """Query repository index - not supported in Docker backend.

        Args:
            query_type: Type of query (e.g., "search_symbols")
            params: Query parameters

        Returns:
            Query results

        Raises:
            ToolBackendError: Repository queries not supported in Docker backend
        """
        raise ToolBackendError(
            message="Repository queries not supported in Docker backend",
        )

    async def search_text(  # noqa: C901
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
    ) -> Any:
        """Search for text in files using ripgrep in container.

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

        Returns:
            Search results

        Raises:
            ToolExecutionError: If search fails
        """
        container_path = self._map_to_container_path(path)

        # Build ripgrep command
        command = ["rg", query, str(container_path)]

        if pattern == "regex":
            command.append("--regex")
        else:
            command.append("--fixed-strings")

        if case_sensitive:
            command.append("--case-sensitive")
        else:
            command.append("--ignore-case")

        if context_lines:
            command.extend(["-C", str(context_lines)])

        if max_matches:
            command.extend(["-m", str(max_matches)])

        if includes:
            for include in includes:
                command.extend(["-g", include])

        if excludes:
            for exclude in excludes:
                command.extend(["-g", f"!{exclude}"])

        # Execute and collect output
        stdout = ""
        async for chunk in self.execute_command(
            argv=command,
            cwd=Path("/workspace"),
        ):
            if chunk.channel == "stdout":
                stdout += chunk.payload

        # Parse ripgrep output (simplified)
        # TODO: Implement proper parsing
        return {"output": stdout, "matches": []}

    async def match_paths(
        self,
        pattern: str,
        path: Path,
        include_hidden: bool = False,
        include_ignored: bool = False,
        file_type: str | None = None,
        max_results: int = 200,
    ) -> Any:
        """Match files using glob patterns in container.

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
            ToolExecutionError: If matching fails
        """
        container_path = self._map_to_container_path(path)

        # Use find command for glob matching
        command = ["find", str(container_path), "-type", "f"]

        if not include_hidden:
            command.extend(["!", "-name", ".*"])

        if file_type:
            command.extend(["-name", f"*.{file_type}"])

        if max_results:
            command.extend(["-print", "-quit"])  # Simplified

        # Execute and collect output
        stdout = ""
        async for chunk in self.execute_command(
            argv=command,
            cwd=Path("/workspace"),
        ):
            if chunk.channel == "stdout":
                stdout += chunk.payload

        # Parse output and map back to workspace-relative paths
        results = []
        for line in stdout.splitlines():
            if line:
                # Convert container path to workspace-relative
                workspace_path = self._map_from_container_path(Path(line))
                results.append(str(workspace_path))
                if len(results) >= max_results:
                    break

        return {"matches": results}

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

    def _map_to_container_path(self, path: Path) -> Path:
        """Map workspace-relative path to container path.

        Args:
            path: Workspace-relative path

        Returns:
            Container path
        """
        # Ensure path is relative to workspace
        if path.is_absolute():
            path = path.relative_to(self._workspace_root)
        return Path("/workspace") / path

    def _map_from_container_path(self, container_path: Path) -> Path:
        """Map container path to workspace-relative path.

        Args:
            container_path: Container path

        Returns:
            Workspace-relative path
        """
        # Remove /workspace prefix
        if str(container_path).startswith("/workspace"):
            return Path(str(container_path)[len("/workspace") :].lstrip("/"))
        return container_path

    def get_backend_type(self) -> str:
        """Get the backend type identifier."""
        return BackendType.DOCKER

    async def initialize(self) -> None:
        """Initialize the backend (e.g., start container)."""
        if not self._initialized:
            # Container initialization will be handled by sandbox infrastructure
            self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the backend (e.g., stop container)."""
        # Container shutdown will be handled by sandbox infrastructure
        self._initialized = False

    def is_available(self) -> bool:
        """Check if backend is available."""
        try:
            import docker

            docker.from_env()
            return True
        except Exception:
            return False
