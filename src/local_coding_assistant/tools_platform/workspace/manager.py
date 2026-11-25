"""Workspace manager for path policy and transaction management.

This module provides the main WorkspaceManager class that:
- Enforces path policy (workspace root, allowed paths, traversal limits)
- Tracks file versions and detects conflicts (WORKSPACE = transaction + write integrity layer)
- Manages transactions with atomic operations
- Coordinates with file monitoring for ADVISORY invalidation signals only

RESPONSIBILITY SPLIT (per phase 5d):
- FileMonitoringService = change feed / event source (passive observer)
- WorkspaceManager = transaction + write integrity layer (authority for conflicts)
- ConflictDetector = version validator for active workspace state
- Repository reindexing callbacks remain attached to file monitoring, but they
  are not part of conflict detection.

File monitoring events are treated as advisory invalidation, not authority.
The final conflict decision comes from the workspace baseline check at write time.
"""

from __future__ import annotations

import fnmatch
import hashlib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_coding_assistant.tools_platform.contracts import (
    ConflictReport,
    FileObservation,
    ObservationSource,
)
from local_coding_assistant.tools_platform.exceptions import (
    ToolBackendError,
    ToolPathViolationError,
)
from local_coding_assistant.tools_platform.workspace.conflicts import (
    ConflictDetectionConfig,
    ConflictDetector,
)
from local_coding_assistant.tools_platform.workspace.content_store import (
    BaselineStore,
    BaselineStoreConfig,
    CheckpointStore,
    CheckpointStoreConfig,
)
from local_coding_assistant.tools_platform.workspace.line_normalization import (
    LineEnding,
    LineNormalizer,
    detect_line_ending,
    generate_failure_hint,
)
from local_coding_assistant.tools_platform.workspace.models import (
    PathPolicy,
    WorkspaceState,
)
from local_coding_assistant.tools_platform.workspace.snapshots import (
    SnapshotManager,
    SnapshotStorageConfig,
)
from local_coding_assistant.tools_platform.workspace.transactions import (
    TransactionConfig,
    TransactionManager,
)
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.core.protocols import IConfigManager
    from local_coding_assistant.repository.service import RepositoryContextService
    from local_coding_assistant.storage import IExecutionStore


logger = get_logger("tools_platform.workspace.manager")


def _hash_config(config_manager: IConfigManager) -> str:
    """Compute a stable hash of the effective tools_platform config.

    Used as the workspace record's config_hash so config drift can be detected
    against the stored value instead of silently invalidating history.

    Args:
        config_manager: Config manager for dynamic configuration resolution

    Returns:
        A 16-char hex digest of the effective tools_platform config
    """
    config = config_manager.global_config.tools_platform
    payload = {
        "backend": config.backend,
        "workspace_root": config.workspace_root,
        "allowed_patterns": list(config.path_policy.allowed_patterns),
        "denied_patterns": list(config.path_policy.denied_patterns),
        "allow_symlinks": config.path_policy.allow_symlinks,
        "max_depth": config.path_policy.max_depth,
        "default_policy": config.default_policy,
        "allow_network": config.allow_network,
        "non_interactive": config.non_interactive,
        "max_output_bytes": config.max_output_bytes,
        "max_read_lines": config.max_read_lines,
        "max_search_matches": config.max_search_matches,
    }
    import json

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


class WorkspaceManager:
    """Manages workspace state, path policy, and transactions.

    This class is the central coordinator for workspace operations:
    - Validates all paths against policy
    - Tracks file versions for conflict detection
    - Manages transactions with atomic operations
    - Coordinates with file monitoring

    Workspace identity is persisted (see :meth:`from_config`): the workspace_id
    is a stored UUID keyed by the resolved workspace root, so session resume and
    file-baseline lookups survive process restarts, path renames, symlinks and
    config drift. The config_hash recorded at creation time lets callers detect
    drift against the stored value instead of silently invalidating history.
    """

    def __init__(
        self,
        *,
        workspace_id: str,
        workspace_root: Path,
        path_policy: PathPolicy,
        config_manager: IConfigManager | None = None,
        repository_context_service: RepositoryContextService | None = None,
    ) -> None:
        """Initialize the workspace manager with explicit identity.

        Args:
            workspace_id: Persisted/stable workspace identifier
            workspace_root: Absolute path to the workspace root
            path_policy: Pre-built path policy (workspace_root must match)
            config_manager: Config manager for dynamic configuration resolution
            repository_context_service: Optional repository context service for
                symbol search operations
        """
        self._config_manager = config_manager
        self._repository_context_service = repository_context_service
        self.workspace_id = workspace_id
        self.workspace_root = workspace_root
        self.path_policy = path_policy

        # Line-ending normalizer for LLM-facing content (config-driven).
        # Baseline hashes and snapshots remain on raw disk bytes.
        self._line_normalizer = LineNormalizer(
            enabled=self._get_normalize_to_lf_enabled()
        )

        # Initialize content stores (baselines and checkpoints)
        self.baseline_store = BaselineStore(self._build_baseline_store_config())
        self.checkpoint_store = CheckpointStore(self._build_checkpoint_store_config())

        # Initialize snapshot manager
        self.snapshot_manager = SnapshotManager(self._build_snapshot_config())

        # Initialize transaction manager (baseline persistence is wired up in
        # init_identity once the store connection is open)
        self.transaction_manager = TransactionManager(
            workspace_id=self.workspace_id,
            workspace_root=self.workspace_root,
            snapshot_manager=self.snapshot_manager,
            config=TransactionConfig(),
            baseline_store=self.baseline_store,
        )

        # Initialize conflict detector
        self.conflict_detector = ConflictDetector(self._build_conflict_config())

        logger.info(
            "Initialized workspace manager",
            workspace_id=self.workspace_id,
            workspace_root=str(self.workspace_root),
        )

    @classmethod
    def from_config_manager(
        cls,
        config_manager: IConfigManager,
        repository_context_service: RepositoryContextService | None = None,
    ) -> WorkspaceManager:
        """Build a WorkspaceManager from a config manager (no DB round-trip).

        Convenience constructor that resolves the workspace root and path
        policy from config and assigns a process-local workspace_id. Tests and
        direct callers use this; the app path uses :meth:`from_config` /
        :meth:`init_identity` for persisted identity.

        Args:
            config_manager: Config manager for dynamic configuration resolution
            repository_context_service: Optional repository context service for
                symbol search operations

        Returns:
            A fully initialized WorkspaceManager
        """
        workspace_root = config_manager.path_manager.resolve_path(
            config_manager.global_config.tools_platform.workspace_root
        )
        path_policy = cls._build_path_policy_static(config_manager, workspace_root)
        return cls(
            workspace_id=str(uuid.uuid4()),
            workspace_root=workspace_root,
            path_policy=path_policy,
            config_manager=config_manager,
            repository_context_service=repository_context_service,
        )

    @classmethod
    def from_config_sync(
        cls,
        config_manager: IConfigManager,
        repository_context_service: RepositoryContextService | None = None,
    ) -> WorkspaceManager:
        """Build a WorkspaceManager synchronously (no DB round-trip).

        Used by ToolGateway.__init__ when constructed directly (e.g. tests,
        CLI tool commands). Workspace identity is not persisted here: callers
        that need persistence should use :meth:`from_config` /
        :meth:`init_identity` instead.

        Args:
            config_manager: Config manager for dynamic configuration resolution
            repository_context_service: Optional repository context service for
                symbol search operations

        Returns:
            A fully initialized WorkspaceManager
        """
        workspace_root = config_manager.path_manager.resolve_path(
            config_manager.global_config.tools_platform.workspace_root
        )
        workspace_id = str(uuid.uuid4())
        path_policy = cls._build_path_policy_static(config_manager, workspace_root)
        return cls(
            workspace_id=workspace_id,
            workspace_root=workspace_root,
            path_policy=path_policy,
            config_manager=config_manager,
            repository_context_service=repository_context_service,
        )

    async def init_identity(
        self,
        store: IExecutionStore | None = None,
        repository_context_service: RepositoryContextService | None = None,
    ) -> str:
        """Resolve the persisted workspace identity and re-seed dependent managers.

        Called after construction (from the async app-initialize step) so the
        gateway can be built synchronously by ``bootstrap`` while still ending
        up with a stable, persisted workspace_id. Reuses an existing record for
        the resolved root, otherwise creates one; the stored config_hash is
        refreshed so config drift is detectable rather than silently
        invalidating history.

        Args:
            store: Optional execution store for identity persistence
            repository_context_service: Optional repository context service for
                symbol search operations

        Returns:
            The resolved workspace_id
        """
        if repository_context_service is not None:
            self._repository_context_service = repository_context_service

        if store is None or self._config_manager is None:
            return self.workspace_id

        config_hash = _hash_config(self._config_manager)
        record = await store.get_workspace(str(self.workspace_root))
        if record is not None:
            self.workspace_id = record.id
            if record.config_hash != config_hash:
                await store.refresh_workspace_config_hash(
                    self.workspace_id, config_hash
                )
                logger.info(
                    "Workspace config drift detected; refreshed stored hash",
                    workspace_id=self.workspace_id,
                    stored_hash=record.config_hash,
                    current_hash=config_hash,
                )
        else:
            self.workspace_id = str(uuid.uuid4())
            await store.upsert_workspace(
                self.workspace_id, str(self.workspace_root), config_hash
            )
            logger.info(
                "Created persisted workspace identity",
                workspace_id=self.workspace_id,
                workspace_root=str(self.workspace_root),
                config_hash=config_hash,
            )

        # Re-seed the transaction manager with the resolved identity.
        self.transaction_manager = TransactionManager(
            workspace_id=self.workspace_id,
            workspace_root=self.workspace_root,
            snapshot_manager=self.snapshot_manager,
            baseline_store=self.baseline_store,
            config=TransactionConfig(),
        )

        # Run retention cleanup for baselines and checkpoints
        try:
            self.cleanup_workspace_state()
        except Exception as e:
            logger.warning(
                "Retention cleanup failed during init_identity", error=str(e)
            )

        return self.workspace_id

    async def search_symbols(self, params: dict[str, Any]) -> list[Any]:
        """Search for symbols via the repository context service.

        Args:
            params: Query parameters (query|pattern, symbol_types, file_path, limit)

        Returns:
            List of symbol search results

        Raises:
            ToolBackendError: If the repository context service is not available
        """
        if self._repository_context_service is None:
            raise ToolBackendError(
                message="Repository context service not available",
            )
        return self._repository_context_service.search_symbols(
            query=params.get("query", params.get("pattern", "")),
            symbol_types=params.get("symbol_types"),
            file_path=params.get("file_path"),
            limit=params.get("limit", 20),
        )

    async def get_symbol(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Get symbols by name via the repository context service.

        Supports multiple symbol returns when several symbols share a name
        (e.g. methods on different classes). Optional ``symbol_type`` and
        ``file_path`` filters narrow the result set.

        Args:
            params: Query parameters (name, symbol_type, file_path, limit)

        Returns:
            List of symbol detail dicts (possibly empty)

        Raises:
            ToolBackendError: If the repository context service is not available
        """
        if self._repository_context_service is None:
            raise ToolBackendError(
                message="Repository context service not available",
            )
        name = params.get("name", "")
        if not name:
            return []

        symbol_type = params.get("symbol_type")
        file_path = params.get("file_path", params.get("path"))
        limit = params.get("limit", 20)

        results = self._repository_context_service.get_symbol(
            name=name,
            symbol_type=symbol_type,
            file_path=file_path,
            limit=limit,
        )
        if not results:
            return []

        return [
            {
                "symbol_id": result.symbol_id,
                "file_id": result.file_id,
                "name": result.name,
                "symbol_type": str(result.symbol_type),
                "line_number": result.line_number,
                "file_path": result.file_path,
            }
            for result in results
        ]

    @staticmethod
    def _build_path_policy_static(
        config_manager: IConfigManager, workspace_root: Path
    ) -> PathPolicy:
        """Build path policy from config for a given resolved root."""
        config = config_manager.global_config.tools_platform
        return PathPolicy(
            workspace_root=workspace_root,
            allowed_patterns=list(config.path_policy.allowed_patterns),
            denied_patterns=list(config.path_policy.denied_patterns),
            allow_symlinks=config.path_policy.allow_symlinks,
            max_depth=config.path_policy.max_depth,
            require_relative=True,
        )

    def _build_snapshot_config(self) -> SnapshotStorageConfig:
        """Build snapshot storage config from config."""
        if self._config_manager is None:
            return SnapshotStorageConfig(
                snapshots_dir=self.workspace_root / ".locca" / "snapshots"
            )
        config = self._config_manager.global_config.tools_platform
        if config.snapshots.enabled:
            snapshots_dir = self._config_manager.path_manager.resolve_path(
                config.snapshots.directory
            )
            return SnapshotStorageConfig(
                snapshots_dir=snapshots_dir,
                max_snapshots=config.snapshots.max_snapshots,
                max_snapshot_age_days=config.snapshots.max_snapshot_age_days,
            )
        return SnapshotStorageConfig(
            snapshots_dir=self.workspace_root / ".locca" / "snapshots"
        )

    def _build_conflict_config(self) -> ConflictDetectionConfig:
        """Build conflict detection config from config."""
        if self._config_manager is None:
            return ConflictDetectionConfig()
        config = self._config_manager.global_config.tools_platform
        conflict_policy = config.concurrency.write_conflict_policy
        if conflict_policy not in ("deny", "ask", "allow"):
            conflict_policy = "deny"
        return ConflictDetectionConfig(conflict_policy=conflict_policy)

    def _get_normalize_to_lf_enabled(self) -> bool:
        """Get normalize_to_lf_line_endings config dynamically."""
        if self._config_manager is None:
            return False
        return self._config_manager.global_config.tools_platform.normalize_to_lf_line_endings

    def _build_baseline_store_config(self) -> BaselineStoreConfig:
        """Build baseline store config from config."""
        if self._config_manager is None:
            return BaselineStoreConfig(
                base_dir=self.workspace_root / ".locca" / "baselines"
            )
        ws_config = getattr(
            self._config_manager.global_config.tools_platform, "workspace_state", None
        )
        if ws_config is None:
            return BaselineStoreConfig(
                base_dir=self.workspace_root / ".locca" / "baselines"
            )
        baselines_dir = self._config_manager.path_manager.resolve_path(
            getattr(
                ws_config,
                "baselines_directory",
                self.workspace_root / ".locca" / "baselines",
            )
        )
        return BaselineStoreConfig(
            base_dir=baselines_dir,
            ttl_days=getattr(ws_config, "baselines_ttl_days", 7),
            compression_enabled=getattr(ws_config, "compression_enabled", True),
            large_file_threshold_mb=getattr(ws_config, "large_file_threshold_mb", 1),
        )

    def _build_checkpoint_store_config(self) -> CheckpointStoreConfig:
        """Build checkpoint store config from config."""
        if self._config_manager is None:
            return CheckpointStoreConfig(
                base_dir=self.workspace_root / ".locca" / "checkpoints"
            )
        ws_config = getattr(
            self._config_manager.global_config.tools_platform, "workspace_state", None
        )
        if ws_config is None:
            return CheckpointStoreConfig(
                base_dir=self.workspace_root / ".locca" / "checkpoints"
            )
        checkpoints_dir = self._config_manager.path_manager.resolve_path(
            getattr(
                ws_config,
                "checkpoints_directory",
                self.workspace_root / ".locca" / "checkpoints",
            )
        )
        return CheckpointStoreConfig(
            base_dir=checkpoints_dir,
            max_turns=getattr(ws_config, "checkpoints_max_turns", 10),
            ttl_days=getattr(ws_config, "checkpoints_ttl_days", 2),
            compression_enabled=getattr(ws_config, "compression_enabled", True),
            large_file_threshold_mb=getattr(ws_config, "large_file_threshold_mb", 1),
        )

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a path relative to workspace root.

        Args:
            path: Path to resolve (absolute or relative)

        Returns:
            Resolved absolute path

        Raises:
            ToolPathViolationError: If path violates policy
        """
        path_obj = Path(path)

        # Convert to absolute if relative
        if not path_obj.is_absolute():
            path_obj = self.workspace_root / path_obj
        else:
            # If absolute, ensure it's under workspace root
            try:
                path_obj = path_obj.resolve()
                path_obj.relative_to(self.workspace_root)
            except ValueError as exc:
                raise ToolPathViolationError(
                    f"Path {path} is outside workspace root {self.workspace_root}"
                ) from exc

        # Check path policy
        self._validate_path(path_obj)

        return path_obj

    def _validate_path(self, path: Path) -> None:
        """Validate a path against policy.

        Args:
            path: Absolute path to validate

        Raises:
            ToolPathViolationError: If path violates policy
        """
        # Resolve to normalize any '..' segments so the containment check is
        # accurate on all platforms (e.g. PurePath.relative_to tolerates '..'
        # after a matching prefix on Windows, so resolving first is required
        # to detect escapes such as "../etc/passwd").
        resolved = path.resolve()
        try:
            rel_path = resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ToolPathViolationError(
                f"Path {path} is outside workspace root {self.workspace_root}"
            ) from exc

        rel_path_str = str(rel_path).replace("\\", "/")

        # Check denied patterns
        for pattern in self.path_policy.denied_patterns:
            if self._matches_pattern(rel_path, pattern):
                raise ToolPathViolationError(
                    f"Path {rel_path_str} matches denied pattern: {pattern}"
                )

        # Check allowed patterns
        allowed = False
        for pattern in self.path_policy.allowed_patterns:
            if self._matches_pattern(rel_path, pattern):
                allowed = True
                break

        if not allowed:
            raise ToolPathViolationError(
                f"Path {rel_path_str} does not match any allowed pattern"
            )

        # Check depth limit
        if self.path_policy.max_depth is not None:
            depth = len(rel_path.parts)
            if depth > self.path_policy.max_depth:
                raise ToolPathViolationError(
                    f"Path {rel_path_str} exceeds maximum depth of {self.path_policy.max_depth}"
                )

        # Check symlinks
        if not self.path_policy.allow_symlinks:
            self._check_symlink_traversal(path, resolved, rel_path_str)

    def _check_symlink_traversal(
        self, path: Path, resolved: Path, rel_path_str: str
    ) -> None:
        """Check for symlink traversal outside workspace."""
        # Check if the path itself is a symlink
        if path.is_symlink():
            raise ToolPathViolationError(f"Symlinks are not allowed: {rel_path_str}")
        # Check if any parent directory in the resolved path is a symlink
        # that escapes the workspace (traversal protection)
        for parent in resolved.parents:
            if parent == self.workspace_root:
                break
            if parent.is_symlink():
                try:
                    target = parent.resolve()
                    target.relative_to(self.workspace_root)
                except ValueError as exc:
                    raise ToolPathViolationError(
                        f"Symlink traversal outside workspace detected: {rel_path_str}"
                    ) from exc

    def _matches_pattern(self, path: Path, pattern: str) -> bool:
        """Check if a path matches a glob pattern.

        Args:
            path: Relative path to check
            pattern: Glob pattern

        Returns:
            True if path matches pattern
        """
        path_str = str(path).replace("\\", "/")

        # Handle ** patterns by checking if path starts with the prefix
        if pattern == "**" or pattern == "**/*":
            return True

        # Handle patterns ending with **/*
        if pattern.endswith("/**/*"):
            prefix = pattern[:-5]  # Remove /**/*
            return path_str.startswith(prefix) or path_str == prefix.rstrip("/")

        # Use fnmatch for other patterns
        return fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(
            path_str + "/", pattern
        )

    def get_relative_path(self, path: Path) -> str:
        """Get a path relative to workspace root.

        Args:
            path: Absolute path

        Returns:
            Relative path string

        Raises:
            ToolPathViolationError: If path is outside workspace
        """
        try:
            rel_path = path.relative_to(self.workspace_root)
            return str(rel_path).replace("\\", "/")
        except ValueError as exc:
            raise ToolPathViolationError(
                f"Path {path} is outside workspace root {self.workspace_root}"
            ) from exc

    def is_path_inside_workspace(self, path: str | Path) -> bool:
        """Check if a path is inside the workspace without raising.

        Args:
            path: Path to check (absolute or relative)

        Returns:
            True if path is inside workspace, False otherwise
        """
        path_obj = Path(path)

        # Convert to absolute if relative
        if not path_obj.is_absolute():
            path_obj = (self.workspace_root / path_obj).resolve()
        else:
            try:
                path_obj = path_obj.resolve()
            except Exception:
                return False

        try:
            path_obj.relative_to(self.workspace_root)
            return True
        except ValueError:
            return False

    def read_file_raw(self, path: Path) -> str:
        """Read raw file content from disk (no line-ending normalization).

        Used for baseline hashing where raw disk bytes must be preserved.
        """
        with path.open(mode="r", encoding="utf-8", newline="") as f:
            return f.read()

    def read_file_normalized(self, path: Path) -> str:
        """Read file content and normalize line endings for LLM consumption.

        When ``normalize_to_lf_line_endings`` is enabled, CRLF/CR are converted
        to LF so LLM-generated edits match on-disk content. Otherwise content is
        returned as-is.
        """
        return self._line_normalizer.normalize(self.read_file_raw(path))

    def normalize_for_match(self, content: str) -> str:
        """Normalize ``content`` to LF for edit matching when enabled."""
        return self._line_normalizer.normalize(content)

    def restore_line_endings(self, content: str, original: LineEnding | None) -> str:
        """Restore normalized ``content`` to ``original`` line endings.

        Pass-through when normalization is disabled or ``original`` is
        ``None``/LF/none. Used before writing edited content back to disk so
        the file's native line endings are preserved.
        """
        if original is None:
            return content
        return self._line_normalizer.restore(content, original)

    def write_file_content(
        self,
        path: Path,
        content: str,
        original_ending: LineEnding | None = None,
    ) -> None:
        """Write ``content`` to disk, restoring line endings when configured.

        When normalization is enabled and ``original_ending`` is provided, the
        content is restored to the file's native line endings before writing so
        the disk preserves its original style. Otherwise content is written as-is.
        """
        content = self.restore_line_endings(content, original_ending)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode="w", encoding="utf-8", newline="") as f:
            f.write(content)

    def detect_line_ending(self, content: str) -> LineEnding:
        """Detect the dominant line ending in ``content``."""
        return detect_line_ending(content)

    def generate_failure_hint(
        self,
        content_lf: str,
        old_text_lf: str,
        expected: int,
        actual: int,
    ) -> str:
        """Generate a human-readable hint when old_text does not match file content.

        All inputs should already be normalized to LF.
        """
        return generate_failure_hint(content_lf, old_text_lf, expected, actual)

    def normalize_tool_paths(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Normalize path arguments in tool call arguments.

        This method normalizes paths relative to workspace root and marks
        out-of-workspace paths for boundary checks. Used by gateway validation.

        Args:
            arguments: Tool arguments dict

        Returns:
            Dictionary with normalized paths and metadata
        """
        normalized_paths: dict[str, Any] = {}

        # Check for single path argument
        path_arg = arguments.get("path")
        if path_arg:
            path_obj = Path(path_arg)

            if path_obj.is_absolute():
                if self.is_path_inside_workspace(path_obj):
                    resolved = path_obj.resolve()
                    rel_path = resolved.relative_to(self.workspace_root)
                    normalized_paths["path"] = str(rel_path).replace("\\", "/")
                else:
                    # Path is outside workspace - keep as-is but mark for boundary check
                    normalized_paths["path"] = path_arg
                    normalized_paths["_outside_workspace"] = "true"
            else:
                # For relative paths, resolve against workspace root
                resolved = self.workspace_root / path_obj
                if self.is_path_inside_workspace(resolved):
                    rel_path = resolved.relative_to(self.workspace_root)
                    normalized_paths["path"] = str(rel_path).replace("\\", "/")
                else:
                    normalized_paths["path"] = path_arg

        # Handle multiple paths (e.g., in search tools)
        paths_arg = arguments.get("paths")
        if paths_arg and isinstance(paths_arg, list):
            normalized_list = []
            for p in paths_arg:
                path_obj = Path(p)
                if path_obj.is_absolute():
                    if self.is_path_inside_workspace(path_obj):
                        resolved = path_obj.resolve()
                        rel_path = resolved.relative_to(self.workspace_root)
                        normalized_list.append(str(rel_path).replace("\\", "/"))
                    else:
                        normalized_list.append(p)
                else:
                    resolved = self.workspace_root / path_obj
                    if self.is_path_inside_workspace(resolved):
                        rel_path = resolved.relative_to(self.workspace_root)
                        normalized_list.append(str(rel_path).replace("\\", "/"))
                    else:
                        normalized_list.append(p)
            normalized_paths["paths"] = normalized_list

        return normalized_paths

    def get_baseline_observation(
        self, file_path: str, session_id: str | None = None
    ) -> FileObservation | None:
        """Get the baseline observation for a file (from read/snapshot, not external changes).

        This returns the observation that should be used as the base for conflict detection,
        ignoring observations from external file monitoring events.

        Args:
            file_path: Relative path to file
            session_id: Optional session ID to check session-scoped overrides

        Returns:
            FileObservation with source READ, SNAPSHOT, or AGENT_CREATE, or None if not tracked
        """
        observation = self.get_file_version(file_path, session_id=session_id)
        if observation and observation.source in (
            ObservationSource.READ,
            ObservationSource.SNAPSHOT,
            ObservationSource.AGENT_CREATE,
        ):
            return observation
        return None

    def get_file_version(
        self, file_path: str, session_id: str | None = None
    ) -> FileObservation | None:
        """Get the current version for a file.

        Args:
            file_path: Relative path to file
            session_id: Optional session ID to check session-scoped overrides

        Returns:
            FileObservation if found in session-scoped or global storage, None otherwise
        """
        state = self.transaction_manager.get_workspace_state()
        return state.get_file_version(file_path, session_id=session_id)

    def observe_file(
        self,
        file_path: str,
        source: ObservationSource = ObservationSource.READ,
        session_id: str | None = None,
    ) -> FileObservation:
        """Centralized file observation with baseline management.

        This is the single entry point for all file observations.
        - Updates session_file_versions
        - For baseline sources (READ, SNAPSHOT, AGENT_CREATE): creates/updates baseline store
        - Deletes old baseline file when hash changes
        - For EXTERNAL source: only updates in-memory, no baseline store changes

        Args:
            file_path: Relative path to file
            source: Origin of this observation
            session_id: Session ID for tracking

        Returns:
            FileObservation with current state
        """
        resolved_path = self.resolve_path(file_path)

        return self.transaction_manager.observe_file(
            resolved_path, source=source, session_id=session_id
        )

    def get_baseline_content(
        self, file_path: str, session_id: str | None = None
    ) -> str | None:
        """Get baseline content for a file from disk (lazy load).

        This loads from the content-addressable baseline store using the
        hash from session_file_versions.
        """
        resolved_path = self.resolve_path(file_path)
        rel_path = self.get_relative_path(resolved_path)

        # Get hash from session_file_versions
        baseline_obs = self.get_baseline_observation(rel_path, session_id=session_id)
        if baseline_obs is None:
            return None

        # Load from baseline store
        return self.baseline_store.load(baseline_obs.hash)

    def begin_transaction(self, transaction_id: str | None = None) -> str:
        """Begin a new transaction.

        Args:
            transaction_id: Optional transaction ID

        Returns:
            Transaction ID
        """
        transaction = self.transaction_manager.begin_transaction(transaction_id)
        return transaction.transaction_id

    def commit_transaction(self) -> Any:
        """Commit the active transaction.

        Returns:
            ChangeSet of changes
        """
        return self.transaction_manager.commit_transaction()

    def rollback_transaction(self) -> None:
        """Rollback the active transaction."""
        self.transaction_manager.rollback_transaction()

    def get_active_transaction_id(self) -> str | None:
        """Get the currently active transaction ID.

        Returns:
            Transaction ID or None
        """
        transaction = self.transaction_manager.get_active_transaction()
        return transaction.transaction_id if transaction else None

    def _get_diff_limits(self) -> tuple[int, int]:
        """Get diff truncation limits from config.

        Returns:
            Tuple of (max_lines, max_chars)
        """
        if self._config_manager is None:
            return 80, 6000
        tc = self._config_manager.global_config.tools_platform
        max_lines = tc.max_read_lines
        max_chars = tc.max_output_bytes
        return max_lines, max_chars

    def _compute_truncated_diff(
        self, from_content: str, to_content: str, from_label: str, to_label: str
    ) -> tuple[bool, str]:
        """Compute unified diff and truncate if necessary.

        Args:
            from_content: Original content
            to_content: New content
            from_label: Label for original (e.g., "baseline")
            to_label: Label for new (e.g., "current")

        Returns:
            Tuple of (was_truncated, diff_string)
        """
        import difflib

        max_lines, max_chars = self._get_diff_limits()

        diff_lines = list(
            difflib.unified_diff(
                from_content.splitlines(keepends=True),
                to_content.splitlines(keepends=True),
                fromfile=from_label,
                tofile=to_label,
            )
        )

        if not diff_lines:
            return False, ""

        diff_str_unnormalized = "".join(diff_lines)
        diff_str = self.normalize_for_match(diff_str_unnormalized)

        lines = diff_str.splitlines(keepends=True)
        if len(lines) <= max_lines and len(diff_str) <= max_chars:
            return False, diff_str

        keep = max_lines // 2
        head = lines[:keep]
        tail = lines[-keep:]
        truncated = "".join(
            [*head, f"\n... [{len(lines) - 2 * keep} lines truncated] ...\n", *tail]
        )

        if len(truncated) > max_chars:
            truncated = truncated[:max_chars] + "\n[truncated]"

        return True, truncated

    def validate_write_version(
        self,
        file_path: str,
        proposed_hash: str,
        proposed_content: str,
        session_id: str,
    ) -> ConflictReport:
        """Validate the expected file version before a write.

        Single entrypoint for conflict detection. Compares the baseline hash
        (from reads, stored in workspace ``file_versions``) against the current
        disk hash. Returns a ConflictReport always: ``conflict_detected=False``
        when the baseline is unguarded or matches, ``conflict_detected=True``
        with diffs and recommendation when the file changed externally.

        Args:
            file_path: Relative path to file
            proposed_hash: Hash of the proposed file content
            proposed_content: The proposed file content (for proposed diff)
            session_id: Session ID to scope the baseline lookup

        Returns:
            A ConflictReport. When ``conflict_detected`` is True, the report
            carries external_diff, proposed_diff, recommendation, and message
            for reporting back to the LLM.
        """

        resolved_path = self.resolve_path(file_path)
        rel_path = self.get_relative_path(resolved_path)

        # Baseline comes from workspace file_versions (read/snapshot/create),
        # not from external monitoring or a caller-supplied hash. The LLM never
        # touches hashes. Try session-scoped baseline first, fall back to the
        # global (session_id=None) baseline for backward compatibility.
        baseline_observation = self.get_baseline_observation(
            rel_path, session_id=session_id
        )
        if baseline_observation is None:
            baseline_observation = self.get_baseline_observation(rel_path)
        if baseline_observation is None:
            # No baseline to compare against → unguarded write.
            return ConflictReport(
                path=rel_path,
                base_hash="",
                current_hash="",
                proposed_hash=proposed_hash,
                conflict_type="none",
                conflict_detected=False,
            )

        # Delegate detection to the conflict detector (hash comparison).
        conflict = self.conflict_detector.detect_write_conflict(
            workspace_root=self.workspace_root,
            file_path=rel_path,
            base_observation=baseline_observation,
            proposed_hash=proposed_hash,
        )

        if not conflict.conflict_detected:
            return conflict

        # Conflict detected: compute diffs and recommendation.
        try:
            current_content = self.read_file_raw(resolved_path)
        except OSError:
            current_content = ""

        baseline_content = self.get_baseline_content(rel_path, session_id=session_id)

        if baseline_content:
            ext_truncated, external_diff = self._compute_truncated_diff(
                baseline_content, current_content, "baseline", "current_disk"
            )
            prop_truncated, proposed_diff = self._compute_truncated_diff(
                baseline_content, proposed_content, "baseline", "proposed_edit"
            )
            was_truncated = ext_truncated or prop_truncated
            if was_truncated:
                recommendation = "re_read_file"
                message = (
                    "File modified externally since you read it (diff truncated). "
                    "Please re-read the file to get current content, then retry your edit."
                )
            else:
                recommendation = "retry_or_adapt"
                message = (
                    "File modified externally. See diffs below. "
                    "You can adapt your edit and retry, or re-read the file."
                )
        else:
            external_diff = ""
            proposed_diff = ""
            recommendation = "re_read_file"
            message = (
                "File modified externally since you read it. "
                "Please re-read the file to get current content, then retry your edit."
            )

        conflict.external_diff = external_diff
        conflict.proposed_diff = proposed_diff
        conflict.recommendation = recommendation
        conflict.message = message

        return conflict

    def atomic_write(
        self,
        file_path: str,
        content: str,
        session_id: str,
        expected_hash: str | None = None,
    ) -> tuple[FileObservation | None, FileObservation]:
        """Perform an atomic write with conflict detection.

        Args:
            file_path: Relative path to file
            content: New file content
            session_id: Session ID for the file_versions baseline
            expected_hash: Expected hash before write. When omitted, derived from
                the workspace ``file_versions`` baseline (populated by reads).

        Returns:
            Tuple of (before_observation, after_observation). before_observation
            is None if the file was newly created.

        Raises:
            ValueError: If conflict detected or hash mismatch
        """
        resolved_path = self.resolve_path(file_path)

        # Perform atomic write
        return self.transaction_manager.atomic_write(
            resolved_path=resolved_path,
            content=content,
            expected_hash=expected_hash,
            session_id=session_id,
        )

    def atomic_create(
        self, file_path: str, content: str, session_id: str | None = None
    ) -> FileObservation:
        """Perform an atomic file creation.

        Args:
            file_path: Relative path to file
            content: File content
            session_id: Session ID for the file_versions baseline

        Returns:
            After observation

        Raises:
            FileExistsError: If file already exists
        """
        resolved_path = self.resolve_path(file_path)

        return self.transaction_manager.atomic_create(
            resolved_path, content, session_id=session_id
        )

    def atomic_delete(
        self,
        file_path: str,
        session_id: str | None = None,
    ) -> FileObservation | None:
        """Perform an atomic file deletion.

        Args:
            file_path: Relative path to file
            session_id: Session ID for tracking

        Returns:
            Before observation

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If hash mismatch
        """
        resolved_path = self.resolve_path(file_path)

        return self.transaction_manager.atomic_delete(
            resolved_path, session_id=session_id
        )

    def create_snapshot(
        self,
        file_paths: list[str],
        transaction_id: str | None = None,
    ) -> str:
        """Create a snapshot of multiple files.

        Args:
            file_paths: List of relative file paths
            transaction_id: Optional transaction ID

        Returns:
            Snapshot ID
        """
        snapshot = self.snapshot_manager.create_workspace_snapshot(
            workspace_id=self.workspace_id,
            workspace_root=self.workspace_root,
            file_paths=file_paths,
            transaction_id=transaction_id,
        )
        return snapshot.snapshot_id

    def restore_snapshot(self, snapshot_id: str, file_path: str) -> None:
        """Restore a file from a snapshot.

        Args:
            snapshot_id: Snapshot identifier
            file_path: Relative path to file
        """
        self.snapshot_manager.restore_file_snapshot(
            workspace_root=self.workspace_root,
            snapshot_id=snapshot_id,
            file_path=file_path,
        )

    def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete a snapshot.

        Args:
            snapshot_id: Snapshot identifier
        """
        self.snapshot_manager.delete_snapshot(snapshot_id)

    def get_workspace_state(self) -> WorkspaceState:
        """Get the current workspace state.

        Returns:
            WorkspaceState with file versions
        """
        return self.transaction_manager.get_workspace_state()

    def clear_session_overrides(self, session_id: str) -> None:
        """Clear session-scoped file version overrides for a session.

        Args:
            session_id: Session ID to clear overrides for
        """
        state = self.transaction_manager.get_workspace_state()
        state.clear_session_overrides(session_id)

    def override_file_versions(
        self, file_versions: dict[str, FileObservation], session_id: str | None = None
    ) -> None:
        """Override the entire file versions dictionary in workspace state.

        Args:
            file_versions: New dictionary mapping paths to file observations
            session_id: Optional session ID for session-scoped overrides. If provided,
                       overrides are scoped to this session only. If None, overrides
                       the global file_versions dictionary.
        """
        state = self.transaction_manager.get_workspace_state()
        state.override_file_versions(file_versions, session_id=session_id)

    def capture_checkpoint(
        self,
        session_id: str,
        turn_id: str,
        file_path: str,
        content: str | None = None,
    ) -> bool:
        """Capture a checkpoint for a file in a turn if not already captured.

        This should be called before a write/create/delete operation.
        If content is provided, it's used directly; otherwise reads from disk.

        Returns True if checkpoint was captured (first time for this file in this turn),
        False if already captured.
        """
        resolved_path = self.resolve_path(file_path)
        rel_path = self.get_relative_path(resolved_path)

        state = self.transaction_manager.get_workspace_state()
        if state.has_checkpoint(session_id, turn_id, rel_path):
            return False

        # Get content to store
        if content is None:
            if resolved_path.exists():
                try:
                    content = resolved_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    content = None  # Binary file - skip
            else:
                content = None  # File doesn't exist (creation)

        # Store checkpoint on disk
        checkpoint_hash = self.checkpoint_store.store(
            session_id=session_id,
            turn_id=turn_id,
            file_path=rel_path,
            content=content,
        )

        # Track in memory
        state.capture_checkpoint(session_id, turn_id, rel_path, checkpoint_hash)
        return True

    def load_checkpoint(
        self, session_id: str, turn_id: str, file_path: str
    ) -> str | None:
        """Load checkpoint content for a file in a turn.

        Returns:
            Content string if file existed at checkpoint
            Empty string if deletion marker (file was created)
            None if checkpoint not found
        """
        resolved_path = self.resolve_path(file_path)
        rel_path = self.get_relative_path(resolved_path)
        return self.checkpoint_store.load(session_id, turn_id, rel_path)

    def undo_file(self, session_id: str, file_path: str, turn_id: str) -> str:
        """Undo a file to its checkpoint state.

        turn_id is required - caller must specify which turn to undo.

        Returns the restored content.
        """
        resolved_path = self.resolve_path(file_path)
        rel_path = self.get_relative_path(resolved_path)

        content = self.load_checkpoint(session_id, turn_id, rel_path)
        if content is None:
            raise ValueError(
                f"Checkpoint not found for file {file_path} in turn {turn_id}"
            )

        if content == "":
            # Deletion marker - file was created, so delete it
            if resolved_path.exists():
                resolved_path.unlink()
            logger.info("Undid file creation", file=file_path, turn=turn_id)
        else:
            # Restore content
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(content, encoding="utf-8")
            logger.info("Undid file change", file=file_path, turn=turn_id)

        # NOTE: Do NOT update baseline here - user undo is not an LLM action
        # The LLM is not aware of user-initiated undos

        return content

    def cleanup_workspace_state(self) -> dict[str, int]:
        """Run cleanup for baselines and checkpoints.

        Returns dict with counts of deleted items.
        """
        # Get active sessions and their turns from memory
        state = self.transaction_manager.get_workspace_state()
        active_sessions: dict[str, list[int]] = {}
        for session_id in state.session_checkpoints.keys():
            active_sessions[session_id] = state.get_active_turns(session_id)

        # Get referenced baseline hashes
        referenced_hashes = set()
        for session_versions in state.session_file_versions.values():
            for obs in session_versions.values():
                referenced_hashes.add(obs.hash)

        # Cleanup baselines (TTL only, no orphan deletion)
        baselines_deleted = self.baseline_store.cleanup_expired(referenced_hashes)

        # Cleanup checkpoints
        checkpoints_deleted = self.checkpoint_store.cleanup_expired(active_sessions)

        return {
            "baselines_deleted": baselines_deleted,
            "checkpoints_deleted": checkpoints_deleted,
        }
