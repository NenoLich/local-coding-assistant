"""Workspace models for file version tracking and transaction state.

This module provides models for:
- File version tracking (hash, size, mtime, inode)
- Transaction journal entries
- Path policy configuration
- Workspace state
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from local_coding_assistant.tools_platform import ToolPathViolationError
from local_coding_assistant.tools_platform.contracts import (
    FileObservation,
    ObservationSource,
    TransactionState,
)
from local_coding_assistant.tools_platform.workspace.content_store import BaselineStore
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("tools_platform.workspace.models")


BASELINE_SOURCES = frozenset(
    {
        ObservationSource.READ,
        ObservationSource.SNAPSHOT,
        ObservationSource.AGENT_CREATE,
    }
)


class ChangeOrigin(str, Enum):
    """Origin of a file change."""

    AGENT = "agent"
    USER = "user"
    EXTERNAL = "external"
    SYSTEM = "system"


class PathPolicyType(str, Enum):
    """Types of path policies."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_WORKSPACE = "require_workspace"


@dataclass(frozen=True)
class PathPolicy:
    """Policy for workspace path access.

    Attributes:
        workspace_root: Absolute path to workspace root
        allowed_patterns: Glob patterns for allowed paths (relative to root)
        denied_patterns: Glob patterns for denied paths (relative to root)
        allow_symlinks: Whether to allow symlink traversal
        max_depth: Maximum directory depth (None for unlimited)
        require_relative: Whether paths must be relative to workspace
    """

    workspace_root: Path
    allowed_patterns: list[str] = field(default_factory=lambda: ["**"])
    denied_patterns: list[str] = field(default_factory=list)
    allow_symlinks: bool = False
    max_depth: int | None = None
    require_relative: bool = True


class TransactionJournalEntry(BaseModel):
    """Entry in the transaction journal.

    Attributes:
        entry_id: Unique entry identifier
        transaction_id: Transaction this entry belongs to
        operation: Type of operation (read, write, create, delete)
        path: File path (relative to workspace)
        before_observation: File state before operation
        after_observation: File state after operation (if completed)
        status: Entry status (pending, completed, failed, rolled_back)
        error: Error if operation failed
        timestamp: When this entry was created
        metadata: Additional operation metadata
    """

    entry_id: str
    transaction_id: str
    operation: str
    path: str
    before_observation: FileObservation | None = None
    after_observation: FileObservation | None = None
    status: str = "pending"
    error: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransactionJournal(BaseModel):
    """Journal of all operations in a transaction.

    Attributes:
        transaction_id: Transaction identifier
        workspace_id: Workspace identifier
        entries: List of journal entries
        state: Current transaction state
        started_at: When transaction started
        committed_at: When transaction was committed (if applicable)
        rolled_back_at: When transaction was rolled back (if applicable)
    """

    transaction_id: str
    workspace_id: str
    entries: list[TransactionJournalEntry] = Field(default_factory=list)
    state: TransactionState = TransactionState.ACTIVE
    started_at: datetime = Field(default_factory=datetime.utcnow)
    committed_at: datetime | None = None
    rolled_back_at: datetime | None = None

    def add_entry(self, entry: TransactionJournalEntry) -> None:
        """Add an entry to the journal."""
        self.entries.append(entry)

    def get_pending_entries(self) -> list[TransactionJournalEntry]:
        """Get all pending entries."""
        return [e for e in self.entries if e.status == "pending"]

    def get_completed_entries(self) -> list[TransactionJournalEntry]:
        """Get all completed entries."""
        return [e for e in self.entries if e.status == "completed"]

    def get_failed_entries(self) -> list[TransactionJournalEntry]:
        """Get all failed entries."""
        return [e for e in self.entries if e.status == "failed"]


class CheckpointMetadata(BaseModel):
    """Metadata for a single checkpoint."""

    file_path: str
    checkpoint_hash: str
    is_deletion_marker: bool = False  # True if file was created (undo = delete)
    captured_at: datetime = Field(default_factory=datetime.utcnow)


class TurnCheckpoints(BaseModel):
    """All checkpoints for a single turn."""

    turn_id: str
    checkpoints: dict[str, CheckpointMetadata] = Field(
        default_factory=dict
    )  # file_path -> metadata

    def add_checkpoint(self, metadata: CheckpointMetadata) -> bool:
        """Add checkpoint if not already present. Returns True if added."""
        if metadata.file_path in self.checkpoints:
            return False
        self.checkpoints[metadata.file_path] = metadata
        return True

    def has_checkpoint(self, file_path: str) -> bool:
        return file_path in self.checkpoints

    def get_checkpoint(self, file_path: str) -> CheckpointMetadata | None:
        return self.checkpoints.get(file_path)


class SessionCheckpoints(BaseModel):
    """All checkpoints for a session across turns."""

    session_id: str
    turns: dict[str, TurnCheckpoints] = Field(
        default_factory=dict
    )  # turn_id -> TurnCheckpoints

    def get_turn(self, turn_id: str) -> TurnCheckpoints | None:
        return self.turns.get(turn_id)

    def get_or_create_turn(self, turn_id: str) -> TurnCheckpoints:
        if turn_id not in self.turns:
            self.turns[turn_id] = TurnCheckpoints(turn_id=turn_id)
        return self.turns[turn_id]

    def capture_checkpoint(
        self,
        turn_id: str,
        file_path: str,
        checkpoint_hash: str,
        is_deletion_marker: bool = False,
    ) -> bool:
        """Capture checkpoint for a file in a turn if not already captured."""
        turn = self.get_or_create_turn(turn_id)
        return turn.add_checkpoint(
            CheckpointMetadata(
                file_path=file_path,
                checkpoint_hash=checkpoint_hash,
                is_deletion_marker=is_deletion_marker,
            )
        )

    def has_checkpoint(self, turn_id: str, file_path: str) -> bool:
        turn = self.turns.get(turn_id)
        return turn.has_checkpoint(file_path) if turn else False

    def get_turn_checkpoints(self, turn_id: str) -> dict[str, CheckpointMetadata]:
        turn = self.turns.get(turn_id)
        return turn.checkpoints if turn else {}

    def get_active_turn_ids(self) -> list[int]:
        turns = []
        for turn_id_str in self.turns.keys():
            try:
                turns.append(int(turn_id_str))
            except ValueError:
                continue
        return sorted(turns)


class WorkspaceState(BaseModel):
    """Current state of the workspace.

    Attributes:
        workspace_id: Workspace identifier
        workspace_root: Absolute path to workspace root
        active_transaction_id: Currently active transaction (if any)
        session_file_versions: Session-scoped file version overrides keyed by session_id
        session_checkpoints: Session-scoped checkpoint metadata {session_id: SessionCheckpoints}
        last_scan_time: When workspace was last scanned
        total_files: Total number of tracked files
        total_bytes: Total bytes in tracked files
    """

    model_config = {"arbitrary_types_allowed": True}

    workspace_id: str
    workspace_root: str
    baseline_store: BaselineStore
    active_transaction_id: str | None = None
    session_file_versions: dict[str | None, dict[str, FileObservation]] = Field(
        default_factory=dict
    )
    session_checkpoints: dict[str, SessionCheckpoints] = Field(default_factory=dict)

    last_scan_time: datetime | None = None
    total_files: int = 0
    total_bytes: int = 0

    def update_file_version(
        self, path: str, observation: FileObservation, session_id: str | None = None
    ) -> None:
        """Update the version for a file in a session scope."""
        self.session_file_versions.setdefault(session_id, {})[path] = observation

    def get_file_version(
        self, path: str, session_id: str | None = None
    ) -> FileObservation | None:
        """Get the current version for a file.

        Args:
            path: File path to look up
            session_id: Optional session ID to check session-scoped overrides first

        Returns:
            FileObservation if found in session-scoped or global storage, None otherwise
        """
        session_overrides = self.session_file_versions.get(session_id)
        if session_overrides is not None and path in session_overrides:
            return session_overrides[path]
        return None

    def remove_file_version(self, path: str, session_id: str | None = None) -> None:
        """Remove a file from version tracking."""
        bucket = self.session_file_versions.get(session_id)
        if bucket is not None:
            obs = bucket.pop(path, None)
            if obs:
                self.baseline_store.delete(obs.hash)

    def get_session_checkpoints(self, session_id: str) -> SessionCheckpoints:
        """Get or create session checkpoints."""
        if session_id not in self.session_checkpoints:
            self.session_checkpoints[session_id] = SessionCheckpoints(
                session_id=session_id
            )
        return self.session_checkpoints[session_id]

    def capture_checkpoint(
        self,
        session_id: str,
        turn_id: str,
        file_path: str,
        checkpoint_hash: str,
        is_deletion_marker: bool = False,
    ) -> bool:
        """Capture checkpoint for a file in a turn if not already captured."""
        session_checkpoints = self.get_session_checkpoints(session_id)
        return session_checkpoints.capture_checkpoint(
            turn_id, file_path, checkpoint_hash, is_deletion_marker
        )

    def has_checkpoint(self, session_id: str, turn_id: str, file_path: str) -> bool:
        """Check if checkpoint already captured for file in turn."""
        session_checkpoints = self.session_checkpoints.get(session_id)
        if not session_checkpoints:
            return False
        return session_checkpoints.has_checkpoint(turn_id, file_path)

    def get_turn_checkpoints(
        self, session_id: str, turn_id: str
    ) -> dict[str, CheckpointMetadata]:
        """Get all checkpoint metadata for a turn."""
        session_checkpoints = self.session_checkpoints.get(session_id)
        if not session_checkpoints:
            return {}
        return session_checkpoints.get_turn_checkpoints(turn_id)

    def override_file_versions(
        self, file_versions: dict[str, FileObservation], session_id: str | None = None
    ) -> None:
        """Override the entire file versions dictionary.

        Args:
            file_versions: New dictionary mapping paths to file observations
            session_id: Optional session ID for session-scoped overrides. If provided,
                       overrides are scoped to this session only. If None, overrides
                       the global file_versions dictionary.
        """
        self.session_file_versions.setdefault(session_id, {}).update(file_versions)

    def clear_session_overrides(self, session_id: str) -> None:
        """Clear session-scoped file version overrides for a session.

        Args:
            session_id: Session ID to clear overrides for
        """
        self.session_file_versions.pop(session_id, None)
        self.session_checkpoints.pop(session_id, None)

    def get_active_turns(self, session_id: str) -> list[int]:
        """Get list of active turn IDs for a session (for cleanup)."""
        session_checkpoints = self.session_checkpoints.get(session_id)
        if not session_checkpoints:
            return []
        return session_checkpoints.get_active_turn_ids()

    def get_relative_path(self, path: Path) -> str:
        """Get a path relative to workspace root.

        Args:
            path: Absolute or relative path

        Returns:
            Relative path string

        Raises:
            ToolPathViolationError: If path is outside workspace
        """
        # Resolve relative paths against workspace root
        if not path.is_absolute():
            path = self.workspace_root / path
        try:
            rel_path = path.relative_to(self.workspace_root)
            return str(rel_path).replace("\\", "/")
        except ValueError as exc:
            raise ToolPathViolationError(
                f"Path {path} is outside workspace root {self.workspace_root}"
            ) from exc

    def observe_file(
        self,
        resolved_path: Path,
        source: ObservationSource = ObservationSource.READ,
        session_id: str | None = None,
        store_baseline: bool = True,
    ) -> FileObservation:
        """Centralized file observation with baseline management.

        This is the single entry point for all file observations.
        - Updates session_file_versions
        - For baseline sources (READ, SNAPSHOT, AGENT_CREATE): creates/updates baseline store
        - Deletes old baseline file when hash changes
        - For EXTERNAL source: only updates in-memory, no baseline store changes

        Args:
            resolved_path: Resolved path to file
            source: Origin of this observation
            session_id: Session ID for tracking
            store_baseline: Whether to store baseline

        Returns:
            FileObservation with current state
        """
        rel_path = self.get_relative_path(resolved_path)

        if not resolved_path.exists():
            observation = FileObservation(
                path=rel_path,
                hash="",
                size=0,
                mtime=datetime.utcnow(),
                inode=None,
                source=source,
            )
            self.update_file_version(rel_path, observation, session_id=session_id)
            return observation

        stat = resolved_path.stat()
        content_bytes = resolved_path.read_bytes()
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        mtime = datetime.fromtimestamp(stat.st_mtime)

        observation = FileObservation(
            path=rel_path,
            hash=content_hash,
            size=stat.st_size,
            mtime=mtime,
            inode=str(stat.st_ino) if hasattr(stat, "st_ino") else None,
            source=source,
        )

        # Handle baseline store for baseline sources
        if source in BASELINE_SOURCES:
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Binary file - skip baseline store, but still update file version
                self.update_file_version(rel_path, observation, session_id=session_id)
                return observation

            # Check if baseline already exists with same hash
            existing_obs = self.get_file_version(rel_path, session_id=session_id)
            if existing_obs and existing_obs.hash == content_hash:
                return observation  # Baseline already up to date

            # Update file version (always, for new or changed files)
            self.update_file_version(rel_path, observation, session_id=session_id)

            if store_baseline:
                # Store new baseline
                self.baseline_store.store(content, content_hash)

                # If hash changed, delete old baseline file
                if existing_obs and existing_obs.hash != content_hash:
                    self.baseline_store.delete(existing_obs.hash)

        else:
            # Non-baseline sources (EXTERNAL) - only update in-memory
            self.update_file_version(rel_path, observation, session_id=session_id)

        return observation


class WorkspaceSnapshot(BaseModel):
    """Snapshot of workspace state at a point in time.

    Attributes:
        snapshot_id: Unique snapshot identifier
        workspace_id: Workspace this snapshot belongs to
        snapshot_time: When snapshot was created
        file_snapshots: Dictionary mapping paths to file snapshots
        transaction_id: Transaction that created this snapshot (if applicable)
        metadata: Additional snapshot metadata
    """

    snapshot_id: str
    workspace_id: str
    snapshot_time: datetime = Field(default_factory=datetime.utcnow)
    file_snapshots: dict[str, FileObservation] = Field(default_factory=dict)
    transaction_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConflictResolution(str, Enum):
    """Options for resolving conflicts."""

    REFRESH_REBASE = "refresh_rebase"
    OVERWRITE = "overwrite"
    OVERWRITE_WITH_APPROVAL = "overwrite_with_approval"
    SKIP = "skip"
    MANUAL = "manual"
