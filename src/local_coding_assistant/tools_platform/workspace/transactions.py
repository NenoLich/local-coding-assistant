"""Transaction management for workspace operations.

This module provides functionality for:
- Beginning, committing, and rolling back transactions
- Maintaining a journal of operations within a transaction
- Atomic writes with rollback capability
- Transaction state management
- File version tracking with source attribution (read, write, snapshot, agent_create, external)

RESPONSIBILITY (per phase 5d):
- WorkspaceManager owns the per-session workspace state: file baselines,
  active transactions, file snapshots, hash history, and pending write expectations.
- TransactionManager provides the core transaction mechanics and workspace state storage.
- FileObservation includes source field to distinguish baseline observations
  (READ, SNAPSHOT, AGENT_CREATE) from external observations (EXTERNAL).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from local_coding_assistant.tools_platform.contracts import (
    ChangeSet,
    FileChange,
    FileObservation,
    ObservationSource,
    TransactionState,
)
from local_coding_assistant.tools_platform.workspace.content_store import BaselineStore
from local_coding_assistant.tools_platform.workspace.models import (
    TransactionJournal,
    TransactionJournalEntry,
    WorkspaceState,
)
from local_coding_assistant.tools_platform.workspace.snapshots import SnapshotManager
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("tools_platform.workspace.transactions")


class OperationType(str, Enum):
    """Types of operations in a transaction."""

    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"
    REPLACE_EXACT = "replace_exact"


@dataclass
class TransactionConfig:
    """Configuration for transaction behavior.

    Attributes:
        enable_snapshots: Whether to create snapshots before writes
        auto_rollback_on_error: Whether to auto-rollback on error
        max_operations: Maximum operations per transaction
        timeout_seconds: Transaction timeout in seconds
    """

    enable_snapshots: bool = True
    auto_rollback_on_error: bool = True
    max_operations: int = 1000
    timeout_seconds: int = 300


class TransactionManager:
    """Manages workspace transactions with atomic operations.

    This class:
    - Begins and commits transactions
    - Tracks operations in a journal
    - Creates snapshots for rollback
    - Performs atomic writes
    - Handles rollback on errors
    """

    def __init__(
        self,
        workspace_id: str,
        workspace_root: Path,
        snapshot_manager: SnapshotManager,
        baseline_store: BaselineStore,
        config: TransactionConfig | None = None,
    ) -> None:
        """Initialize the transaction manager.

        Args:
            workspace_id: Workspace identifier
            workspace_root: Workspace root directory
            snapshot_manager: Snapshot manager for rollback
            config: Transaction configuration
        """
        self.workspace_id = workspace_id
        self.workspace_root = workspace_root
        self.snapshot_manager = snapshot_manager
        self.config = config or TransactionConfig()

        self._active_transaction: TransactionJournal | None = None
        self._transaction_history: dict[str, TransactionJournal] = {}
        self._workspace_state = WorkspaceState(
            workspace_id=workspace_id,
            workspace_root=str(workspace_root),
            baseline_store=baseline_store,
        )

    def _compute_hash(self, content: bytes) -> str:
        """Compute SHA256 hash of content.

        Args:
            content: Content bytes to hash

        Returns:
            Hex digest of SHA256 hash
        """
        return hashlib.sha256(content).hexdigest()

    def observe_file(
        self,
        resolved_path: Path,
        source: ObservationSource = ObservationSource.READ,
        session_id: str | None = None,
        store_baseline: bool = True,
    ) -> FileObservation:
        """Observe the current state of a file.

        Delegates to WorkspaceManager.observe_file if available for centralized
        baseline management, otherwise uses local implementation.

        Args:
            resolved_path: Resolved path to file
            source: Origin of this observation
            session_id: Session ID for tracking
            store_baseline: Whether to store baseline

        Returns:
            FileObservation with current state
        """
        return self._workspace_state.observe_file(
            resolved_path, source, session_id, store_baseline
        )

    def begin_transaction(
        self, transaction_id: str | None = None
    ) -> TransactionJournal:
        """Begin a new transaction.

        Args:
            transaction_id: Optional transaction ID (generated if not provided)

        Returns:
            TransactionJournal for the new transaction

        Raises:
            RuntimeError: If a transaction is already active
        """
        if self._active_transaction is not None:
            raise RuntimeError(
                f"Transaction already active: {self._active_transaction.transaction_id}"
            )

        transaction_id = transaction_id or str(uuid4())
        transaction = TransactionJournal(
            transaction_id=transaction_id,
            workspace_id=self.workspace_id,
            state=TransactionState.ACTIVE,
        )

        self._active_transaction = transaction
        self._workspace_state.active_transaction_id = transaction_id

        logger.info(
            "Began transaction",
            transaction_id=transaction_id,
            workspace_id=self.workspace_id,
        )

        return transaction

    def commit_transaction(self) -> ChangeSet:
        """Commit the active transaction.

        Returns:
            ChangeSet of all changes made

        Raises:
            RuntimeError: If no transaction is active
        """
        if self._active_transaction is None:
            raise RuntimeError("No active transaction to commit")

        transaction = self._active_transaction

        # Build change set from completed entries
        changeset = self._build_changeset(transaction)

        # Update transaction state
        transaction.state = TransactionState.COMMITTED
        transaction.committed_at = datetime.utcnow()

        # Store in history
        self._transaction_history[transaction.transaction_id] = transaction

        # Clear active transaction
        self._active_transaction = None
        self._workspace_state.active_transaction_id = None

        logger.info(
            "Committed transaction",
            transaction_id=transaction.transaction_id,
            changes=changeset.total_files_changed,
        )

        return changeset

    def rollback_transaction(self) -> None:
        """Rollback the active transaction.

        Raises:
            RuntimeError: If no transaction is active
        """
        if self._active_transaction is None:
            raise RuntimeError("No active transaction to rollback")

        transaction = self._active_transaction

        # Rollback completed operations in reverse order
        completed_entries = reversed(transaction.get_completed_entries())

        for entry in completed_entries:
            try:
                self._rollback_entry(entry)
                entry.status = "rolled_back"
            except Exception as e:
                logger.error(
                    "Failed to rollback entry",
                    entry_id=entry.entry_id,
                    error=str(e),
                )

        # Update transaction state
        transaction.state = TransactionState.ROLLED_BACK
        transaction.rolled_back_at = datetime.utcnow()

        # Store in history
        self._transaction_history[transaction.transaction_id] = transaction

        # Clear active transaction
        self._active_transaction = None
        self._workspace_state.active_transaction_id = None

        logger.info(
            "Rolled back transaction",
            transaction_id=transaction.transaction_id,
        )

    def _rollback_entry(self, entry: TransactionJournalEntry) -> None:
        """Rollback a single journal entry.

        Args:
            entry: Entry to rollback
        """
        if entry.before_observation is None:
            # File was created, delete it
            full_path = self.workspace_root / entry.path
            if full_path.exists():
                full_path.unlink()
                logger.debug("Rolled back file creation", path=entry.path)
        else:
            # Restore from snapshot if available
            snapshot_id = entry.metadata.get("snapshot_id")
            if snapshot_id is not None:
                try:
                    self.snapshot_manager.restore_file_snapshot(
                        self.workspace_root,
                        str(snapshot_id),
                        entry.path,
                    )
                    logger.debug("Rolled back file from snapshot", path=entry.path)
                except Exception as e:
                    logger.error(
                        "Failed to restore from snapshot",
                        path=entry.path,
                        error=str(e),
                    )

    def record_operation(
        self,
        operation: OperationType,
        file_path: str,
        before_observation: FileObservation | None = None,
        after_observation: FileObservation | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TransactionJournalEntry:
        """Record an operation in the transaction journal.

        Args:
            operation: Type of operation
            file_path: Relative path to file
            before_observation: File state before operation
            after_observation: File state after operation
            metadata: Additional operation metadata

        Returns:
            TransactionJournalEntry for the operation

        Raises:
            RuntimeError: If no transaction is active
        """
        if self._active_transaction is None:
            raise RuntimeError("No active transaction")

        entry = TransactionJournalEntry(
            entry_id=str(uuid4()),
            transaction_id=self._active_transaction.transaction_id,
            operation=operation.value,
            path=file_path,
            before_observation=before_observation,
            after_observation=after_observation,
            status="completed",
            metadata=metadata or {},
        )

        self._active_transaction.add_entry(entry)

        logger.debug(
            "Recorded operation",
            operation=operation.value,
            file_path=file_path,
            entry_id=entry.entry_id,
        )

        return entry

    def atomic_write(
        self,
        resolved_path: Path,
        content: str,
        expected_hash: str | None = None,
        create_snapshot: bool = True,
        session_id: str | None = None,
    ) -> tuple[FileObservation | None, FileObservation]:
        """Perform an atomic write within the transaction.

        Args:
            resolved_path: Resolved path to file
            content: New file content
            expected_hash: Expected hash before write (for validation)
            create_snapshot: Whether to create a snapshot before write
            session_id: Session ID for tracking

        Returns:
            Tuple of (before_observation, after_observation). before_observation
            is None if the file was newly created.

        Raises:
            RuntimeError: If no transaction is active
            ValueError: If expected hash doesn't match
        """
        if self._active_transaction is None:
            raise RuntimeError("No active transaction")

        rel_path = self._workspace_state.get_relative_path(resolved_path)

        # Observe before state
        before_observation = self._workspace_state.get_file_version(
            rel_path, session_id=session_id
        )

        # Fall back to reading from disk when the file exists but was never
        # observed, so rollback can restore the pre-transaction content.
        if before_observation is None and resolved_path.exists():
            before_observation = self.observe_file(
                resolved_path, session_id=session_id, store_baseline=False
            )

        # Validate expected hash if provided (only for existing files)
        if before_observation is not None and expected_hash is not None:
            if before_observation.hash != expected_hash:
                error = f"Hash mismatch for {rel_path}: expected {expected_hash}, got {before_observation.hash}"
                logger.error(error)
                raise ValueError(error)

        # Create snapshot if enabled (only for existing files)
        snapshot_id = None
        if (
            create_snapshot
            and self.config.enable_snapshots
            and before_observation is not None
        ):
            try:
                snapshot = self.snapshot_manager.create_file_snapshot(
                    self.workspace_root, rel_path
                )
                snapshot_id = snapshot.snapshot_id
            except Exception as e:
                logger.error(
                    "Failed to create snapshot before write, aborting operation",
                    file_path=rel_path,
                    error=str(e),
                )
                raise RuntimeError(
                    f"Snapshot creation failed for {rel_path}, write aborted to ensure rollback capability"
                ) from e

        # Perform write
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        with resolved_path.open(mode="w", encoding="utf-8", newline="") as f:
            f.write(content)

        # Observe after state
        after_observation = self.observe_file(resolved_path, session_id=session_id)

        # Record operation
        self.record_operation(
            operation=OperationType.WRITE,
            file_path=rel_path,
            before_observation=before_observation,
            after_observation=after_observation,
            metadata={"snapshot_id": snapshot_id},
        )

        return before_observation, after_observation

    def atomic_create(
        self,
        resolved_path: Path,
        content: str,
        session_id: str | None = None,
    ) -> FileObservation:
        """Perform an atomic file creation within the transaction.

        Args:
            resolved_path: Resolved path to file
            content: File content
            session_id: Session ID for tracking

        Returns:
            After observation

        Raises:
            RuntimeError: If no transaction is active
            FileExistsError: If file already exists
        """
        if self._active_transaction is None:
            raise RuntimeError("No active transaction")

        rel_path = self._workspace_state.get_relative_path(resolved_path)

        if resolved_path.exists():
            raise FileExistsError(f"File already exists: {rel_path}")

        # Perform creation
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_path.open(mode="w", encoding="utf-8", newline="") as f:
            f.write(content)

        # Observe after state
        after_observation = self.observe_file(resolved_path, session_id=session_id)

        # Record operation
        self.record_operation(
            operation=OperationType.CREATE,
            file_path=rel_path,
            before_observation=None,
            after_observation=after_observation,
        )

        return after_observation

    def atomic_delete(
        self,
        resolved_path: Path,
        create_snapshot: bool = True,
        session_id: str | None = None,
    ) -> FileObservation | None:
        """Perform an atomic file deletion within the transaction.

        Args:
            resolved_path: Resolved path to file
            create_snapshot: Whether to create a snapshot before deletion
            session_id: Session ID for tracking

        Returns:
            Before observation

        Raises:
            RuntimeError: If no transaction is active
            FileNotFoundError: If file doesn't exist
            ValueError: If expected hash doesn't match
        """
        if self._active_transaction is None:
            raise RuntimeError("No active transaction")

        rel_path = self._workspace_state.get_relative_path(resolved_path)

        if not resolved_path.exists():
            raise FileNotFoundError(f"File not found: {rel_path}")

        # Observe before state
        before_observation = self._workspace_state.get_file_version(
            rel_path,
            session_id=session_id,
        )
        # Fall back to reading from disk when the file exists but was never
        # observed, so rollback can restore the pre-transaction content.
        if before_observation is None and resolved_path.exists():
            before_observation = self.observe_file(
                resolved_path, session_id=session_id, store_baseline=False
            )

        # Create snapshot if enabled
        snapshot_id = None
        if create_snapshot and self.config.enable_snapshots:
            try:
                snapshot = self.snapshot_manager.create_file_snapshot(
                    self.workspace_root, rel_path
                )
                snapshot_id = snapshot.snapshot_id
            except Exception as e:
                logger.error(
                    "Failed to create snapshot before delete, aborting operation",
                    file_path=rel_path,
                    error=str(e),
                )
                raise RuntimeError(
                    f"Snapshot creation failed for {rel_path}, delete aborted to ensure rollback capability"
                ) from e

        # Perform deletion
        resolved_path.unlink()

        # Remove from workspace state
        self._workspace_state.remove_file_version(
            rel_path,
            session_id=session_id,
        )

        # Record operation
        self.record_operation(
            operation=OperationType.DELETE,
            file_path=rel_path,
            before_observation=before_observation,
            after_observation=None,
            metadata={"snapshot_id": snapshot_id},
        )

        return before_observation

    def _build_changeset(self, transaction: TransactionJournal) -> ChangeSet:
        """Build a change set from transaction entries.

        Args:
            transaction: Transaction to build changeset from

        Returns:
            ChangeSet with all changes
        """
        changes: list[FileChange] = []
        total_bytes = 0

        for entry in transaction.get_completed_entries():
            if entry.operation == OperationType.WRITE:
                # Modify operation
                after_size = (
                    entry.after_observation.size if entry.after_observation else None
                )
                before_size = (
                    entry.before_observation.size if entry.before_observation else None
                )
                change = FileChange(
                    path=entry.path,
                    before_hash=entry.before_observation.hash
                    if entry.before_observation
                    else None,
                    after_hash=entry.after_observation.hash
                    if entry.after_observation
                    else None,
                    before_size=before_size,
                    after_size=after_size,
                    change_type="modify",
                )
                total_bytes = (
                    total_bytes + abs(after_size - before_size)
                    if after_size is not None and before_size is not None
                    else 0
                )
            elif entry.operation == OperationType.CREATE:
                # Create operation
                change = FileChange(
                    path=entry.path,
                    before_hash=None,
                    after_hash=entry.after_observation.hash
                    if entry.after_observation
                    else None,
                    before_size=None,
                    after_size=entry.after_observation.size
                    if entry.after_observation
                    else None,
                    change_type="create",
                )
                total_bytes = total_bytes + (
                    entry.after_observation.size if entry.after_observation else 0
                )
            elif entry.operation == OperationType.DELETE:
                # Delete operation
                change = FileChange(
                    path=entry.path,
                    before_hash=entry.before_observation.hash
                    if entry.before_observation
                    else None,
                    after_hash=None,
                    before_size=entry.before_observation.size
                    if entry.before_observation
                    else None,
                    after_size=None,
                    change_type="delete",
                )
            else:
                continue

            changes.append(change)

        return ChangeSet(
            changes=changes,
            total_bytes_changed=total_bytes,
            total_files_changed=len({change.path for change in changes}),
        )

    def get_active_transaction(self) -> TransactionJournal | None:
        """Get the currently active transaction.

        Returns:
            Active transaction or None
        """
        return self._active_transaction

    def get_transaction_history(self) -> dict[str, TransactionJournal]:
        """Get transaction history.

        Returns:
            Dictionary mapping transaction IDs to journals
        """
        return self._transaction_history.copy()

    def get_workspace_state(self) -> WorkspaceState:
        """Get the current workspace state.

        Returns:
            WorkspaceState with current file versions
        """
        return self._workspace_state
