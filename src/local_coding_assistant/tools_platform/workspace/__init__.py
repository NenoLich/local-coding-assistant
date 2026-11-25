"""Workspace management and transaction support."""

from local_coding_assistant.tools_platform.workspace.conflicts import (
    ConflictDetectionConfig,
    ConflictDetector,
    ConflictType,
)
from local_coding_assistant.tools_platform.workspace.manager import WorkspaceManager
from local_coding_assistant.tools_platform.workspace.models import (
    ChangeOrigin,
    ConflictResolution,
    PathPolicy,
    TransactionJournal,
    TransactionJournalEntry,
    WorkspaceSnapshot,
    WorkspaceState,
)
from local_coding_assistant.tools_platform.workspace.snapshots import (
    SnapshotManager,
    SnapshotStorageConfig,
)
from local_coding_assistant.tools_platform.workspace.transactions import (
    OperationType,
    TransactionConfig,
    TransactionManager,
)

__all__ = [
    # Models
    "ChangeOrigin",
    # Conflicts
    "ConflictDetectionConfig",
    "ConflictDetector",
    "ConflictResolution",
    "ConflictType",
    "OperationType",
    "PathPolicy",
    # Snapshots
    "SnapshotManager",
    "SnapshotStorageConfig",
    "TransactionConfig",
    "TransactionJournal",
    "TransactionJournalEntry",
    # Transactions
    "TransactionManager",
    # Manager
    "WorkspaceManager",
    "WorkspaceSnapshot",
    "WorkspaceState",
]
