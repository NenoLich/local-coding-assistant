"""Workspace identity repository for execution store.

Owns the persisted mapping between a physical workspace root and a stable
workspace_id. The id is a stored UUID so session resume and file-baseline
lookups survive process restarts, path renames, symlinks and config drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import aiosqlite

from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("storage.repositories.workspace_repo")


@dataclass(frozen=True, slots=True)
class WorkspaceRecord:
    """A persisted workspace identity record."""

    id: str
    workspace_root: str
    config_hash: str
    updated_at: int


class WorkspaceRepo:
    """Repository for workspace identity operations."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def get_by_root(self, workspace_root: str) -> WorkspaceRecord | None:
        """Find an existing workspace record for the given root.

        Args:
            workspace_root: Absolute workspace root path

        Returns:
            The matching WorkspaceRecord, or None if not present
        """
        cursor = await self._conn.execute(
            """
            SELECT id, workspace_root, config_hash, updated_at
            FROM workspaces WHERE workspace_root = ?
            """,
            (workspace_root,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        now = int(datetime.now(UTC).timestamp() * 1000)
        await self._conn.execute(
            "UPDATE workspaces SET updated_at = ? WHERE workspace_root = ?",
            (now, workspace_root),
        )
        await self._conn.commit()
        return WorkspaceRecord(
            id=row[0],
            workspace_root=row[1],
            config_hash=row[2],
            updated_at=now,
        )

    async def create(
        self,
        workspace_id: str,
        workspace_root: str,
        config_hash: str,
    ) -> WorkspaceRecord:
        """Insert a new workspace record.

        Args:
            workspace_id: Stable UUID for the workspace
            workspace_root: Absolute workspace root path
            config_hash: Hash of the effective config at creation time

        Returns:
            The inserted WorkspaceRecord
        """
        updated_at = int(datetime.now(UTC).timestamp() * 1000)
        await self._conn.execute(
            """
            INSERT OR IGNORE INTO workspaces (id, workspace_root, config_hash, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (workspace_id, workspace_root, config_hash, updated_at),
        )
        await self._conn.commit()
        return WorkspaceRecord(
            id=workspace_id,
            workspace_root=workspace_root,
            config_hash=config_hash,
            updated_at=updated_at,
        )

    async def update_config_hash(self, workspace_id: str, config_hash: str) -> None:
        """Refresh the recorded config hash for an existing workspace.

        Called when a known workspace is re-opened and the effective config has
        drifted; this lets callers detect drift against the stored value
        instead of silently invalidating history.

        Args:
            workspace_id: Workspace to update
            config_hash: New effective config hash
        """
        now = int(datetime.now(UTC).timestamp() * 1000)
        await self._conn.execute(
            "UPDATE workspaces SET config_hash = ?, updated_at = ? WHERE id = ?",
            (config_hash, now, workspace_id),
        )
        await self._conn.commit()
