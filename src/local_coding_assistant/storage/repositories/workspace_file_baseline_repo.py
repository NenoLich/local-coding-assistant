"""Workspace file baseline repository for execution store."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import aiosqlite

if TYPE_CHECKING:
    pass

from local_coding_assistant.utils.logging import get_logger

logger = get_logger("storage.repositories.workspace_file_baseline_repo")


class WorkspaceFileBaselineRepo:
    """Repository for workspace file baseline operations."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def upsert(
        self,
        workspace_id: str,
        observation: dict[str, Any],
        turn_id: str | None,
        frame_id: str | None,
        session_id: str | None = None,
    ) -> None:
        """Store or update a baseline file observation.

        Args:
            workspace_id: Workspace identifier
            observation: File observation data as dict
            turn_id: Optional turn ID
            frame_id: Optional frame ID
            session_id: Optional session ID
        """
        source_value = (
            observation.get("source", {}).get("value", "unknown")
            if isinstance(observation.get("source"), dict)
            else observation.get("source", "unknown")
        )
        await self._conn.execute(
            """
            INSERT INTO workspace_file_baselines (
                workspace_id, session_id, file_path, hash, size, mtime, inode, source, observed_at, turn_id, frame_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workspace_id, session_id, file_path) DO UPDATE SET
                hash = excluded.hash,
                size = excluded.size,
                mtime = excluded.mtime,
                inode = excluded.inode,
                source = excluded.source,
                observed_at = excluded.observed_at,
                turn_id = excluded.turn_id,
                frame_id = excluded.frame_id
            """,
            (
                workspace_id,
                session_id,
                observation["path"],
                observation["hash"],
                observation["size"],
                observation["mtime"].isoformat()
                if isinstance(observation["mtime"], datetime)
                else observation["mtime"],
                observation.get("inode"),
                source_value,
                observation["observed_at"].isoformat()
                if isinstance(observation["observed_at"], datetime)
                else observation["observed_at"],
                turn_id,
                frame_id,
            ),
        )

    async def load(
        self, workspace_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Load all baseline observations for a workspace and optionally a session.

        Args:
            workspace_id: Workspace identifier
            session_id: Optional session ID to filter by

        Returns:
            List of dict observations (WRITE and DELETE sources are filtered out)
        """
        if session_id:
            cursor = await self._conn.execute(
                """
                SELECT file_path, hash, size, mtime, inode, source, observed_at
                FROM workspace_file_baselines
                WHERE workspace_id = ? AND session_id = ?
                AND source NOT IN ('write', 'delete')
                """,
                (workspace_id, session_id),
            )
        else:
            cursor = await self._conn.execute(
                """
                SELECT file_path, hash, size, mtime, inode, source, observed_at
                FROM workspace_file_baselines
                WHERE workspace_id = ?
                AND source NOT IN ('write', 'delete')
                """,
                (workspace_id,),
            )
        rows = await cursor.fetchall()

        baselines: list[dict[str, Any]] = []
        for row in rows:
            file_path, hash_val, size, mtime_str, inode, source_str, observed_at_str = (
                row
            )
            baselines.append(
                {
                    "path": file_path,
                    "hash": hash_val,
                    "size": size,
                    "mtime": mtime_str,
                    "inode": inode,
                    "source": {"value": source_str},
                    "observed_at": observed_at_str,
                }
            )
        return baselines

    async def get_by_turn_id(
        self, session_id: str, turn_ids: set[str]
    ) -> list[tuple[str, str, str, int, str, str, str, str, str, str | None]]:
        """Get baseline observations for specific turn IDs in a session.

        Args:
            session_id: Session identifier
            turn_ids: Set of turn identifiers

        Returns:
            List of tuples containing (workspace_id, file_path, hash, size, mtime, inode, source, observed_at, turn_id, frame_id)
        """
        placeholders = ",".join("?" for _ in turn_ids)
        query = (
            "SELECT workspace_id, file_path, hash, size, mtime, inode, source, observed_at, turn_id, frame_id FROM workspace_file_baselines WHERE session_id = ? AND turn_id IN ("  # noqa S608
            + placeholders
            + ")"
        )
        cursor = await self._conn.execute(query, (session_id, *tuple(turn_ids)))

        rows = await cursor.fetchall()
        if not rows:
            return []

        return [
            (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                row[8],
                row[9],
            )
            for row in rows
        ]

    async def delete(self, workspace_id: str, file_path: str) -> None:
        """Delete a baseline observation.

        Args:
            workspace_id: Workspace identifier
            file_path: File path to delete
        """
        await self._conn.execute(
            "DELETE FROM workspace_file_baselines WHERE workspace_id = ? AND file_path = ?",
            (workspace_id, file_path),
        )
