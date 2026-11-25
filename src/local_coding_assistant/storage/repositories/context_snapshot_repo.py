"""Context Snapshot repository for execution store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from local_coding_assistant.runtime.context.profile import ContextSnapshot
from local_coding_assistant.storage.contracts import StoredContextSnapshot
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("storage.repositories.context_snapshot_repo")


class ContextSnapshotRepo:
    """Repository for context snapshot operations."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def create(
        self,
        snapshot_id: str,
        turn_id: str,
        schema_version: int = 1,
        tool_set_ref: str | None = None,
        template_set_ref: str | None = None,
        repo_snapshot_ref: str | None = None,
        memory_snapshot_ref: str | None = None,
        message_boundary_json: str | None = None,
        content_hash: str = "",
        payload_json: str | None = None,
        created_at_ms: int | None = None,
    ) -> None:
        """Create a new context snapshot."""
        if created_at_ms is None:
            from datetime import UTC, datetime

            created_at_ms = int(datetime.now(UTC).timestamp() * 1000)

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO context_snapshots (id, turn_id, schema_version, tool_set_ref, template_set_ref,
                                           repo_snapshot_ref, memory_snapshot_ref,
                                           message_boundary_json, content_hash, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                turn_id,
                schema_version,
                tool_set_ref,
                template_set_ref,
                repo_snapshot_ref,
                memory_snapshot_ref,
                message_boundary_json,
                content_hash,
                payload_json,
                created_at_ms,
            ),
        )

    def _row_to_stored(self, row: aiosqlite.Row) -> StoredContextSnapshot:
        """Convert database row to StoredContextSnapshot."""
        return StoredContextSnapshot(
            id=row[0],
            turn_id=row[1],
            schema_version=row[2],
            tool_set_ref=row[3],
            template_set_ref=row[4],
            repo_snapshot_ref=row[5],
            memory_snapshot_ref=row[6],
            message_boundary_json=row[7],
            content_hash=row[8],
            payload_json=row[9],
            created_at_ms=row[10],
        )

    async def get(self, snapshot_id: str) -> ContextSnapshot | None:
        """Get context snapshot by ID."""
        cursor = await self._conn.execute(
            """
            SELECT id, turn_id, schema_version, tool_set_ref, template_set_ref,
                   repo_snapshot_ref, memory_snapshot_ref,
                   message_boundary_json, content_hash, payload_json, created_at
            FROM context_snapshots WHERE id = ?
            """,
            (snapshot_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        stored = self._row_to_stored(row)
        return stored.to_runtime()

    async def get_by_turn(self, turn_id: str) -> ContextSnapshot | None:
        """Get context snapshot for a turn."""
        cursor = await self._conn.execute(
            """
            SELECT id, turn_id, schema_version, tool_set_ref, template_set_ref,
                   repo_snapshot_ref, memory_snapshot_ref,
                   message_boundary_json, content_hash, payload_json, created_at
            FROM context_snapshots WHERE turn_id = ?
            """,
            (turn_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        stored = self._row_to_stored(row)
        return stored.to_runtime()

    async def get_stored(self, snapshot_id: str) -> StoredContextSnapshot | None:
        """Get context snapshot as StoredContextSnapshot (without parsing refs)."""
        cursor = await self._conn.execute(
            """
            SELECT id, turn_id, schema_version, tool_set_ref, template_set_ref,
                   repo_snapshot_ref, memory_snapshot_ref,
                   message_boundary_json, content_hash, payload_json, created_at
            FROM context_snapshots WHERE id = ?
            """,
            (snapshot_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_stored(row)

    async def get_stored_by_turn(self, turn_id: str) -> StoredContextSnapshot | None:
        """Get context snapshot as StoredContextSnapshot by turn ID."""
        cursor = await self._conn.execute(
            """
            SELECT id, turn_id, schema_version, tool_set_ref, template_set_ref,
                   repo_snapshot_ref, memory_snapshot_ref,
                   message_boundary_json, content_hash, payload_json, created_at
            FROM context_snapshots WHERE turn_id = ?
            """,
            (turn_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_stored(row)

    async def upsert(
        self,
        snapshot_id: str,
        turn_id: str,
        schema_version: int = 1,
        tool_set_ref: str | None = None,
        template_set_ref: str | None = None,
        repo_snapshot_ref: str | None = None,
        memory_snapshot_ref: str | None = None,
        message_boundary_json: str | None = None,
        content_hash: str = "",
        payload_json: str | None = None,
        created_at_ms: int | None = None,
    ) -> None:
        """Insert or update context snapshot in single operation."""
        if created_at_ms is None:
            from datetime import UTC, datetime

            created_at_ms = int(datetime.now(UTC).timestamp() * 1000)
        await self._conn.execute(
            """
            INSERT INTO context_snapshots (id, turn_id, schema_version, tool_set_ref, template_set_ref,
                                           repo_snapshot_ref, memory_snapshot_ref,
                                           message_boundary_json, content_hash, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                tool_set_ref = excluded.tool_set_ref,
                template_set_ref = excluded.template_set_ref,
                repo_snapshot_ref = excluded.repo_snapshot_ref,
                memory_snapshot_ref = excluded.memory_snapshot_ref,
                message_boundary_json = excluded.message_boundary_json,
                content_hash = excluded.content_hash,
                payload_json = excluded.payload_json
            """,
            (
                snapshot_id,
                turn_id,
                schema_version,
                tool_set_ref,
                template_set_ref,
                repo_snapshot_ref,
                memory_snapshot_ref,
                message_boundary_json,
                content_hash,
                payload_json,
                created_at_ms,
            ),
        )
