"""Message repository for execution store.

Messages are a pure projection of runtime events. Each write carries a
``dedupe_key`` derived from its source event so re-projecting an event is a
no-op, and ``seq`` is computed in SQL as ``MAX(seq) + 1`` per session when
not supplied, so batches stay gap-free and idempotent within a transaction.
"""

from __future__ import annotations

from typing import Any

import aiosqlite

from local_coding_assistant.utils.logging import get_logger

logger = get_logger("storage.repositories.message_repo")


class MessageRepo:
    """Repository for message operations."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def append(
        self,
        session_id: str,
        turn_id: str,
        frame_id: str,
        seq: int | None,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        metadata_json: str | None = None,
        created_at_ms: int | None = None,
        dedupe_key: str | None = None,
    ) -> None:
        """Append a message (idempotent per (session_id, dedupe_key)).

        Args:
            session_id: Session identifier
            turn_id: Turn identifier
            frame_id: Frame identifier (turn-level messages use "turn_start")
            seq: Explicit sequence number, or None to compute MAX(seq)+1
                for the session inside this transaction
            role: Message role (user/assistant/tool)
            content: Message content
            tool_call_id: Tool call ID for tool messages
            tool_name: Tool name for tool messages
            metadata_json: Optional metadata payload (e.g. tool_calls)
            created_at_ms: Creation timestamp (epoch ms), auto-filled if None
            dedupe_key: Stable identity from the source event; duplicates
                are ignored
        """
        if created_at_ms is None:
            from datetime import UTC, datetime

            created_at_ms = int(datetime.now(UTC).timestamp() * 1000)

        await self._conn.execute(
            """
            INSERT OR IGNORE INTO messages (
                session_id, turn_id, frame_id, seq, role, content,
                tool_call_id, tool_name, metadata_json, dedupe_key, created_at
            )
            VALUES (
                ?, ?, ?,
                COALESCE(?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE session_id = ?)),
                ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                session_id,
                turn_id,
                frame_id,
                seq,
                session_id,
                role,
                content,
                tool_call_id,
                tool_name,
                metadata_json,
                dedupe_key,
                created_at_ms,
            ),
        )

    async def get_by_frame(self, frame_id: str) -> list[dict[str, Any]]:
        """Get all messages for a frame."""
        cursor = await self._conn.execute(
            """
            SELECT id, session_id, turn_id, frame_id, seq, role, content, tool_call_id, tool_name, metadata_json, dedupe_key, created_at
            FROM messages WHERE frame_id = ? ORDER BY seq
            """,
            (frame_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    async def get_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """Get all messages for a session."""
        cursor = await self._conn.execute(
            """
            SELECT id, session_id, turn_id, frame_id, seq, role, content, tool_call_id, tool_name, metadata_json, dedupe_key, created_at
            FROM messages WHERE session_id = ? ORDER BY seq
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    async def get_by_range(
        self, session_id: str, start_seq: int, end_seq: int
    ) -> list[dict[str, Any]]:
        """Get messages in sequence range."""
        cursor = await self._conn.execute(
            """
            SELECT id, session_id, turn_id, frame_id, seq, role, content, tool_call_id, tool_name, metadata_json, dedupe_key, created_at
            FROM messages WHERE session_id = ? AND seq >= ? AND seq <= ? ORDER BY seq
            """,
            (session_id, start_seq, end_seq),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    async def delete_by_frame(self, frame_id: str) -> None:
        """Delete all messages for a frame."""
        await self._conn.execute("DELETE FROM messages WHERE frame_id = ?", (frame_id,))

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "id": row[0],
            "session_id": row[1],
            "turn_id": row[2],
            "frame_id": row[3],
            "seq": row[4],
            "role": row[5],
            "content": row[6],
            "tool_call_id": row[7],
            "tool_name": row[8],
            "metadata_json": row[9],
            "dedupe_key": row[10],
            "created_at": row[11],
        }
