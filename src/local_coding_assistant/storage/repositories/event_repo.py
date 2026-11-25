"""Event repository for execution store."""

from __future__ import annotations

from typing import Any

import aiosqlite

from local_coding_assistant.utils.logging import get_logger

logger = get_logger("storage.repositories.event_repo")


class EventRepo:
    """Repository for event operations."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def get_max_seq(self, session_id: str) -> int:
        """Get the highest persisted event sequence for a session."""
        cursor = await self._conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM events WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return 0
        return int(row[0])

    async def append(
        self,
        trace_id: str,
        session_id: str,
        turn_id: str | None,
        frame_id: str | None,
        seq: int,
        event_type: str,
        data_json: str,
        timestamp_ms: int,
    ) -> None:
        """Append an event."""
        await self._conn.execute(
            """
            INSERT INTO events (trace_id, session_id, turn_id, frame_id, seq, type, data_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                session_id,
                turn_id,
                frame_id,
                seq,
                event_type,
                data_json,
                timestamp_ms,
            ),
        )

    async def get_by_frame(self, frame_id: str) -> list[dict[str, Any]]:
        """Get all events for a frame."""
        cursor = await self._conn.execute(
            """
            SELECT id, trace_id, session_id, turn_id, frame_id, seq, type, data_json, timestamp
            FROM events WHERE frame_id = ? ORDER BY id
            """,
            (frame_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "trace_id": row[1],
                "session_id": row[2],
                "turn_id": row[3],
                "frame_id": row[4],
                "seq": row[5],
                "type": row[6],
                "data_json": row[7],
                "timestamp": row[8],
            }
            for row in rows
        ]

    async def get_by_turn(self, turn_id: str) -> list[dict[str, Any]]:
        """Get all events for a turn."""
        cursor = await self._conn.execute(
            """
            SELECT id, trace_id, session_id, turn_id, frame_id, seq, type, data_json, timestamp
            FROM events WHERE turn_id = ? ORDER BY id
            """,
            (turn_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "trace_id": row[1],
                "session_id": row[2],
                "turn_id": row[3],
                "frame_id": row[4],
                "seq": row[5],
                "type": row[6],
                "data_json": row[7],
                "timestamp": row[8],
            }
            for row in rows
        ]

    async def get_by_frames(self, frame_ids: list[str]) -> list[dict[str, Any]]:
        """Get all events for multiple frames in a single query.

        Args:
            frame_ids: Frame identifiers to fetch events for (chunked IN clause)
        """
        if not frame_ids:
            return []

        results: list[dict[str, Any]] = []
        chunk_size = 100
        for start in range(0, len(frame_ids), chunk_size):
            chunk = frame_ids[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT id, trace_id, session_id, turn_id, frame_id, seq, type, data_json, timestamp FROM events WHERE frame_id IN ("  # noqa S608
                + placeholders
                + ") ORDER BY id"
            )
            cursor = await self._conn.execute(query, chunk)

            rows = await cursor.fetchall()
            results.extend(
                {
                    "id": row[0],
                    "trace_id": row[1],
                    "session_id": row[2],
                    "turn_id": row[3],
                    "frame_id": row[4],
                    "seq": row[5],
                    "type": row[6],
                    "data_json": row[7],
                    "timestamp": row[8],
                }
                for row in rows
            )
        return results

    async def get_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """Get all events for a session."""
        cursor = await self._conn.execute(
            """
            SELECT id, trace_id, session_id, turn_id, frame_id, seq, type, data_json, timestamp
            FROM events WHERE session_id = ? ORDER BY id
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "trace_id": row[1],
                "session_id": row[2],
                "turn_id": row[3],
                "frame_id": row[4],
                "seq": row[5],
                "type": row[6],
                "data_json": row[7],
                "timestamp": row[8],
            }
            for row in rows
        ]

    async def get_by_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """Get all events for a trace."""
        cursor = await self._conn.execute(
            """
            SELECT id, trace_id, session_id, turn_id, frame_id, seq, type, data_json, timestamp
            FROM events WHERE trace_id = ? ORDER BY id
            """,
            (trace_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "trace_id": row[1],
                "session_id": row[2],
                "turn_id": row[3],
                "frame_id": row[4],
                "seq": row[5],
                "type": row[6],
                "data_json": row[7],
                "timestamp": row[8],
            }
            for row in rows
        ]

    async def get_by_type(
        self, event_type: str, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Get recent events by type."""
        cursor = await self._conn.execute(
            """
            SELECT id, trace_id, session_id, turn_id, frame_id, seq, type, data_json, timestamp
            FROM events WHERE type = ? ORDER BY timestamp DESC LIMIT ?
            """,
            (event_type, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "trace_id": row[1],
                "session_id": row[2],
                "turn_id": row[3],
                "frame_id": row[4],
                "seq": row[5],
                "type": row[6],
                "data_json": row[7],
                "timestamp": row[8],
            }
            for row in rows
        ]

    async def delete_by_frame(self, frame_id: str) -> None:
        """Delete all events for a frame."""
        await self._conn.execute("DELETE FROM events WHERE frame_id = ?", (frame_id,))
