"""Turn repository for execution store."""

from __future__ import annotations

import aiosqlite

from local_coding_assistant.storage.contracts import StoredTurn
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("storage.repositories.turn_repo")


class TurnRepo:
    """Repository for turn operations."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def create(
        self,
        turn_id: str,
        session_id: str,
        trace_id: str,
        turn_index: int,
        user_input: str,
        status: str,
        started_at_ms: int,
        context_snapshot_id: str | None = None,
        config_snapshot_json: str | None = None,
        final_answer: str | None = None,
    ) -> None:
        """Create a new turn."""
        await self._conn.execute(
            """
            INSERT INTO turns (id, session_id, trace_id, turn_index, user_input, status, started_at,
                               context_snapshot_id, config_snapshot_json, final_answer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn_id,
                session_id,
                trace_id,
                turn_index,
                user_input,
                status,
                started_at_ms,
                context_snapshot_id,
                config_snapshot_json,
                final_answer,
            ),
        )

    async def get(self, turn_id: str) -> StoredTurn | None:
        """Get turn by ID."""
        cursor = await self._conn.execute(
            """
            SELECT id, session_id, trace_id, turn_index, user_input, status, started_at, finished_at,
                   paused_at, pause_reason, resumed_at, context_snapshot_id, config_snapshot_json, total_input_tokens, total_output_tokens, total_frames,
                   final_answer
            FROM turns WHERE id = ?
            """,
            (turn_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        return StoredTurn(
            id=row[0],
            session_id=row[1],
            trace_id=row[2],
            turn_index=row[3],
            user_input=row[4],
            status=row[5],
            started_at_ms=row[6],
            finished_at_ms=row[7],
            paused_at_ms=row[8],
            pause_reason=row[9],
            resumed_at_ms=row[10],
            context_snapshot_id=row[11],
            config_snapshot_json=row[12],
            total_input_tokens=row[13] or 0,
            total_output_tokens=row[14] or 0,
            total_frames=row[15] or 0,
            final_answer=row[16],
        )

    async def update_status(
        self,
        turn_id: str,
        status: str,
        finished_at_ms: int | None = None,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        total_frames: int | None = None,
        paused_at_ms: int | None = None,
        pause_reason: str | None = None,
        resumed_at_ms: int | None = None,
        final_answer: str | None = None,
    ) -> None:
        """Update turn status and metadata."""
        await self._conn.execute(
            """
            UPDATE turns
            SET status = ?, finished_at = ?,
                paused_at = COALESCE(?, paused_at),
                pause_reason = COALESCE(?, pause_reason),
                resumed_at = COALESCE(?, resumed_at),
                total_input_tokens = COALESCE(?, total_input_tokens),
                total_output_tokens = COALESCE(?, total_output_tokens),
                total_frames = COALESCE(?, total_frames),
                final_answer = COALESCE(?, final_answer)
            WHERE id = ?
            """,
            (
                status,
                finished_at_ms,
                paused_at_ms,
                pause_reason,
                resumed_at_ms,
                total_input_tokens,
                total_output_tokens,
                total_frames,
                final_answer,
                turn_id,
            ),
        )

    async def increment_totals(
        self,
        turn_id: str,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        total_frames: int = 0,
    ) -> None:
        """Atomically add usage deltas to turn totals."""
        await self._conn.execute(
            """
            UPDATE turns
            SET total_input_tokens = COALESCE(total_input_tokens, 0) + ?,
                total_output_tokens = COALESCE(total_output_tokens, 0) + ?,
                total_frames = COALESCE(total_frames, 0) + ?
            WHERE id = ?
            """,
            (total_input_tokens, total_output_tokens, total_frames, turn_id),
        )

    async def list_by_session(self, session_id: str) -> list[StoredTurn]:
        """List all turns for a session."""
        cursor = await self._conn.execute(
            """
            SELECT id, session_id, trace_id, turn_index, user_input, status, started_at, finished_at,
                   paused_at, pause_reason, resumed_at, context_snapshot_id, config_snapshot_json, total_input_tokens, total_output_tokens, total_frames,
                   final_answer
            FROM turns WHERE session_id = ? ORDER BY turn_index
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            StoredTurn(
                id=row[0],
                session_id=row[1],
                trace_id=row[2],
                turn_index=row[3],
                user_input=row[4],
                status=row[5],
                started_at_ms=row[6],
                finished_at_ms=row[7],
                paused_at_ms=row[8],
                pause_reason=row[9],
                resumed_at_ms=row[10],
                context_snapshot_id=row[11],
                config_snapshot_json=row[12],
                total_input_tokens=row[13] or 0,
                total_output_tokens=row[14] or 0,
                total_frames=row[15] or 0,
                final_answer=row[16],
            )
            for row in rows
        ]

    async def upsert(
        self,
        turn_id: str,
        session_id: str,
        trace_id: str,
        status: str,
        turn_index: int = 0,
        user_input: str = "",
        started_at_ms: int = 0,
        finished_at_ms: int | None = None,
        paused_at_ms: int | None = None,
        pause_reason: str | None = None,
        resumed_at_ms: int | None = None,
        context_snapshot_id: str | None = None,
        config_snapshot_json: str | None = None,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        total_frames: int | None = None,
        message_boundary_json: str | None = None,
        final_answer: str | None = None,
    ) -> None:
        """Insert or update turn in single operation."""
        await self._conn.execute(
            """
            INSERT INTO turns (id, session_id, trace_id, turn_index, user_input, status, started_at,
                               finished_at, paused_at, pause_reason, resumed_at, context_snapshot_id, config_snapshot_json, total_input_tokens, total_output_tokens, total_frames, message_boundary_json, final_answer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                finished_at = COALESCE(excluded.finished_at, turns.finished_at),
                paused_at = COALESCE(excluded.paused_at, turns.paused_at),
                pause_reason = COALESCE(excluded.pause_reason, turns.pause_reason),
                resumed_at = COALESCE(excluded.resumed_at, turns.resumed_at),
                context_snapshot_id = COALESCE(excluded.context_snapshot_id, turns.context_snapshot_id),
                config_snapshot_json = COALESCE(excluded.config_snapshot_json, turns.config_snapshot_json),
                total_input_tokens = CASE WHEN ? IS NULL THEN turns.total_input_tokens ELSE excluded.total_input_tokens END,
                total_output_tokens = CASE WHEN ? IS NULL THEN turns.total_output_tokens ELSE excluded.total_output_tokens END,
                total_frames = CASE WHEN ? IS NULL THEN turns.total_frames ELSE excluded.total_frames END,
                message_boundary_json = COALESCE(excluded.message_boundary_json, turns.message_boundary_json),
                final_answer = COALESCE(excluded.final_answer, turns.final_answer)
            """,
            (
                turn_id,
                session_id,
                trace_id,
                turn_index,
                user_input,
                status,
                started_at_ms,
                finished_at_ms,
                paused_at_ms,
                pause_reason,
                resumed_at_ms,
                context_snapshot_id,
                config_snapshot_json,
                total_input_tokens,
                total_output_tokens,
                total_frames,
                message_boundary_json,
                final_answer,
                total_input_tokens,
                total_output_tokens,
                total_frames,
            ),
        )
