"""Frame repository for execution store."""

from __future__ import annotations

import aiosqlite

from local_coding_assistant.storage.contracts import StoredFrame
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("storage.repositories.frame_repo")


class FrameRepo:
    """Repository for frame operations."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def create(
        self,
        frame_id: str,
        session_id: str,
        trace_id: str,
        iteration: int,
        status: str,
        started_at_ms: int,
        model_name: str | None = None,
        turn_id: str | None = None,
        context_snapshot_id: str | None = None,
        history_through_seq: int = 0,
        prompt_hash: str | None = None,
    ) -> None:
        """Create a new frame."""
        await self._conn.execute(
            """
            INSERT INTO frames (id, session_id, turn_id, trace_id, iteration, status, started_at, model_name, context_snapshot_id, history_through_seq, prompt_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frame_id,
                session_id,
                turn_id,
                trace_id,
                iteration,
                status,
                started_at_ms,
                model_name,
                context_snapshot_id,
                history_through_seq,
                prompt_hash,
            ),
        )

    async def get(self, frame_id: str) -> StoredFrame | None:
        """Get frame by ID."""
        cursor = await self._conn.execute(
            """
            SELECT id, session_id, turn_id, trace_id, iteration, status, started_at, finished_at,
                   duration_ms, input_tokens, output_tokens, model_name, finish_reason, error_message, pending_call_id, context_snapshot_id,
                   history_through_seq, prompt_hash
            FROM frames WHERE id = ?
            """,
            (frame_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        return StoredFrame(
            id=row[0],
            session_id=row[1],
            turn_id=row[2],
            trace_id=row[3],
            iteration=row[4],
            status=row[5],
            started_at_ms=row[6],
            finished_at_ms=row[7],
            duration_ms=row[8],
            input_tokens=row[9] or 0,
            output_tokens=row[10] or 0,
            model_name=row[11],
            finish_reason=row[12],
            error_message=row[13],
            pending_call_id=row[14],
            context_snapshot_id=row[15],
            history_through_seq=row[16],
            prompt_hash=row[17],
        )

    async def update_status(
        self,
        frame_id: str,
        status: str,
        finished_at_ms: int | None = None,
        duration_ms: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        finish_reason: str | None = None,
        error_message: str | None = None,
        pending_call_id: str | None = None,
    ) -> None:
        """Update frame status and metadata."""
        await self._conn.execute(
            """
            UPDATE frames
            SET status = ?, finished_at = ?, duration_ms = ?,
                input_tokens = COALESCE(?, input_tokens),
                output_tokens = COALESCE(?, output_tokens),
                finish_reason = ?, error_message = ?, pending_call_id = ?
            WHERE id = ?
            """,
            (
                status,
                finished_at_ms,
                duration_ms,
                input_tokens,
                output_tokens,
                finish_reason,
                error_message,
                pending_call_id,
                frame_id,
            ),
        )

    async def increment_tokens(
        self, frame_id: str, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        """Atomically add token usage to a frame."""
        await self._conn.execute(
            "UPDATE frames SET input_tokens = COALESCE(input_tokens, 0) + ?, output_tokens = COALESCE(output_tokens, 0) + ? WHERE id = ?",
            (input_tokens, output_tokens, frame_id),
        )

    async def list_by_session(self, session_id: str) -> list[StoredFrame]:
        """List all frames for a session."""
        cursor = await self._conn.execute(
            """
            SELECT id, session_id, turn_id, trace_id, iteration, status, started_at, finished_at,
                   duration_ms, input_tokens, output_tokens, model_name, finish_reason, context_snapshot_id, history_through_seq,
                   prompt_hash
            FROM frames WHERE session_id = ? ORDER BY iteration
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            StoredFrame(
                id=row[0],
                session_id=row[1],
                turn_id=row[2],
                trace_id=row[3],
                iteration=row[4],
                status=row[5],
                started_at_ms=row[6],
                finished_at_ms=row[7],
                duration_ms=row[8],
                input_tokens=row[9] or 0,
                output_tokens=row[10] or 0,
                model_name=row[11],
                finish_reason=row[12],
                context_snapshot_id=row[13],
                history_through_seq=row[14],
                prompt_hash=row[15],
            )
            for row in rows
        ]

    async def list_by_turn(self, turn_id: str) -> list[StoredFrame]:
        """List all frames for a turn (ordered by iteration)."""
        cursor = await self._conn.execute(
            """
            SELECT id, session_id, turn_id, trace_id, iteration, status, started_at, finished_at,
                   duration_ms, input_tokens, output_tokens, model_name, finish_reason, context_snapshot_id, history_through_seq,
                   prompt_hash
            FROM frames WHERE turn_id = ? ORDER BY iteration
            """,
            (turn_id,),
        )
        rows = await cursor.fetchall()
        return [
            StoredFrame(
                id=row[0],
                session_id=row[1],
                turn_id=row[2],
                trace_id=row[3],
                iteration=row[4],
                status=row[5],
                started_at_ms=row[6],
                finished_at_ms=row[7],
                duration_ms=row[8],
                input_tokens=row[9] or 0,
                output_tokens=row[10] or 0,
                model_name=row[11],
                finish_reason=row[12],
                context_snapshot_id=row[13],
                history_through_seq=row[14],
                prompt_hash=row[15],
            )
            for row in rows
        ]

    async def list_ids_by_turn(self, turn_id: str) -> list[str]:
        """List all frame IDs for a turn."""
        cursor = await self._conn.execute(
            "SELECT id FROM frames WHERE turn_id = ? ORDER BY iteration",
            (turn_id,),
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_running_frames(self) -> list[StoredFrame]:
        """Get all frames that are still running (for crash recovery)."""
        cursor = await self._conn.execute(
            """
            SELECT id, session_id, turn_id, trace_id, iteration, status, started_at,
                   history_through_seq, prompt_hash
            FROM frames WHERE status IN ('running_llm', 'running_tools')
            """
        )
        rows = await cursor.fetchall()
        return [
            StoredFrame(
                id=row[0],
                session_id=row[1],
                turn_id=row[2],
                trace_id=row[3],
                iteration=row[4],
                status=row[5],
                started_at_ms=row[6],
                history_through_seq=row[7],
                prompt_hash=row[8],
            )
            for row in rows
        ]

    async def mark_failed(self, frame_id: str, error_message: str) -> None:
        """Mark a frame as failed (for crash recovery)."""
        from datetime import UTC, datetime

        finished_at_ms = int(datetime.now(UTC).timestamp() * 1000)
        await self.update_status(
            frame_id,
            status="failed",
            finished_at_ms=finished_at_ms,
            error_message=error_message,
        )

    async def upsert(
        self,
        frame_id: str,
        session_id: str,
        turn_id: str,
        trace_id: str,
        status: str,
        iteration: int = 0,
        started_at_ms: int = 0,
        finished_at_ms: int | None = None,
        duration_ms: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        model_name: str | None = None,
        finish_reason: str | None = None,
        error_message: str | None = None,
        pending_call_id: str | None = None,
        context_snapshot_id: str | None = None,
        history_through_seq: int = 0,
        prompt_hash: str | None = None,
    ) -> None:
        """Insert or update frame in single operation."""
        await self._conn.execute(
            """
            INSERT INTO frames (id, session_id, turn_id, trace_id, iteration, status, started_at,
                               finished_at, duration_ms, input_tokens, output_tokens, model_name, finish_reason,
                               error_message, pending_call_id, context_snapshot_id, history_through_seq,
                               prompt_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                finished_at = COALESCE(excluded.finished_at, frames.finished_at),
                duration_ms = COALESCE(excluded.duration_ms, frames.duration_ms),
                input_tokens = CASE WHEN ? IS NULL THEN frames.input_tokens ELSE excluded.input_tokens END,
                output_tokens = CASE WHEN ? IS NULL THEN frames.output_tokens ELSE excluded.output_tokens END,
                model_name = COALESCE(excluded.model_name, frames.model_name),
                finish_reason = COALESCE(excluded.finish_reason, frames.finish_reason),
                error_message = COALESCE(excluded.error_message, frames.error_message),
                pending_call_id = COALESCE(excluded.pending_call_id, frames.pending_call_id),
                context_snapshot_id = COALESCE(excluded.context_snapshot_id, frames.context_snapshot_id),
                history_through_seq = CASE
                    WHEN excluded.history_through_seq = 0 THEN frames.history_through_seq
                    ELSE excluded.history_through_seq
                END,
                prompt_hash = COALESCE(excluded.prompt_hash, frames.prompt_hash)
            """,
            (
                frame_id,
                session_id,
                turn_id,
                trace_id,
                iteration,
                status,
                started_at_ms,
                finished_at_ms,
                duration_ms,
                input_tokens,
                output_tokens,
                model_name,
                finish_reason,
                error_message,
                pending_call_id,
                context_snapshot_id,
                history_through_seq,
                prompt_hash,
                input_tokens,
                output_tokens,
            ),
        )
