"""Session repository for execution store."""

from __future__ import annotations

import aiosqlite

from local_coding_assistant.storage.contracts import StoredSession
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("storage.repositories.session_repo")


class SessionRepo:
    """Repository for session operations."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def create(
        self,
        session_id: str,
        user_input: str,
        status: str,
        started_at_ms: int,
        agent_mode: str,
        model_name: str | None,
        metadata_json: str | None = None,
    ) -> None:
        """Create a new session."""
        await self._conn.execute(
            """
            INSERT INTO sessions (id, user_input, status, started_at, agent_mode, model_name, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_input,
                status,
                started_at_ms,
                agent_mode,
                model_name,
                metadata_json,
            ),
        )

    async def get(self, session_id: str) -> StoredSession | None:
        """Get session by ID."""
        cursor = await self._conn.execute(
            """
            SELECT id, user_input, status, started_at, updated_at,
                   agent_mode, model_name, total_input_tokens, total_output_tokens, total_turns, total_frames, metadata_json, context_profile_json
            FROM sessions WHERE id = ?
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        return StoredSession(
            id=row[0],
            user_input=row[1],
            status=row[2],
            started_at_ms=row[3],
            updated_at_ms=row[4],
            agent_mode=row[5],
            model_name=row[6],
            total_input_tokens=row[7] or 0,
            total_output_tokens=row[8] or 0,
            total_turns=row[9] or 0,
            total_frames=row[10] or 0,
            metadata_json=row[11],
            context_profile_json=row[12],
        )

    async def update_status(
        self, session_id: str, status: str, updated_at_ms: int | None = None
    ) -> None:
        """Update session status."""
        if updated_at_ms:
            await self._conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
                (status, updated_at_ms, session_id),
            )
        else:
            await self._conn.execute(
                "UPDATE sessions SET status = ? WHERE id = ?",
                (status, session_id),
            )

    async def update_totals(
        self,
        session_id: str,
        total_input_tokens: int,
        total_output_tokens: int,
        total_turns: int,
        total_frames: int,
    ) -> None:
        """Update session totals."""
        await self._conn.execute(
            "UPDATE sessions SET total_input_tokens = ?, total_output_tokens = ?, total_turns = ?, total_frames = ? WHERE id = ?",
            (
                total_input_tokens,
                total_output_tokens,
                total_turns,
                total_frames,
                session_id,
            ),
        )

    async def increment_totals(
        self,
        session_id: str,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        total_turns: int = 0,
        total_frames: int = 0,
    ) -> None:
        """Atomically add usage deltas to session totals."""
        await self._conn.execute(
            """
            UPDATE sessions
            SET total_input_tokens = COALESCE(total_input_tokens, 0) + ?,
                total_output_tokens = COALESCE(total_output_tokens, 0) + ?,
                total_turns = COALESCE(total_turns, 0) + ?,
                total_frames = COALESCE(total_frames, 0) + ?
            WHERE id = ?
            """,
            (
                total_input_tokens,
                total_output_tokens,
                total_turns,
                total_frames,
                session_id,
            ),
        )

    async def list_recent(self, limit: int = 100) -> list[StoredSession]:
        """List recent sessions."""
        cursor = await self._conn.execute(
            """
            SELECT id, user_input, status, started_at, updated_at,
                   agent_mode, model_name, total_input_tokens, total_output_tokens, total_turns, total_frames
            FROM sessions ORDER BY started_at DESC LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            StoredSession(
                id=row[0],
                user_input=row[1],
                status=row[2],
                started_at_ms=row[3],
                updated_at_ms=row[4],
                agent_mode=row[5],
                model_name=row[6],
                total_input_tokens=row[7] or 0,
                total_output_tokens=row[8] or 0,
                total_turns=row[9] or 0,
                total_frames=row[10] or 0,
            )
            for row in rows
        ]

    async def upsert(
        self,
        session_id: str,
        status: str,
        user_input: str = "",
        started_at_ms: int = 0,
        agent_mode: str = "",
        model_name: str | None = None,
        metadata_json: str | None = None,
        updated_at_ms: int | None = None,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        total_turns: int | None = None,
        total_frames: int | None = None,
        context_profile_json: str | None = None,
    ) -> None:
        """Insert or update session in single operation.

        Partial-update semantics: every column is only overwritten when the
        caller actually supplies it. NULL (metadata_json, context_profile_json,
        model_name, updated_at_ms, total_*) fall back to the stored value, so
        status-only updates cannot nullify the session profile or metadata.
        For the NOT NULL identity columns (user_input, started_at, agent_mode)
        the sentinels "" / 0 mean "not updated" on conflict.
        """
        await self._conn.execute(
            """
            INSERT INTO sessions (id, user_input, status, started_at, updated_at,
                                   agent_mode, model_name, total_input_tokens, total_output_tokens, total_turns, total_frames, metadata_json, context_profile_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, 0), COALESCE(?, 0), COALESCE(?, 0), COALESCE(?, 0), ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                updated_at = COALESCE(excluded.updated_at, sessions.updated_at),
                user_input = CASE WHEN excluded.user_input = '' THEN sessions.user_input ELSE excluded.user_input END,
                agent_mode = CASE WHEN excluded.agent_mode = '' THEN sessions.agent_mode ELSE excluded.agent_mode END,
                started_at = CASE WHEN excluded.started_at = 0 THEN sessions.started_at ELSE excluded.started_at END,
                model_name = COALESCE(excluded.model_name, sessions.model_name),
                total_input_tokens = CASE WHEN ? IS NULL THEN sessions.total_input_tokens ELSE excluded.total_input_tokens END,
                total_output_tokens = CASE WHEN ? IS NULL THEN sessions.total_output_tokens ELSE excluded.total_output_tokens END,
                total_turns = CASE WHEN ? IS NULL THEN sessions.total_turns ELSE excluded.total_turns END,
                total_frames = CASE WHEN ? IS NULL THEN sessions.total_frames ELSE excluded.total_frames END,
                metadata_json = COALESCE(excluded.metadata_json, sessions.metadata_json),
                context_profile_json = COALESCE(excluded.context_profile_json, sessions.context_profile_json)
            """,
            (
                session_id,
                user_input,
                status,
                started_at_ms,
                updated_at_ms,
                agent_mode,
                model_name,
                total_input_tokens,
                total_output_tokens,
                total_turns,
                total_frames,
                metadata_json,
                context_profile_json,
                total_input_tokens,
                total_output_tokens,
                total_turns,
                total_frames,
            ),
        )

    async def delete(self, session_id: str) -> None:
        """Delete a session (cascades to turns, frames, messages, events)."""
        await self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
