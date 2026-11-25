"""Historical session service backed by the read-only execution store."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from local_coding_assistant.dashboard.exceptions import (
    NotFoundError,
    StoreUnavailableError,
)
from local_coding_assistant.dashboard.models.historical import (
    ContextBlock,
    SessionDetail,
    SessionSummary,
    TurnSummary,
)
from local_coding_assistant.dashboard.services.snapshot_context_resolution import (
    render_snapshot_context,
)
from local_coding_assistant.runtime.context.profile import RuntimeContextConfig
from local_coding_assistant.storage.execution_store_reader import ExecutionStoreReader
from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.services.session_service")


def _ms_to_iso(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, UTC)


class SessionService:
    """Read-only historical session queries."""

    def __init__(self, store: ExecutionStoreReader):
        self._store = store

    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """List historical sessions with pagination and filtering."""
        if self._store is None:
            raise StoreUnavailableError("Execution store not available")

        sessions = await self._store.list_sessions(limit=10000, status=status)

        # Apply date and model filters in Python (store only supports limit/status)
        date_from_dt: datetime | None = None
        date_to_dt: datetime | None = None
        if date_from:
            date_from_dt = datetime.fromisoformat(date_from).replace(tzinfo=UTC)
        if date_to:
            date_to_dt = datetime.fromisoformat(date_to).replace(tzinfo=UTC)

        filtered = []
        for s in sessions:
            if date_from_dt and s.started_at_ms:
                started = datetime.fromtimestamp(s.started_at_ms / 1000, UTC)
                if started < date_from_dt:
                    continue
            if date_to_dt and s.started_at_ms:
                started = datetime.fromtimestamp(s.started_at_ms / 1000, UTC)
                if started > date_to_dt:
                    continue
            if model and s.model_name and model.lower() not in s.model_name.lower():
                continue
            filtered.append(s)

        total = len(filtered)
        paginated = filtered[offset : offset + limit]

        items = [
            SessionSummary(
                session_id=s.id,
                user_input=s.user_input,
                status=s.status,
                started_at=_ms_to_iso(s.started_at_ms) or datetime.now(UTC),
                updated_at=_ms_to_iso(s.updated_at_ms),
                agent_mode=s.agent_mode,
                model_name=s.model_name,
                input_tokens=int(s.total_input_tokens or 0),
                output_tokens=int(s.total_output_tokens or 0),
                total_turns=int(s.total_turns or 0),
                total_frames=int(s.total_frames or 0),
            )
            for s in paginated
        ]

        return {
            "items": items,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_next": offset + limit < total,
            "has_prev": offset > 0,
        }

    async def get_session_detail(self, session_id: str) -> SessionDetail:
        """Get full session detail with turns."""
        if self._store is None:
            raise StoreUnavailableError("Execution store not available")

        session = await self._store.get_session(session_id)
        if session is None:
            raise NotFoundError(f"Session not found: {session_id}")

        turns = await self._store.list_turns_by_session(session_id)
        turn_summaries = [
            TurnSummary(
                turn_id=t.id,
                session_id=t.session_id,
                turn_index=t.turn_index,
                trace_id=t.trace_id,
                user_input=t.user_input,
                status=t.status,
                started_at=_ms_to_iso(t.started_at_ms) or datetime.now(UTC),
                finished_at=_ms_to_iso(t.finished_at_ms),
                duration_ms=(t.finished_at_ms - t.started_at_ms)
                if t.finished_at_ms and t.started_at_ms
                else None,
                input_tokens=t.total_input_tokens or 0,
                output_tokens=t.total_output_tokens or 0,
                total_frames=t.total_frames or 0,
                final_answer=t.final_answer,
                pause_reason=t.pause_reason,
            )
            for t in turns
        ]

        return SessionDetail(
            session=SessionSummary(
                session_id=session.id,
                user_input=session.user_input,
                status=session.status,
                started_at=_ms_to_iso(session.started_at_ms) or datetime.now(UTC),
                updated_at=_ms_to_iso(session.updated_at_ms),
                agent_mode=session.agent_mode,
                model_name=session.model_name,
                input_tokens=session.total_input_tokens,
                output_tokens=session.total_output_tokens,
                total_turns=session.total_turns,
                total_frames=session.total_frames,
            ),
            turns=turn_summaries,
            context=await self._build_context_block(
                turns, session.id, session.updated_at_ms
            ),
        )

    async def _build_context_block(
        self, turns: list[Any], session_id: str, last_updated_at: int | None
    ) -> ContextBlock | None:
        """Build the context block from the last turn's stored context snapshot.

        System messages are resampled from the last turn's context_snapshot_id;
        config (RuntimeContextConfig) comes from that turn's config_snapshot_json.
        Missing refs degrade gracefully with warnings instead of raising.
        """
        if not turns:
            return None

        last_turn = max(turns, key=lambda t: t.turn_index)
        snapshot_id = last_turn.context_snapshot_id
        config_json = last_turn.config_snapshot_json
        if not snapshot_id or not config_json:
            return None

        snapshot = await self._store.get_context_snapshot(snapshot_id)
        if snapshot is None:
            return None

        try:
            config_data = json.loads(config_json)
            config_snapshot = RuntimeContextConfig(
                **config_data.get("context_config", {})
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "Failed to parse config_snapshot_json for session %s: %s",
                session_id,
                exc,
            )
            return ContextBlock(
                warnings=[f"config: {exc}"],
                resolved=False,
                context_snapshot_id=snapshot_id,
            )
        result = await render_snapshot_context(
            self._store, snapshot, config_snapshot, session_id, last_updated_at
        )
        return ContextBlock(
            system_messages=result.system_messages,
            tool_schemas=result.tool_schemas,
            system_reminder=result.system_reminder,
            warnings=result.warnings,
            resolved=result.resolved,
            context_snapshot_id=snapshot_id,
        )
