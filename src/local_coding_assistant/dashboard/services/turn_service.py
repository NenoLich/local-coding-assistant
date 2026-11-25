"""Historical turn service backed by the read-only execution store."""

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
    FrameSummary,
    MessageEntry,
    TurnDetail,
    TurnSummary,
    compute_frame_token_totals,
)
from local_coding_assistant.dashboard.services.snapshot_context_resolution import (
    render_snapshot_context,
)
from local_coding_assistant.runtime.context.profile import RuntimeContextConfig
from local_coding_assistant.storage.execution_store_reader import ExecutionStoreReader
from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.services.turn_service")


def _ms_to_iso(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, UTC)


def _parse_metadata(metadata_json: str | None) -> dict[str, Any]:
    if not metadata_json:
        return {}
    import json

    try:
        return json.loads(metadata_json)
    except Exception:
        return {}


class TurnService:
    """Read-only historical turn queries."""

    def __init__(self, store: ExecutionStoreReader):
        self._store = store

    async def get_turn_detail(self, turn_id: str) -> TurnDetail:
        """Get full turn detail with frames and messages."""
        if self._store is None:
            raise StoreUnavailableError("Execution store not available")

        turn = await self._store.get_turn(turn_id)
        if turn is None:
            raise NotFoundError(f"Turn not found: {turn_id}")

        frame_ids = await self._store.get_frame_ids_by_turn(turn_id)
        frames = []
        for frame_id in frame_ids:
            frame = await self._store.get_frame(frame_id)
            if not frame:
                continue
            events_raw = await self._store.get_frame_events(frame_id)
            totals = compute_frame_token_totals(events_raw)
            frames.append(
                FrameSummary(
                    frame_id=frame.id,
                    turn_id=frame.turn_id,
                    iteration=frame.iteration,
                    status=frame.status,
                    started_at=_ms_to_iso(frame.started_at_ms) or datetime.now(UTC),
                    finished_at=_ms_to_iso(frame.finished_at_ms),
                    duration_ms=frame.duration_ms,
                    model_name=frame.model_name,
                    finish_reason=frame.finish_reason,
                    error_message=frame.error_message,
                    input_tokens=totals["input_tokens"],
                    output_tokens=totals["output_tokens"],
                    cache_tokens=totals["cache_tokens"],
                )
            )

        frames.sort(key=lambda f: f.iteration)

        messages = await self._store.get_all_messages(turn.session_id)
        turn_messages = [
            MessageEntry(
                seq=m["seq"],
                role=m["role"],
                content=m["content"] or "",
                tool_call_id=m.get("tool_call_id"),
                tool_name=m.get("tool_name"),
                turn_id=m["turn_id"],
                frame_id=m.get("frame_id"),
                created_at=_ms_to_iso(int(m["created_at"]))
                if m.get("created_at")
                else datetime.now(UTC),
                metadata=_parse_metadata(m.get("metadata_json")),
            )
            for m in messages
            if m.get("turn_id") == turn_id
        ]

        turn_summary = None
        # We need to build a TurnSummary; get_turn returns StoredTurn.
        # Token totals are summed from frames (cumulative from LLM_COMPLETE
        # events persisted by the store adapter).
        turn_summary = TurnSummary(
            turn_id=turn.id,
            session_id=turn.session_id,
            turn_index=turn.turn_index,
            trace_id=turn.trace_id,
            user_input=turn.user_input,
            status=turn.status,
            started_at=_ms_to_iso(turn.started_at_ms) or datetime.now(UTC),
            finished_at=_ms_to_iso(turn.finished_at_ms),
            duration_ms=(turn.finished_at_ms - turn.started_at_ms)
            if turn.finished_at_ms and turn.started_at_ms
            else None,
            input_tokens=sum(f.input_tokens for f in frames),
            output_tokens=sum(f.output_tokens for f in frames),
            cache_tokens=sum(f.cache_tokens for f in frames),
            total_frames=turn.total_frames or len(frames),
            final_answer=turn.final_answer,
            pause_reason=turn.pause_reason,
        )

        return TurnDetail(
            turn=turn_summary,
            frames=frames,
            messages=turn_messages,
            context=await self._build_context_block(turn),
        )

    async def _build_context_block(self, turn: Any) -> ContextBlock | None:
        """Build the context block from this turn's stored context snapshot.

        System messages are resampled from the turn's context_snapshot_id;
        config (RuntimeContextConfig) comes from config_snapshot_json.
        Missing refs degrade gracefully with warnings instead of raising.
        """
        snapshot_id = turn.context_snapshot_id
        config_json = turn.config_snapshot_json
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
                "Failed to parse config_snapshot_json for turn %s: %s",
                turn.id,
                exc,
            )
            return ContextBlock(
                warnings=[f"config: {exc}"],
                resolved=False,
                context_snapshot_id=snapshot_id,
            )

        result = await render_snapshot_context(
            self._store, snapshot, config_snapshot, turn.session_id, turn.finished_at_ms
        )
        return ContextBlock(
            system_messages=result.system_messages,
            tool_schemas=result.tool_schemas,
            system_reminder=result.system_reminder,
            warnings=result.warnings,
            resolved=result.resolved,
            context_snapshot_id=snapshot_id,
        )
