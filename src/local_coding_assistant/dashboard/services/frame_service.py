"""Historical frame service backed by the read-only execution store."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from local_coding_assistant.dashboard.exceptions import (
    NotFoundError,
    StoreUnavailableError,
)
from local_coding_assistant.dashboard.models.historical import (
    EventTimelineEntry,
    FrameDetail,
    FrameSummary,
    compute_frame_token_totals,
)
from local_coding_assistant.storage.execution_store_reader import ExecutionStoreReader
from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.services.frame_service")


def _ms_to_iso(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, UTC)


_PREVIEW_KEYS: dict[str, tuple[str, ...]] = {
    "llm_complete": (
        "model",
        "provider",
        "finish_reason",
        "input_tokens",
        "output_tokens",
        "tool_calls",
    ),
    "tool_result": (
        "tool_name",
        "success",
        "execution_time_ms",
        "error",
    ),
    "turn_complete": (
        "status",
        "final_answer",
        "total_input_tokens",
        "total_output_tokens",
        "total_frames",
        "finish_reason",
    ),
    "frame_complete": (
        "status",
        "model_name",
        "finish_reason",
        "duration_ms",
        "error_message",
    ),
    "turn_start": ("user_input", "turn_index", "agent_mode", "model_name"),
    "frame_start": ("iteration", "model_name"),
    "session_start": ("user_input", "agent_mode", "model_name"),
    "error": ("error_type", "error_message", "recoverable"),
    "continuation_decision": ("decision", "action", "stop_reason", "reason"),
}


def _preview_for_event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """Extract important preview fields for the timeline."""
    preview: dict[str, Any] = {}

    for key in _PREVIEW_KEYS.get(event_type, ()):
        if key in data:
            preview[key] = data[key]

    if event_type == "llm_complete":
        content = data.get("content")
        if content:
            preview["content"] = str(content)[:200]
    elif event_type == "approval_required":
        preview["approval_requests"] = len(data.get("approval_requests", []))
    elif event_type == "approval_granted":
        preview["approval_token"] = data.get("approval_token")
        preview["tool_call_id"] = data.get("tool_call_id")
    elif event_type == "approval_denied":
        preview["approval_token"] = data.get("approval_token")
        preview["tool_call_id"] = data.get("tool_call_id")
        preview["reason"] = data.get("reason")

    return preview


class FrameService:
    """Read-only historical frame queries."""

    def __init__(self, store: ExecutionStoreReader):
        self._store = store

    async def get_frame_detail(self, frame_id: str) -> FrameDetail:
        """Get full frame detail with event timeline."""
        if self._store is None:
            raise StoreUnavailableError("Execution store not available")

        frame = await self._store.get_frame(frame_id)
        if frame is None:
            raise NotFoundError(f"Frame not found: {frame_id}")

        events_raw = await self._store.get_frame_events(frame_id)
        events = []
        for ev in events_raw:
            import json

            try:
                data = json.loads(ev.get("data_json") or "{}")
            except Exception:
                data = {}
            events.append(
                EventTimelineEntry(
                    event_id=int(ev["id"]),
                    seq=int(ev.get("seq", 0)),
                    type=ev.get("type", "unknown"),
                    timestamp=_ms_to_iso(int(ev.get("timestamp", 0)))
                    if ev.get("timestamp")
                    else datetime.now(UTC),
                    frame_id=ev.get("frame_id"),
                    preview=_preview_for_event(ev.get("type", ""), data),
                    has_payload=bool(data),
                )
            )

        events.sort(key=lambda e: e.seq)

        totals = compute_frame_token_totals(events_raw)
        frame_summary = FrameSummary(
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

        return FrameDetail(frame=frame_summary, events=events)

    async def list_frames_by_turn(self, turn_id: str) -> list[FrameSummary]:
        """List all frames for a turn."""
        if self._store is None:
            raise StoreUnavailableError("Execution store not available")

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
        return frames
