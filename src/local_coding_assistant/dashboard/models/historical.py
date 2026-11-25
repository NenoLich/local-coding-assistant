"""Pydantic models for the historical (store-backed) plane.

These are read-only projections of durable execution store records for
session, turn, frame, event, and message detail pages.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def _ms_to_iso(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, UTC)


class SessionSummary(BaseModel):
    """Summary of a historical session for the sessions list."""

    session_id: str
    user_input: str
    status: str
    started_at: datetime
    updated_at: datetime | None = None
    agent_mode: str
    model_name: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_turns: int = 0
    total_frames: int = 0


class TurnSummary(BaseModel):
    """Summary of a historical turn."""

    turn_id: str
    session_id: str
    turn_index: int
    trace_id: str
    user_input: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    total_frames: int = 0
    final_answer: str | None = None
    pause_reason: str | None = None


class FrameSummary(BaseModel):
    """Summary of a historical frame."""

    frame_id: str
    turn_id: str
    iteration: int
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    model_name: str | None = None
    finish_reason: str | None = None
    error_message: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0


class EventTimelineEntry(BaseModel):
    """A single event in the timeline preview (important fields only)."""

    event_id: int
    seq: int
    type: str
    timestamp: datetime | None = None
    frame_id: str | None = None
    # Preview fields extracted from the event payload for quick scanning
    preview: dict[str, Any] = Field(default_factory=dict)
    # Full payload available on demand via expand
    payload: dict[str, Any] = Field(default_factory=dict)
    has_payload: bool = False


class MessageEntry(BaseModel):
    """A transcript message, clickable from detail pages."""

    seq: int
    role: str  # user | assistant | tool | system
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    turn_id: str
    frame_id: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBlock(BaseModel):
    """Rendered system message + tool spec blocks assembled from a stored
    context snapshot. Separate from transcript messages; always rendered at
    the top of session/turn detail pages and controlled by its own toggle
    checkboxes rather than message filters.
    """

    system_messages: list[str] = Field(
        default_factory=list, description="Rendered system message(s) in effect"
    )
    tool_schemas: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tool schemas (OpenAI function form) in effect",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal resolution warnings (partial context)",
    )
    resolved: bool = Field(
        default=True, description="Whether all context refs resolved successfully"
    )
    system_reminder: str | None = Field(
        default=None,
        description="Rendered system reminder (developer hint) to show as the last red transcript message",
    )
    context_snapshot_id: str | None = Field(
        default=None, description="Source context snapshot id"
    )


class SessionDetail(BaseModel):
    """Full historical session detail with turns."""

    session: SessionSummary
    turns: list[TurnSummary]
    context: ContextBlock | None = Field(
        default=None,
        description="System message + tool spec blocks from the last turn's context snapshot",
    )


class TurnDetail(BaseModel):
    """Full historical turn detail with frames and messages."""

    turn: TurnSummary
    frames: list[FrameSummary]
    messages: list[MessageEntry]
    context: ContextBlock | None = Field(
        default=None,
        description="System message + tool spec blocks from this turn's context snapshot",
    )


class FrameDetail(BaseModel):
    """Full historical frame detail with events."""

    frame: FrameSummary
    events: list[EventTimelineEntry]


def compute_frame_token_totals(events: list[dict[str, Any]]) -> dict[str, int]:
    """Sum input_tokens, output_tokens, cache_tokens from LLM_COMPLETE events.

    Accepts the stored frame and its raw event rows (each with ``type`` and
    ``data_json``). Only ``llm_complete`` events contribute; their
    ``data_json`` payload carries ``input_tokens``, ``output_tokens``, and
    ``cached_tokens``.
    """
    input_tokens = 0
    output_tokens = 0
    cache_tokens = 0
    for ev in events:
        if ev.get("type") != "llm_complete":
            continue
        data = ev.get("data_json") or {}
        if isinstance(data, str):
            import json

            try:
                data = json.loads(data)
            except Exception:
                data = {}
        input_tokens += int(data.get("input_tokens") or 0)
        output_tokens += int(data.get("output_tokens") or 0)
        cache_tokens += int(data.get("cached_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_tokens": cache_tokens,
    }
