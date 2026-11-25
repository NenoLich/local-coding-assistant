"""Pydantic models for the live (in-memory) monitoring plane.

These models describe the in-memory state reconstructed from runtime events
for the runs page. They are not persisted to the execution store.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from local_coding_assistant.runtime.events.contracts import EventType


class LiveEvent(BaseModel):
    """A single runtime event in the live view."""

    seq: int
    type: EventType
    timestamp: datetime
    frame_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class LiveFrame(BaseModel):
    """A live frame (iteration within a turn)."""

    frame_id: str
    iteration: int
    status: str  # running_llm | running_tools | paused_approval | completed | failed
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    model_name: str | None = None
    finish_reason: str | None = None
    error_message: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    tool_call_count: int = 0
    pending_call_id: str | None = None
    events: list[LiveEvent] = Field(default_factory=list)

    class Config:
        from_attributes = True


class LiveTurn(BaseModel):
    """A live turn within a session."""

    turn_id: str
    turn_index: int
    trace_id: str
    user_input: str
    status: str  # running | completed | error | paused
    started_at: datetime
    finished_at: datetime | None = None
    pause_reason: str | None = None  # approval | interrupt | deadline | denied
    final_answer: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    total_frames: int = 0
    frames: dict[str, LiveFrame] = Field(default_factory=dict)

    class Config:
        from_attributes = True


class LiveSession(BaseModel):
    """A live session being monitored."""

    session_id: str
    status: str  # idle | running | interrupted | completed | error
    started_at: datetime
    updated_at: datetime
    user_input: str
    agent_mode: str
    model_name: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    turns: dict[str, LiveTurn] = Field(default_factory=dict)

    class Config:
        from_attributes = True


class LiveRunSummary(BaseModel):
    """Summary of a live run (turn) for the runs list."""

    run_id: str
    session_id: str
    turn_index: int
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    duration: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    total_frames: int = 0
    current_frame_id: str | None = None
    current_frame_status: str | None = None


class LiveStats(BaseModel):
    """Live aggregate statistics."""

    active_sessions: int
    running_turns: int
    total_events: int
    events_per_second: float = 0.0
    error_count_last_minute: int = 0
