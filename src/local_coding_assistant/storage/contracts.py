"""Storage contracts for durable execution records.

These are the STORED, durable models that map directly to database tables.
They are lightweight, reference-based, and exclude runtime-only fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from local_coding_assistant.runtime.context.profile import ContextSnapshot


class StoredSession(BaseModel):
    """Durable session record for storage.

    Session status can only be 'idle' or 'running'.
    - 'running': a turn is currently in progress (status='running' in store)
    - 'idle': no turn in progress, all started turns are completed
    """

    id: str = Field(..., description="Unique session identifier")
    user_input: str = Field(
        default="", description="Initial user input for this session"
    )
    agent_mode: str = Field(default="frame_agent", description="Agent mode identifier")
    status: str = Field(
        default="idle",
        description="Session status: 'idle' or 'running'",
        pattern="^(idle|running)$",
    )
    started_at_ms: int = Field(
        ..., description="Session start time in UTC epoch milliseconds"
    )
    updated_at_ms: int | None = Field(
        default=None, description="Session last update time in UTC epoch milliseconds"
    )
    model_name: str | None = Field(default=None, description="Model name used")
    total_input_tokens: int = Field(
        default=0, description="Total input tokens used in this session"
    )
    total_output_tokens: int = Field(
        default=0, description="Total output tokens used in this session"
    )
    total_turns: int = Field(default=0, description="Total turns in this session")
    total_frames: int = Field(default=0, description="Total frames in this session")
    # Context profile baseline for this session (serialized ContextProfile)
    context_profile_json: str | None = Field(
        default=None, description="ContextProfile as JSON for reconstruction"
    )
    metadata_json: str | None = Field(
        default=None, description="Session metadata as JSON"
    )


class StoredTurn(BaseModel):
    """Durable turn record for storage."""

    id: str = Field(..., description="Unique turn identifier")
    session_id: str = Field(..., description="Parent session identifier")
    trace_id: str = Field(..., description="Trace correlation ID for this turn")
    turn_index: int = Field(..., description="Sequential turn index within session")
    user_input: str = Field(..., description="User input for this turn")
    config_snapshot_json: str | None = Field(
        ...,
        description="Optional config snapshot JSON (contains context_config and limits_config)",
    )
    status: str = Field(default="running", description="Turn status")
    started_at_ms: int = Field(
        ..., description="Turn start time in UTC epoch milliseconds"
    )
    finished_at_ms: int | None = Field(
        default=None, description="Turn end time in UTC epoch milliseconds"
    )
    paused_at_ms: int | None = Field(
        default=None, description="Turn pause time in UTC epoch milliseconds"
    )
    pause_reason: str | None = Field(
        default=None, description="Reason for pause: approval|interrupt|deadline|denied"
    )
    resumed_at_ms: int | None = Field(
        default=None, description="Turn resume time in UTC epoch milliseconds"
    )
    context_snapshot_id: str | None = Field(
        default=None, description="Reference to immutable context snapshot"
    )
    total_input_tokens: int = Field(
        default=0, description="Total input tokens used in this turn"
    )
    total_output_tokens: int = Field(
        default=0, description="Total output tokens used in this turn"
    )
    total_frames: int = Field(
        default=0, description="Total frames executed in this turn"
    )
    # Turn-level message boundary (unlike snapshot.message_boundary, which is
    # immutable and set at turn start). Optional: transcript messages carry
    # their turn_id, so history reconstruction and branching filter by turn.
    message_boundary_json: str | None = Field(
        default=None,
        description="MessageBoundary as JSON: {session_id, start_seq, end_seq}",
    )
    final_answer: str | None = Field(
        default=None, description="Final answer text produced by the turn"
    )
    metadata_json: str | None = Field(
        default=None, description="Additional metadata as JSON"
    )


class StoredTrace(BaseModel):
    """Durable trace correlation record for storage."""

    trace_id: str = Field(..., description="Unique trace identifier")
    session_id: str = Field(..., description="Parent session identifier")
    turn_id: str = Field(..., description="Owning turn identifier")


class StoredFrame(BaseModel):
    """Durable frame record for storage. Lightweight, no prompt blob."""

    id: str = Field(..., description="Unique frame identifier")
    session_id: str = Field(..., description="Parent session identifier")
    turn_id: str = Field(..., description="Parent turn identifier")
    trace_id: str = Field(..., description="Trace correlation ID")
    iteration: int = Field(..., description="Frame iteration within turn")
    status: str = Field(default="running_llm", description="Frame execution status")
    context_snapshot_id: str | None = Field(
        default=None, description="Reference to context snapshot"
    )
    started_at_ms: int = Field(
        ..., description="Frame start time in UTC epoch milliseconds"
    )
    finished_at_ms: int | None = Field(
        default=None, description="Frame end time in UTC epoch milliseconds"
    )
    duration_ms: int | None = Field(
        default=None, description="Frame duration in milliseconds"
    )
    input_tokens: int = Field(default=0, description="Input tokens used in this frame")
    output_tokens: int = Field(
        default=0, description="Output tokens used in this frame"
    )
    model_name: str | None = Field(
        default=None, description="Model used for this frame"
    )
    finish_reason: str | None = Field(
        default=None, description="Finish reason from LLM"
    )
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )
    pending_call_id: str | None = Field(
        default=None, description="Pending approval call ID"
    )
    history_through_seq: int = Field(
        default=0,
        description="Last transcript sequence used to build this frame's prompt",
    )
    prompt_hash: str | None = Field(
        default=None, description="Hash of the fully rendered prompt used by this frame"
    )


def _normalize_json_column(value: str | None) -> str | None:
    """Normalize a stored JSON ref column to None when effectively absent.

    Older writes stored the literal string ``"null"`` for absent refs, which
    then fails ``model_validate_json`` on read; treat those (and SQL NULL) as
    absent so reconstruction stays tolerant.
    """
    if value is None or value in ("null", '""', ""):
        return None
    return value


class StoredContextSnapshot(BaseModel):
    """Durable context snapshot record for storage."""

    id: str = Field(..., description="Unique context snapshot identifier")
    turn_id: str = Field(..., description="Owning turn identifier")
    schema_version: int = Field(
        default=1, description="Schema version for compatibility"
    )
    tool_set_ref: str | None = Field(
        default=None, description="Reference to tool set as JSON"
    )
    template_set_ref: str | None = Field(
        default=None, description="Reference to template set as JSON"
    )
    repo_snapshot_ref: str | None = Field(
        default=None, description="Reference to repository snapshot as JSON"
    )
    memory_snapshot_ref: str | None = Field(
        default=None, description="Reference to memory snapshot as JSON"
    )
    message_boundary_json: str | None = Field(
        default=None, description="Message boundary references as JSON"
    )
    content_hash: str = Field(..., description="SHA256 hash of snapshot content")
    payload_json: str | None = Field(
        default=None, description="Optional compact structured payload"
    )
    created_at_ms: int = Field(
        ..., description="Creation time in UTC epoch milliseconds"
    )

    def to_runtime(self) -> ContextSnapshot:
        """Convert StoredContextSnapshot to runtime ContextSnapshot."""
        from local_coding_assistant.runtime.context.profile import (
            ContextSnapshot,
            MemorySnapshotRef,
            MessageBoundary,
            RepoSnapshotRef,
            TemplateSetRef,
            ToolSetRef,
        )

        tool_set_json = _normalize_json_column(self.tool_set_ref)
        template_set_json = _normalize_json_column(self.template_set_ref)
        repo_snapshot_json = _normalize_json_column(self.repo_snapshot_ref)
        memory_snapshot_json = _normalize_json_column(self.memory_snapshot_ref)
        message_boundary_json = _normalize_json_column(self.message_boundary_json)

        return ContextSnapshot(
            id=self.id,
            turn_id=self.turn_id,
            schema_version=self.schema_version,
            tool_set_ref=ToolSetRef.model_validate_json(tool_set_json)
            if tool_set_json
            else ToolSetRef(),
            template_set_ref=TemplateSetRef.model_validate_json(template_set_json)
            if template_set_json
            else TemplateSetRef(),
            repo_snapshot_ref=RepoSnapshotRef.model_validate_json(repo_snapshot_json)
            if repo_snapshot_json
            else None,
            memory_snapshot_ref=MemorySnapshotRef.model_validate_json(
                memory_snapshot_json
            )
            if memory_snapshot_json
            else None,
            message_boundary=MessageBoundary.model_validate_json(message_boundary_json)
            if message_boundary_json
            else None,
            content_hash=self.content_hash,
            payload_json=self.payload_json,
            created_at_ms=self.created_at_ms,
        )


class StoredEvent(BaseModel):
    """Compact, durable event record for persistence.

    This is what gets written to the database. It intentionally excludes
    raw prompt text, full repo map dumps, and large context payloads.
    """

    trace_id: str
    session_id: str
    turn_id: str | None = None
    frame_id: str | None = None
    seq: int
    type: str
    payload_json: str
    timestamp_ms: int


# === Resume loading ===


@dataclass(frozen=True, slots=True)
class MissingRefs:
    """Context snapshot references that no longer resolve to stored rows.

    Produced when validating that a snapshot pinned for a turn still
    describes artifacts that exist in the store.
    """

    tool_ids: list[int] = field(default_factory=list)
    template_ids: list[int] = field(default_factory=list)
    repo_hashes: list[str] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (
            self.tool_ids or self.template_ids or self.repo_hashes or self.memory_ids
        )


@dataclass(slots=True)
class ResumeBatch:
    """Everything the resume flow needs, loaded in a constant number of queries.

    Mirrors the live resume path: the session, its turns (for selection and
    branch decisions), the selected turn's snapshot, that turn's lightweight
    frame rows, and the events of the single frame being resumed. The full
    session graph (every turn's frames and events) is intentionally NOT
    loaded here - that shape belongs to the future replay flow.
    """

    session: StoredSession
    turn: StoredTurn
    snapshot: ContextSnapshot | None
    turns: list[StoredTurn] = field(default_factory=list)
    frames: list[StoredFrame] = field(default_factory=list)
    frame_events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    missing_refs: MissingRefs = field(default_factory=MissingRefs)


@dataclass(slots=True)
class TurnReplay:
    """One turn's durable graph: snapshot, frames, messages and events."""

    turn: StoredTurn
    snapshot: ContextSnapshot | None
    frames: list[StoredFrame] = field(default_factory=list)
    frame_events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    missing_refs: MissingRefs = field(default_factory=MissingRefs)


@dataclass(slots=True)
class SessionReplay:
    """Full durable session graph for replay/debug/analytics (read-only)."""

    session: StoredSession
    turns: list[TurnReplay] = field(default_factory=list)
