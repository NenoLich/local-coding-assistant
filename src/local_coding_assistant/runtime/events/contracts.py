"""Typed event contracts for runtime execution events.

These are the RICH, typed event contracts used at runtime for streaming,
CLI/UI display, and debugging. They may contain debug payloads needed by
consumers but are NOT the compact form stored in the database.

The stored events (WriteOp/StoredEvent) are compact, reference-based records
written by the StoreSubscriber after deliberate refinement.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from local_coding_assistant.storage.contracts import StoredEvent


class EventType(str, Enum):
    """Types of events emitted during execution."""

    # Session/Turn lifecycle
    SESSION_START = "session_start"
    SESSION_RESUME = "session_resume"
    SESSION_BRANCH = "session_branch"
    SESSION_INTERRUPT = "session_interrupt"
    TURN_START = "turn_start"
    TURN_COMPLETE = "turn_complete"

    # Frame lifecycle
    FRAME_START = "frame_start"
    FRAME_COMPLETE = "frame_complete"

    # LLM execution
    LLM_START = "llm_start"
    LLM_CHUNK = "llm_chunk"
    LLM_COMPLETE = "llm_complete"

    # Tool execution
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"

    # Approval
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"

    # Errors/partial
    ERROR = "error"
    PARTIAL_RESPONSE = "partial_response"
    CONTINUATION_DECISION = "continuation_decision"


class BaseEvent(BaseModel):
    """Base class for all runtime events."""

    type: ClassVar[EventType]
    trace_id: str
    turn_id: str
    frame_id: str | None = None
    seq: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    debug_preview: str | None = Field(
        default=None, description="Human-readable preview for CLI/debug"
    )

    def to_stored_event(self) -> StoredEvent:
        """Convert to compact stored event format."""

        # Subclasses must implement this to strip debug data
        raise NotImplementedError("Subclasses must implement to_stored_event")


# === Turn Events ===


class TurnStartEvent(BaseEvent):
    """Event emitted when a turn begins."""

    type: ClassVar[Literal[EventType.TURN_START]] = EventType.TURN_START
    session_id: str
    user_input: str
    turn_index: int
    config_snapshot_json: str | None = None
    agent_mode: str | None = None
    model_name: str | None = None
    context_snapshot_id: str | None = None
    context_snapshot_json: str | None = None
    # Session context profile as of this turn start. The bundle cache may
    # rebuild the profile every turn (dynamic mode / config change), so this
    # keeps the session row's baseline in sync with the snapshot.
    context_profile_json: str | None = None

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=None,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


class TurnCompleteEvent(BaseEvent):
    """Event emitted when a turn completes."""

    type: ClassVar[Literal[EventType.TURN_COMPLETE]] = EventType.TURN_COMPLETE
    session_id: str
    final_answer: str | None = None
    status: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_frames: int = 0
    finish_reason: str | None = None

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=None,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


class SessionStartEvent(BaseEvent):
    """Event emitted when a new session begins (first turn)."""

    type: ClassVar[Literal[EventType.SESSION_START]] = EventType.SESSION_START
    session_id: str
    user_input: str
    agent_mode: str
    model_name: str | None = None
    metadata_json: str | None = None

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=None,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


class SessionResumeEvent(BaseEvent):
    """Event emitted when resuming an existing session."""

    type: ClassVar[Literal[EventType.SESSION_RESUME]] = EventType.SESSION_RESUME
    session_id: str
    resumed_from_turn_id: str | None = None
    context_snapshot_id: str | None = None
    frame_id: str | None = None
    validation_warnings: list[str] = Field(default_factory=list)

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=None,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


class SessionBranchEvent(BaseEvent):
    """Event emitted when a session is branched."""

    type: ClassVar[Literal[EventType.SESSION_BRANCH]] = EventType.SESSION_BRANCH
    session_id: str
    parent_session_id: str
    branch_from_turn_id: str
    new_session_id: str
    # turn_id is inherited from BaseEvent (required), use branch_from_turn_id

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=None,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


class SessionInterruptEvent(BaseEvent):
    """Event emitted when a running turn is interrupted (e.g. stop button / CLI interrupt)."""

    type: ClassVar[Literal[EventType.SESSION_INTERRUPT]] = EventType.SESSION_INTERRUPT
    session_id: str
    turn_id: str
    reason: str = "interrupt"
    paused_at: datetime | None = None

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=None,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


# === Frame Events ===


class FrameStartEvent(BaseEvent):
    type: ClassVar[Literal[EventType.FRAME_START]] = EventType.FRAME_START
    session_id: str
    turn_id: str
    iteration: int
    context_snapshot_id: str
    context_hash: str
    message_boundary: dict[str, int] | None = None
    history_through_seq: int = 0
    prompt_hash: str | None = None
    debug_preview: str | None = None

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=self.frame_id,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


class FrameCompleteEvent(BaseEvent):
    """Event emitted when a frame completes execution."""

    type: ClassVar[Literal[EventType.FRAME_COMPLETE]] = EventType.FRAME_COMPLETE
    session_id: str
    status: str
    finish_reason: str | None = None
    duration_ms: int | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    model_name: str | None = None
    error_message: str | None = None
    llm_message: str | None = None
    tool_call_count: int = 0

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=self.frame_id,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


# === LLM Events ===


class LLMStartEvent(BaseEvent):
    """Event emitted when LLM generation starts."""

    type: ClassVar[Literal[EventType.LLM_START]] = EventType.LLM_START
    session_id: str
    model: str | None = None
    prompt_tokens_estimate: int | None = None

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=self.frame_id,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


class LLMChunkEvent(BaseEvent):
    """Event emitted for each LLM streaming chunk."""

    type: ClassVar[Literal[EventType.LLM_CHUNK]] = EventType.LLM_CHUNK
    session_id: str
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        # Chunks are NOT stored individually - they're aggregated in LLM_COMPLETE
        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=self.frame_id,
            seq=self.seq,
            type=self.type.value,
            payload_json="{}",  # Empty - chunks not persisted
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


class LLMCompleteEvent(BaseEvent):
    """Event emitted when LLM generation completes."""

    type: ClassVar[Literal[EventType.LLM_COMPLETE]] = EventType.LLM_COMPLETE
    session_id: str
    content: str
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    reasoning: str | None = None
    reasoning_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float | None = None

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        # Store compact version
        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=self.frame_id,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


# === Tool Events ===


class ToolStartEvent(BaseEvent):
    """Event emitted when tool execution starts."""

    type: ClassVar[Literal[EventType.TOOL_START]] = EventType.TOOL_START
    session_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    execution_mode: str

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=self.frame_id,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


class ToolResultEvent(BaseEvent):
    """Event emitted when tool execution completes."""

    type: ClassVar[Literal[EventType.TOOL_RESULT]] = EventType.TOOL_RESULT
    session_id: str
    # Workspace identity (gateway WorkspaceManager id) - used to key
    # persisted file baselines; NOT the same as session_id
    workspace_id: str | None = None
    tool_call_id: str
    tool_name: str
    success: bool
    output: Any | None = None
    error: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None
    execution_mode: str | None = None
    file_changes: list[dict[str, Any]] = Field(default_factory=list)
    file_observations: list[dict[str, Any]] = Field(default_factory=list)
    backend: str | None = None
    # Original call arguments, carried for transcript projection (tool
    # messages echo their call alongside the result)
    arguments: dict[str, Any] | None = None

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=self.frame_id,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


# === Approval Events ===


class ApprovalRequestForm(BaseModel):
    """Single approval request form."""

    approval_token: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    # Rule information
    rule_id: str | None = None
    rule_description: str | None = None
    rule_priority: int | None = None
    rule_outcome: str | None = None  # What the rule would decide (ask/allow/deny)
    # Proposed override if user approves
    proposed_override: str | None = (
        None  # Description of what override would be created
    )


class ApprovalResponseForm(BaseModel):
    """Single approval response form."""

    approval_token: str
    tool_call_id: str
    response: str  # YES_ONCE, YES_ALWAYS_SESSION, YES_ALWAYS_PROJECT, NO


class ApprovalRequiredEvent(BaseEvent):
    """Event emitted when tool approval is required.

    Contains multiple approval requests that can be approved together.
    """

    type: ClassVar[Literal[EventType.APPROVAL_REQUIRED]] = EventType.APPROVAL_REQUIRED
    session_id: str
    approval_requests: list[ApprovalRequestForm]
    timeout_seconds: int = 300

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=self.frame_id,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


class ApprovalGrantedEvent(BaseEvent):
    """Event emitted when tool approval is granted."""

    type: ClassVar[Literal[EventType.APPROVAL_GRANTED]] = EventType.APPROVAL_GRANTED
    session_id: str
    approval_token: str
    tool_call_id: str

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=self.frame_id,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


class ApprovalDeniedEvent(BaseEvent):
    """Event emitted when tool approval is denied."""

    type: ClassVar[Literal[EventType.APPROVAL_DENIED]] = EventType.APPROVAL_DENIED
    session_id: str
    approval_token: str
    tool_call_id: str
    reason: str | None = None

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=self.frame_id,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


# === Error/Partial Events ===


class ErrorEvent(BaseEvent):
    """Event emitted on execution error."""

    type: ClassVar[Literal[EventType.ERROR]] = EventType.ERROR
    session_id: str
    error_type: str
    error_message: str
    recoverable: bool = False

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=self.frame_id,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


class PartialResponseEvent(BaseEvent):
    """Event emitted when a partial response is detected."""

    type: ClassVar[Literal[EventType.PARTIAL_RESPONSE]] = EventType.PARTIAL_RESPONSE
    session_id: str
    error_type: str  # truncation, parsing_error, tool_failures
    error_message: str
    raw_response: str | None = None
    handler_context: dict[str, Any] | None = None

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=self.frame_id,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


class ContinuationDecisionEvent(BaseEvent):
    """Event emitted when continuation policy makes a decision."""

    type: ClassVar[Literal[EventType.CONTINUATION_DECISION]] = (
        EventType.CONTINUATION_DECISION
    )
    session_id: str
    decision: str  # CONTINUE, STOP, RETRY, COMPACT
    reason: str
    adjusted_options: dict[str, Any] | None = None
    template_path: str | None = None
    # New fields for policy params
    action: str | None = None  # ContinuationAction value
    stop_reason: str | None = None  # StopReason value
    retry_count: int = 0
    max_tokens_adjustment: int | None = None
    temperature_adjustment: float | None = None
    compact_instruction: str | None = None

    def to_stored_event(self) -> StoredEvent:
        from local_coding_assistant.storage.contracts import StoredEvent

        return StoredEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            frame_id=self.frame_id,
            seq=self.seq,
            type=self.type.value,
            payload_json=self.model_dump_json(
                exclude={
                    "debug_preview",
                    "type",
                    "trace_id",
                    "session_id",
                    "turn_id",
                    "frame_id",
                    "seq",
                    "timestamp",
                }
            ),
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
        )


# Union type for all runtime events
RuntimeEvent = (
    SessionStartEvent
    | SessionResumeEvent
    | SessionBranchEvent
    | SessionInterruptEvent
    | TurnStartEvent
    | TurnCompleteEvent
    | FrameStartEvent
    | FrameCompleteEvent
    | LLMStartEvent
    | LLMChunkEvent
    | LLMCompleteEvent
    | ToolStartEvent
    | ToolResultEvent
    | ApprovalRequiredEvent
    | ApprovalGrantedEvent
    | ApprovalDeniedEvent
    | ErrorEvent
    | PartialResponseEvent
    | ContinuationDecisionEvent
)
