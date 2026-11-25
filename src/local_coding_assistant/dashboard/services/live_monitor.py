"""Event projector: converts RuntimeEvent objects into live state mutations.

The projector is stateless: given a RuntimeEvent and the current live monitor
state, it applies the mutation and returns nothing. All mutations happen
in-place on the LiveMonitor's in-memory structures.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from local_coding_assistant.dashboard.models.live import (
    LiveEvent,
    LiveFrame,
    LiveRunSummary,
    LiveSession,
    LiveStats,
    LiveTurn,
)
from local_coding_assistant.runtime.events.contracts import (
    ApprovalDeniedEvent,
    ApprovalGrantedEvent,
    ApprovalRequiredEvent,
    ErrorEvent,
    EventType,
    FrameCompleteEvent,
    FrameStartEvent,
    LLMChunkEvent,
    LLMCompleteEvent,
    LLMStartEvent,
    RuntimeEvent,
    SessionBranchEvent,
    SessionInterruptEvent,
    SessionResumeEvent,
    SessionStartEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.events.projector")


_PREVIEW_KEYS: dict[EventType, tuple[str, ...]] = {
    EventType.LLM_COMPLETE: (
        "model",
        "provider",
        "finish_reason",
        "input_tokens",
        "output_tokens",
        "tool_calls",
    ),
    EventType.TOOL_RESULT: (
        "tool_name",
        "success",
        "execution_time_ms",
        "error",
    ),
    EventType.TURN_COMPLETE: (
        "status",
        "final_answer",
        "total_input_tokens",
        "total_output_tokens",
        "total_frames",
        "finish_reason",
    ),
    EventType.FRAME_COMPLETE: (
        "status",
        "model_name",
        "finish_reason",
        "duration_ms",
        "input_tokens",
        "output_tokens",
        "error_message",
    ),
    EventType.TURN_START: ("user_input", "turn_index", "agent_mode", "model_name"),
    EventType.FRAME_START: ("iteration", "model_name"),
    EventType.SESSION_START: ("user_input", "agent_mode", "model_name"),
    EventType.ERROR: ("error_type", "error_message", "recoverable"),
    EventType.CONTINUATION_DECISION: ("decision", "action", "stop_reason", "reason"),
}


def _to_live_event(event: RuntimeEvent) -> LiveEvent:
    """Convert a RuntimeEvent to a LiveEvent (strips debug/internal fields)."""
    payload: dict[str, Any] = {}
    for key, value in event.model_dump().items():
        if key in (
            "type",
            "trace_id",
            "turn_id",
            "frame_id",
            "seq",
            "timestamp",
            "debug_preview",
        ):
            continue
        if isinstance(value, datetime):
            payload[key] = value.isoformat()
        else:
            payload[key] = value
    return LiveEvent(
        seq=event.seq,
        type=event.type,
        timestamp=event.timestamp,
        frame_id=event.frame_id,
        payload=payload,
    )


def _preview_fields(event_type: EventType, payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the important preview fields for an event type."""
    preview: dict[str, Any] = {}

    keys = _PREVIEW_KEYS.get(event_type, ())
    for key in keys:
        if key in payload:
            preview[key] = payload[key]

    if event_type == EventType.LLM_COMPLETE:
        content = payload.get("content")
        if content:
            preview["content"] = str(content)[:200]
    elif event_type == EventType.APPROVAL_REQUIRED:
        preview["approval_requests"] = len(payload.get("approval_requests", []))
        preview["timeout_seconds"] = payload.get("timeout_seconds")
    elif event_type == EventType.APPROVAL_GRANTED:
        preview["approval_token"] = payload.get("approval_token")
        preview["tool_call_id"] = payload.get("tool_call_id")
    elif event_type == EventType.APPROVAL_DENIED:
        preview["approval_token"] = payload.get("approval_token")
        preview["tool_call_id"] = payload.get("tool_call_id")
        preview["reason"] = payload.get("reason")
    elif event_type == EventType.SESSION_INTERRUPT:
        preview["reason"] = payload.get("reason")
    elif event_type == EventType.PARTIAL_RESPONSE:
        preview["error_type"] = payload.get("error_type")
        preview["error_message"] = payload.get("error_message")

    return preview


class EventProjector:
    """Projects runtime events onto the live monitor state."""

    def __init__(self, max_events_per_frame: int = 200):
        self.max_events_per_frame = max_events_per_frame

    def project(self, event: RuntimeEvent, monitor: LiveMonitor) -> None:
        """Apply a runtime event to the live monitor state."""
        session = self._get_or_create_session(event, monitor)
        session.updated_at = event.timestamp

        live_event = _to_live_event(event)
        self._mutate_state(event, session, monitor)
        self._append_event_to_frame(event, session, live_event)

        # Retain the event keyed by turn so the run details timeline can serve
        # the complete turn timeline (session/turn lifecycle events have
        # frame_id=None and are not stored on any frame).
        if event.turn_id:
            monitor._turn_events.setdefault(event.turn_id, []).append(live_event)

    def _get_or_create_session(
        self, event: RuntimeEvent, monitor: LiveMonitor
    ) -> LiveSession:
        session = monitor.sessions.get(event.session_id)
        if session is None:
            session = LiveSession(
                session_id=event.session_id,
                status="idle",
                started_at=event.timestamp,
                updated_at=event.timestamp,
                user_input="",
                agent_mode="",
            )
            monitor.sessions[event.session_id] = session
        return session

    def _mutate_state(
        self, event: RuntimeEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        handler = self._event_handler_for(event.type)
        if handler:
            handler(event, session, monitor)

    def _event_handler_for(self, event_type: EventType) -> Any | None:
        return {
            EventType.SESSION_START: self._handle_session_start,
            EventType.SESSION_RESUME: self._handle_session_resume,
            EventType.SESSION_INTERRUPT: self._handle_session_interrupt,
            EventType.SESSION_BRANCH: self._handle_branch,
            EventType.TURN_START: self._handle_turn_start,
            EventType.TURN_COMPLETE: self._handle_turn_complete,
            EventType.FRAME_START: self._handle_frame_start,
            EventType.FRAME_COMPLETE: self._handle_frame_complete,
            EventType.LLM_START: self._handle_llm_start,
            EventType.LLM_CHUNK: self._handle_llm_chunk,
            EventType.LLM_COMPLETE: self._handle_llm_complete,
            EventType.TOOL_START: self._handle_tool_start,
            EventType.TOOL_RESULT: self._handle_tool_result,
            EventType.APPROVAL_REQUIRED: self._handle_approval_required,
            EventType.APPROVAL_GRANTED: self._handle_approval_responded,
            EventType.APPROVAL_DENIED: self._handle_approval_responded,
            EventType.ERROR: self._handle_error,
        }.get(event_type)

    def _handle_session_start(
        self, event: SessionStartEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        session.status = "running"
        session.user_input = event.user_input
        session.agent_mode = event.agent_mode or ""
        session.model_name = event.model_name

    def _handle_session_resume(
        self, event: SessionResumeEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        session.status = "running"

    def _handle_session_interrupt(
        self, event: SessionInterruptEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        session.status = "interrupted"

    def _handle_branch(
        self, event: SessionBranchEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        new_session_id = event.new_session_id
        if new_session_id and new_session_id not in monitor.sessions:
            monitor.sessions[new_session_id] = LiveSession(
                session_id=new_session_id,
                status="running",
                started_at=event.timestamp,
                updated_at=event.timestamp,
                user_input="",
                agent_mode="",
            )

    def _handle_turn_start(
        self, event: TurnStartEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        turn = LiveTurn(
            turn_id=event.turn_id,
            turn_index=event.turn_index,
            trace_id=event.trace_id,
            user_input=event.user_input,
            status="running",
            started_at=event.timestamp,
        )
        session.turns[event.turn_id] = turn

    def _handle_turn_complete(
        self, event: TurnCompleteEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        turn = session.turns.get(event.turn_id)
        if turn:
            turn.status = event.status
            turn.finished_at = event.timestamp
            turn.final_answer = event.final_answer
            turn.total_frames = event.total_frames or 0

    def _handle_frame_start(
        self, event: FrameStartEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        turn = session.turns.get(event.turn_id)
        if turn:
            frame = LiveFrame(
                frame_id=event.frame_id or "",
                iteration=event.iteration,
                status="running_llm",
                started_at=event.timestamp,
            )
            turn.frames[frame.frame_id] = frame
            turn.total_frames = len(turn.frames)

    def _handle_frame_complete(
        self, event: FrameCompleteEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        turn = session.turns.get(event.turn_id)
        if turn and event.frame_id and event.frame_id in turn.frames:
            frame = turn.frames[event.frame_id]
            frame.status = event.status
            frame.finished_at = event.timestamp
            frame.duration_ms = event.duration_ms
            frame.model_name = event.model_name
            frame.finish_reason = event.finish_reason
            frame.error_message = event.error_message
            frame.tool_call_count = event.tool_call_count or 0

    def _handle_llm_start(
        self, event: LLMStartEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        turn = session.turns.get(event.turn_id)
        if turn and event.frame_id and event.frame_id in turn.frames:
            frame = turn.frames[event.frame_id]
            frame.status = "running_llm"
            frame.model_name = event.model

    def _handle_llm_complete(
        self, event: LLMCompleteEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        turn = session.turns.get(event.turn_id)
        if turn and event.frame_id and event.frame_id in turn.frames:
            frame = turn.frames[event.frame_id]
            frame.status = "running_tools"
            frame.model_name = event.model
            frame.input_tokens = (
                event.input_tokens
                if event.input_tokens is not None
                else event.prompt_tokens or 0
            )
            frame.output_tokens = (
                event.output_tokens
                if event.output_tokens is not None
                else event.completion_tokens or 0
            )
            frame.cache_tokens = event.cached_tokens or 0
            turn.input_tokens += frame.input_tokens
            turn.output_tokens += frame.output_tokens
            turn.cache_tokens += frame.cache_tokens

    def _handle_llm_chunk(
        self, event: LLMChunkEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        # Streaming chunks don't change frame state; recorded as events only
        pass

    def _handle_tool_start(
        self, event: ToolStartEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        turn = session.turns.get(event.turn_id)
        if turn and event.frame_id and event.frame_id in turn.frames:
            frame = turn.frames[event.frame_id]
            frame.status = "running_tools"
            frame.tool_call_count += 1

    def _handle_tool_result(
        self, event: ToolResultEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        turn = session.turns.get(event.turn_id)
        if turn and event.frame_id and event.frame_id in turn.frames:
            frame = turn.frames[event.frame_id]
            frame.status = "running_tools"
            if not event.success:
                frame.status = "failed"
                frame.error_message = (
                    event.error.get("message") if event.error else "Tool failed"
                )
                frame.finished_at = event.timestamp

    def _handle_approval_required(
        self, event: ApprovalRequiredEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        turn = session.turns.get(event.turn_id)
        if turn and event.frame_id and event.frame_id in turn.frames:
            turn.frames[event.frame_id].status = "paused_approval"
            turn.frames[event.frame_id].pending_call_id = (
                event.approval_requests[0].approval_token
                if event.approval_requests
                else None
            )
            turn.status = "paused"
            turn.pause_reason = "approval"

    def _handle_approval_responded(
        self,
        event: ApprovalGrantedEvent | ApprovalDeniedEvent,
        session: LiveSession,
        monitor: LiveMonitor,
    ) -> None:
        turn = session.turns.get(event.turn_id)
        if turn and event.frame_id and event.frame_id in turn.frames:
            turn.frames[event.frame_id].status = "running_tools"
            turn.frames[event.frame_id].pending_call_id = None

    def _handle_error(
        self, event: ErrorEvent, session: LiveSession, monitor: LiveMonitor
    ) -> None:
        turn = session.turns.get(event.turn_id)
        if turn:
            turn.status = "error"
            if event.frame_id and event.frame_id in turn.frames:
                turn.frames[event.frame_id].status = "failed"
                turn.frames[event.frame_id].error_message = event.error_message
                turn.frames[event.frame_id].finished_at = event.timestamp
            session.status = "error"

    def _append_event_to_frame(
        self,
        event: RuntimeEvent,
        session: LiveSession,
        live_event: LiveEvent,
    ) -> None:
        frame_id = event.frame_id
        turn_id = event.turn_id
        if frame_id:
            turn = session.turns.get(turn_id)
            if turn and frame_id in turn.frames:
                frame = turn.frames[frame_id]
                frame.events.append(live_event)
                if len(frame.events) > self.max_events_per_frame:
                    frame.events = frame.events[-self.max_events_per_frame :]


class LiveMonitor:
    """In-memory live monitoring state reconstructed from runtime events.

    The live monitor receives runtime events via HTTP ingestion (the same
    wire protocol used by DashboardEventSubscriber) and maintains an
    in-memory model of active sessions, turns, and frames. It is the
    backing store for the runs page (live monitoring).
    """

    def __init__(
        self,
        max_events: int = 4000,
        max_events_per_frame: int = 200,
        max_runs: int = 50,
    ):
        self.max_events = max_events
        self.max_events_per_frame = max_events_per_frame
        self.max_runs = max_runs

        self.sessions: dict[str, LiveSession] = {}
        self.projector = EventProjector(max_events_per_frame=max_events_per_frame)

        # Every ingested runtime event, keyed by turn_id so the run details
        # event timeline can reproduce the full turn timeline (including
        # session/turn lifecycle events that are not scoped to a single
        # frame), matching the historical store-backed timeline.
        self._turn_events: dict[str, list[LiveEvent]] = {}

        # Event rate tracking
        self._event_timestamps: list[datetime] = []
        self._error_timestamps: list[datetime] = []

    def ingest_event(self, event: RuntimeEvent) -> None:
        """Ingest a single runtime event into the live monitor."""
        self.projector.project(event, self)

        now = datetime.now(UTC)
        self._event_timestamps.append(now)
        if len(self._event_timestamps) > self.max_events:
            self._event_timestamps = self._event_timestamps[-self.max_events :]

        if event.type == EventType.ERROR:
            self._error_timestamps.append(now)

    def ingest_events(self, events: list[RuntimeEvent]) -> None:
        """Ingest multiple runtime events."""
        for event in events:
            self.ingest_event(event)

    def get_active_sessions(self) -> list[LiveSession]:
        """Return sessions with status 'running' or 'paused', sorted by updated_at desc."""
        active = [
            s
            for s in self.sessions.values()
            if s.status in ("running", "paused", "interrupted")
        ]
        active.sort(key=lambda s: s.updated_at, reverse=True)
        return active

    def get_session(self, session_id: str) -> LiveSession | None:
        """Get a live session by ID."""
        return self.sessions.get(session_id)

    def get_turn_events(self, turn_id: str) -> list[LiveEvent]:
        """Return all ingested events for a turn, sorted by seq."""
        return sorted(self._turn_events.get(turn_id, []), key=lambda e: e.seq)

    def get_run_summaries(self) -> list[LiveRunSummary]:
        """Return summaries of all live runs (turns), sorted by started_at desc."""
        summaries: list[LiveRunSummary] = []
        for session in self.sessions.values():
            for turn in session.turns.values():
                current_frame: LiveFrame | None = None
                current_frame_status: str | None = None
                for frame in turn.frames.values():
                    if frame.status not in ("completed", "failed"):
                        current_frame = frame
                        current_frame_status = frame.status
                        break
                if current_frame is None and turn.frames:
                    current_frame = list(turn.frames.values())[-1]
                    current_frame_status = current_frame.status

                duration: float | None = None
                if turn.finished_at and turn.started_at:
                    duration = (turn.finished_at - turn.started_at).total_seconds()

                summaries.append(
                    LiveRunSummary(
                        run_id=turn.turn_id,
                        session_id=session.session_id,
                        turn_index=turn.turn_index,
                        status=turn.status,
                        started_at=turn.started_at,
                        finished_at=turn.finished_at,
                        duration=duration,
                        input_tokens=turn.input_tokens,
                        output_tokens=turn.output_tokens,
                        cache_tokens=turn.cache_tokens,
                        total_frames=turn.total_frames,
                        current_frame_id=current_frame.frame_id
                        if current_frame
                        else None,
                        current_frame_status=current_frame_status,
                    )
                )
        summaries.sort(key=lambda s: s.started_at, reverse=True)
        return summaries

    def get_stats(self) -> LiveStats:
        """Compute live aggregate statistics."""
        now = datetime.now(UTC)
        one_minute_ago = now - timedelta(minutes=1)

        recent_events = sum(1 for ts in self._event_timestamps if ts > one_minute_ago)
        recent_errors = sum(1 for ts in self._error_timestamps if ts > one_minute_ago)

        eps = recent_events / 60.0

        return LiveStats(
            active_sessions=len(self.get_active_sessions()),
            running_turns=sum(
                1
                for s in self.sessions.values()
                for t in s.turns.values()
                if t.status == "running"
            ),
            total_events=len(self._event_timestamps),
            events_per_second=eps,
            error_count_last_minute=recent_errors,
        )

    def clear(self) -> None:
        """Clear all live state (for testing/reset)."""
        self.sessions.clear()
        self._event_timestamps.clear()
        self._error_timestamps.clear()
