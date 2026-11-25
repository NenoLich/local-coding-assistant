"""
Unit tests for dashboard Pydantic models (live + historical planes).
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from local_coding_assistant.dashboard.models import (
    ErrorResponse,
    EventTimelineEntry,
    FrameDetail,
    FrameSummary,
    LiveEvent,
    LiveFrame,
    LiveRunSummary,
    LiveSession,
    LiveStats,
    LiveTurn,
    MessageDetail,
    MessageEntry,
    PaginatedResponse,
    PaginationParams,
    SessionDetail,
    SessionSummary,
    StatusSummary,
    TurnDetail,
    TurnSummary,
)
from local_coding_assistant.runtime.events.contracts import EventType


def _dt() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


class TestPaginationParams:
    def test_defaults(self):
        params = PaginationParams()
        assert params.limit == 50
        assert params.offset == 0

    def test_limit_bounds(self):
        with pytest.raises(ValidationError):
            PaginationParams(limit=0)
        with pytest.raises(ValidationError):
            PaginationParams(limit=1001)
        PaginationParams(limit=1)
        PaginationParams(limit=1000)


class TestPaginatedResponse:
    def test_round_trip(self):
        response = PaginatedResponse(
            items=[{"id": 1}],
            total=10,
            offset=0,
            limit=2,
            has_next=True,
            has_prev=False,
        )
        assert response.has_next is True
        assert response.has_prev is False
        data = response.model_dump()
        assert data["total"] == 10


class TestErrorResponse:
    def test_minimal(self):
        err = ErrorResponse(error="not found")
        assert err.error == "not found"
        assert err.detail is None

    def test_with_detail(self):
        err = ErrorResponse(error="not found", detail="session missing")
        assert err.detail == "session missing"


class TestStatusSummary:
    def test_creation(self):
        summary = StatusSummary(total=10, running=2, completed=6, error=1, idle=1)
        assert summary.total == 10
        assert summary.running == 2


# --- Live models ---


class TestLiveEvent:
    def test_creation(self):
        event = LiveEvent(
            seq=1,
            type=EventType.TURN_START,
            timestamp=_dt(),
            frame_id=None,
            payload={"user_input": "hi"},
        )
        assert event.seq == 1
        assert event.type == EventType.TURN_START
        assert event.frame_id is None


class TestLiveFrame:
    def test_creation(self):
        frame = LiveFrame(
            frame_id="frame-1",
            iteration=0,
            status="running_llm",
            started_at=_dt(),
            model_name="gpt-4",
        )
        assert frame.frame_id == "frame-1"
        assert frame.status == "running_llm"
        assert frame.events == []


class TestLiveTurn:
    def test_creation(self):
        turn = LiveTurn(
            turn_id="turn-1",
            turn_index=0,
            trace_id="trace-1",
            user_input="hello",
            status="running",
            started_at=_dt(),
        )
        assert turn.turn_index == 0
        assert turn.frames == {}


class TestLiveSession:
    def test_creation(self):
        session = LiveSession(
            session_id="session-1",
            status="running",
            started_at=_dt(),
            updated_at=_dt(),
            user_input="hello",
            agent_mode="frame_agent",
        )
        assert session.status == "running"
        assert session.turns == {}


class TestLiveRunSummary:
    def test_creation(self):
        summary = LiveRunSummary(
            run_id="turn-1",
            session_id="session-1",
            turn_index=0,
            status="running",
            started_at=_dt(),
            input_tokens=100,
            output_tokens=50,
            total_frames=2,
        )
        assert summary.run_id == "turn-1"
        assert summary.duration is None


class TestLiveStats:
    def test_creation(self):
        stats = LiveStats(
            active_sessions=2,
            running_turns=1,
            total_events=10,
            events_per_second=0.5,
            error_count_last_minute=0,
        )
        assert stats.active_sessions == 2


# --- Historical models ---


class TestSessionSummary:
    def test_creation(self):
        summary = SessionSummary(
            session_id="session-1",
            user_input="hello",
            status="completed",
            started_at=_dt(),
            agent_mode="frame_agent",
            model_name="gpt-4",
            total_turns=2,
            total_frames=3,
        )
        assert summary.total_turns == 2


class TestTurnSummary:
    def test_creation(self):
        summary = TurnSummary(
            turn_id="turn-1",
            session_id="session-1",
            turn_index=0,
            trace_id="trace-1",
            user_input="hello",
            status="completed",
            started_at=_dt(),
        )
        assert summary.pause_reason is None


class TestFrameSummary:
    def test_creation(self):
        summary = FrameSummary(
            frame_id="frame-1",
            turn_id="turn-1",
            iteration=0,
            status="completed",
            started_at=_dt(),
            model_name="gpt-4",
            input_tokens=100,
            output_tokens=50,
        )
        assert summary.iteration == 0


class TestEventTimelineEntry:
    def test_creation(self):
        entry = EventTimelineEntry(
            event_id=1,
            seq=1,
            type="turn_start",
            timestamp=_dt(),
            frame_id=None,
            preview={"user_input": "hello"},
            payload={"user_input": "hello"},
            has_payload=True,
        )
        assert entry.event_id == 1
        assert entry.has_payload is True

    def test_defaults(self):
        entry = EventTimelineEntry(event_id=1, seq=1, type="x")
        assert entry.preview == {}
        assert entry.payload == {}
        assert entry.has_payload is False


class TestMessageEntry:
    def test_creation(self):
        message = MessageEntry(
            seq=1,
            role="user",
            content="hello",
            turn_id="turn-1",
        )
        assert message.role == "user"
        assert message.tool_call_id is None


class TestMessageDetail:
    def test_creation(self):
        entry = MessageEntry(seq=1, role="assistant", content="hi", turn_id="turn-1")
        detail = MessageDetail(message=entry.model_dump(), raw_payload={"a": 1})
        assert detail.message.role == "assistant"
        assert detail.raw_payload == {"a": 1}


class TestContextBlock:
    def test_creation(self):
        from local_coding_assistant.dashboard.models.historical import ContextBlock

        block = ContextBlock(
            system_messages=["system: be helpful"],
            tool_schemas=[{"name": "echo"}],
            warnings=["partial context"],
            resolved=False,
        )
        assert block.resolved is False
        assert block.system_reminder is None


class TestSessionDetail:
    def test_creation(self):
        session = SessionSummary(
            session_id="session-1",
            user_input="hello",
            status="completed",
            started_at=_dt(),
            agent_mode="frame_agent",
        )
        turn = TurnSummary(
            turn_id="turn-1",
            session_id="session-1",
            turn_index=0,
            trace_id="trace-1",
            user_input="hello",
            status="completed",
            started_at=_dt(),
        )
        detail = SessionDetail(session=session, turns=[turn])
        assert len(detail.turns) == 1


class TestTurnDetail:
    def test_creation(self):
        turn = TurnSummary(
            turn_id="turn-1",
            session_id="session-1",
            turn_index=0,
            trace_id="trace-1",
            user_input="hello",
            status="completed",
            started_at=_dt(),
        )
        frame = FrameSummary(
            frame_id="frame-1",
            turn_id="turn-1",
            iteration=0,
            status="completed",
            started_at=_dt(),
        )
        message = MessageEntry(seq=1, role="user", content="hi", turn_id="turn-1")
        detail = TurnDetail(turn=turn, frames=[frame], messages=[message])
        assert len(detail.frames) == 1
        assert len(detail.messages) == 1


class TestFrameDetail:
    def test_creation(self):
        frame = FrameSummary(
            frame_id="frame-1",
            turn_id="turn-1",
            iteration=0,
            status="completed",
            started_at=_dt(),
        )
        entry = EventTimelineEntry(event_id=1, seq=1, type="x")
        detail = FrameDetail(frame=frame, events=[entry])
        assert len(detail.events) == 1

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            FrameSummary(
                frame_id="frame-1",
                # missing turn_id
                iteration=0,
                status="completed",
                started_at=_dt(),
            )
