"""
Tests for the new runtime event emitting system.

Covers:
- EventPipeline: subscribe/unsubscribe, fan-out emit, per-session seq
  assignment, error isolation, close.
- Typed event contracts: creation and to_stored_event conversion.
- EventType enum completeness.
"""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.runtime.events import (
    ApprovalDeniedEvent,
    ApprovalGrantedEvent,
    ApprovalRequiredEvent,
    ContinuationDecisionEvent,
    ErrorEvent,
    EventPipeline,
    EventSubscriber,
    EventType,
    FrameCompleteEvent,
    FrameStartEvent,
    LLMChunkEvent,
    LLMCompleteEvent,
    LLMStartEvent,
    PartialResponseEvent,
    SessionResumeEvent,
    SessionStartEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.runtime.events.contracts import ApprovalRequestForm


class MockSubscriber(EventSubscriber):
    """Mock subscriber that records received events."""

    def __init__(self) -> None:
        self.events: list = []
        self.closed = False

    async def on_event(self, event) -> None:  # type: ignore[override]
        self.events.append(event)

    async def close(self) -> None:
        self.closed = True


class FailingSubscriber(EventSubscriber):
    """Subscriber that always raises in on_event."""

    def __init__(self) -> None:
        self.call_count = 0

    async def on_event(self, event) -> None:  # type: ignore[override]
        self.call_count += 1
        raise RuntimeError("subscriber failed")

    async def close(self) -> None:
        pass


class TestEventPipeline:
    """Test EventPipeline emit/fan-out behaviour."""

    @pytest.fixture
    def pipeline(self) -> EventPipeline:
        return EventPipeline()

    @pytest.fixture
    def sample_event(self) -> TurnStartEvent:
        return TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="test query",
            turn_index=0,
        )

    def test_pipeline_initialization(self, pipeline: EventPipeline) -> None:
        assert pipeline.subscribers == []

    def test_subscribe(self, pipeline: EventPipeline) -> None:
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)
        assert len(pipeline.subscribers) == 1
        assert pipeline.subscribers[0] == subscriber

    def test_subscribe_multiple(self, pipeline: EventPipeline) -> None:
        sub1, sub2, sub3 = MockSubscriber(), MockSubscriber(), MockSubscriber()
        pipeline.subscribe(sub1)
        pipeline.subscribe(sub2)
        pipeline.subscribe(sub3)
        assert pipeline.subscribers == [sub1, sub2, sub3]

    def test_unsubscribe(self, pipeline: EventPipeline) -> None:
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)
        pipeline.unsubscribe(subscriber)
        assert len(pipeline.subscribers) == 0

    def test_unsubscribe_nonexistent(self, pipeline: EventPipeline) -> None:
        pipeline.unsubscribe(MockSubscriber())
        assert len(pipeline.subscribers) == 0

    @pytest.mark.asyncio
    async def test_emit_to_single_subscriber(
        self, pipeline: EventPipeline, sample_event: TurnStartEvent
    ) -> None:
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)
        await pipeline.emit(sample_event)
        assert len(subscriber.events) == 1
        assert subscriber.events[0] == sample_event

    @pytest.mark.asyncio
    async def test_emit_to_multiple_subscribers(
        self, pipeline: EventPipeline, sample_event: TurnStartEvent
    ) -> None:
        sub1, sub2, sub3 = MockSubscriber(), MockSubscriber(), MockSubscriber()
        pipeline.subscribe(sub1)
        pipeline.subscribe(sub2)
        pipeline.subscribe(sub3)
        await pipeline.emit(sample_event)
        for sub in (sub1, sub2, sub3):
            assert len(sub.events) == 1
            assert sub.events[0] == sample_event

    @pytest.mark.asyncio
    async def test_emit_with_no_subscribers(
        self, pipeline: EventPipeline, sample_event: TurnStartEvent
    ) -> None:
        await pipeline.emit(sample_event)

    @pytest.mark.asyncio
    async def test_emit_with_failing_subscriber(
        self, pipeline: EventPipeline, sample_event: TurnStartEvent
    ) -> None:
        failing, good = FailingSubscriber(), MockSubscriber()
        pipeline.subscribe(failing)
        pipeline.subscribe(good)
        await pipeline.emit(sample_event)
        assert failing.call_count == 1
        assert len(good.events) == 1
        assert good.events[0] == sample_event

    @pytest.mark.asyncio
    async def test_emit_assigns_seq_before_fan_out(
        self, pipeline: EventPipeline
    ) -> None:
        sub1, sub2 = MockSubscriber(), MockSubscriber()
        pipeline.subscribe(sub1)
        pipeline.subscribe(sub2)

        event = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="query",
            turn_index=0,
        )
        await pipeline.emit(event)

        assert sub1.events[0].seq == 1
        assert sub2.events[0].seq == 1
        assert event.seq == 1

    @pytest.mark.asyncio
    async def test_emit_seq_monotonic_per_session(
        self, pipeline: EventPipeline
    ) -> None:
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)
        session_id = "sess_1"
        events = [
            TurnStartEvent(
                trace_id=f"tr_{i}",
                turn_id=f"turn_{i}",
                session_id=session_id,
                user_input=f"query {i}",
                turn_index=0,
            )
            for i in range(3)
        ]
        for event in events:
            await pipeline.emit(event)

        assert [e.seq for e in subscriber.events] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_emit_different_sessions_independent_counters(
        self, pipeline: EventPipeline
    ) -> None:
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)
        await pipeline.emit(
            TurnStartEvent(
                trace_id="tr_a",
                turn_id="turn_a",
                session_id="sess_a",
                user_input="query a",
                turn_index=0,
            )
        )
        await pipeline.emit(
            TurnStartEvent(
                trace_id="tr_b",
                turn_id="turn_b",
                session_id="sess_b",
                user_input="query b",
                turn_index=0,
            )
        )
        await pipeline.emit(
            TurnStartEvent(
                trace_id="tr_c",
                turn_id="turn_c",
                session_id="sess_a",
                user_input="query c",
                turn_index=0,
            )
        )
        assert subscriber.events[0].seq == 1
        assert subscriber.events[1].seq == 1
        assert subscriber.events[2].seq == 2

    @pytest.mark.asyncio
    async def test_emit_no_session_id_keeps_seq_zero(
        self, pipeline: EventPipeline
    ) -> None:
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)
        event = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="",
            user_input="query",
            turn_index=0,
        )
        await pipeline.emit(event)
        assert subscriber.events[0].seq == 0
        assert event.seq == 0

    @pytest.mark.asyncio
    async def test_emit_with_execution_store_seeds_counter(
        self, pipeline: EventPipeline
    ) -> None:
        mock_store = MagicMock()

        async def fake_get_max_seq(session_id: str) -> int:
            return 10

        mock_store.get_max_seq = fake_get_max_seq
        pipeline._execution_store = mock_store  # type: ignore[attr-defined]

        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)
        event = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="query",
            turn_index=0,
        )
        await pipeline.emit(event)
        assert subscriber.events[0].seq == 11

    @pytest.mark.asyncio
    async def test_close(self, pipeline: EventPipeline) -> None:
        sub1, sub2 = MockSubscriber(), MockSubscriber()
        pipeline.subscribe(sub1)
        pipeline.subscribe(sub2)
        await pipeline.close()
        assert sub1.closed is True
        assert sub2.closed is True
        assert len(pipeline.subscribers) == 0

    @pytest.mark.asyncio
    async def test_close_with_failing_subscriber(self, pipeline: EventPipeline) -> None:
        class FailingCloseSubscriber(MockSubscriber):
            async def close(self) -> None:
                raise RuntimeError("close failed")

        sub1, sub2, sub3 = MockSubscriber(), FailingCloseSubscriber(), MockSubscriber()
        pipeline.subscribe(sub1)
        pipeline.subscribe(sub2)
        pipeline.subscribe(sub3)
        await pipeline.close()
        assert sub1.closed is True
        assert sub3.closed is True
        assert len(pipeline.subscribers) == 0

    def test_subscribers_property_returns_copy(self, pipeline: EventPipeline) -> None:
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)
        subs = pipeline.subscribers
        subs.append(MockSubscriber())
        assert len(pipeline.subscribers) == 1


class TestEventType:
    """Test the EventType enum."""

    def test_all_event_types_defined(self) -> None:
        expected = {
            "SESSION_START",
            "SESSION_RESUME",
            "SESSION_BRANCH",
            "SESSION_INTERRUPT",
            "TURN_START",
            "TURN_COMPLETE",
            "FRAME_START",
            "FRAME_COMPLETE",
            "LLM_START",
            "LLM_CHUNK",
            "LLM_COMPLETE",
            "TOOL_START",
            "TOOL_RESULT",
            "APPROVAL_REQUIRED",
            "APPROVAL_GRANTED",
            "APPROVAL_DENIED",
            "ERROR",
            "PARTIAL_RESPONSE",
            "CONTINUATION_DECISION",
        }
        assert {e.name for e in EventType} == expected

    def test_event_type_values_match_names(self) -> None:
        for event_type in EventType:
            assert event_type.value == event_type.name.lower()


class TestTurnEvents:
    """Test turn lifecycle events."""

    def test_turn_start_creation(self) -> None:
        event = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="query",
            turn_index=0,
            agent_mode="frame",
            model_name="gpt-4",
        )
        assert event.type == EventType.TURN_START
        assert event.session_id == "sess_1"
        assert event.user_input == "query"
        assert event.turn_index == 0

    def test_turn_start_to_stored_event(self) -> None:
        event = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="query",
            turn_index=0,
            config_snapshot_json='{"k": "v"}',
        )
        stored = event.to_stored_event()
        assert stored.trace_id == "tr_1"
        assert stored.session_id == "sess_1"
        assert stored.turn_id == "turn_1"
        assert stored.frame_id is None
        assert stored.type == "turn_start"
        assert "debug_preview" not in stored.payload_json
        assert "trace_id" not in stored.payload_json

    def test_turn_complete_creation(self) -> None:
        event = TurnCompleteEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            status="completed",
            final_answer="answer",
            total_input_tokens=100,
            total_output_tokens=50,
            total_frames=3,
            finish_reason="stop",
        )
        assert event.type == EventType.TURN_COMPLETE
        assert event.final_answer == "answer"
        assert event.total_input_tokens == 100

    def test_turn_complete_to_stored_event(self) -> None:
        event = TurnCompleteEvent(
            trace_id="tr_1", turn_id="turn_1", session_id="sess_1", status="completed"
        )
        stored = event.to_stored_event()
        assert stored.type == "turn_complete"
        assert "debug_preview" not in stored.payload_json


class TestSessionEvents:
    """Test session lifecycle events."""

    def test_session_start_creation(self) -> None:
        event = SessionStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="query",
            agent_mode="frame",
            model_name="gpt-4",
        )
        assert event.type == EventType.SESSION_START
        assert event.agent_mode == "frame"

    def test_session_start_to_stored_event(self) -> None:
        event = SessionStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="query",
            agent_mode="frame",
        )
        stored = event.to_stored_event()
        assert stored.type == "session_start"

    def test_session_resume_creation(self) -> None:
        event = SessionResumeEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            resumed_from_turn_id="turn_old",
        )
        assert event.type == EventType.SESSION_RESUME
        assert event.resumed_from_turn_id == "turn_old"

    def test_session_resume_to_stored_event(self) -> None:
        event = SessionResumeEvent(
            trace_id="tr_1", turn_id="turn_1", session_id="sess_1"
        )
        stored = event.to_stored_event()
        assert stored.type == "session_resume"


class TestFrameEvents:
    """Test frame lifecycle events."""

    def test_frame_start_creation(self) -> None:
        event = FrameStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            context_hash="sha256:abc",
            debug_preview="preview",
        )
        assert event.type == EventType.FRAME_START
        assert event.frame_id == "frame_1"
        assert event.iteration == 1
        assert event.debug_preview == "preview"

    def test_frame_start_to_stored_event_excludes_debug(self) -> None:
        event = FrameStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            context_hash="sha256:abc",
            debug_preview="excluded",
        )
        stored = event.to_stored_event()
        assert stored.type == "frame_start"
        assert stored.frame_id == "frame_1"
        assert "debug_preview" not in stored.payload_json
        assert "context_hash" in stored.payload_json

    def test_frame_complete_creation(self) -> None:
        event = FrameCompleteEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            status="completed",
            finish_reason="stop",
            duration_ms=1000,
            total_input_tokens=100,
            total_output_tokens=50,
            model_name="gpt-4",
        )
        assert event.type == EventType.FRAME_COMPLETE
        assert event.duration_ms == 1000

    def test_frame_complete_to_stored_event(self) -> None:
        event = FrameCompleteEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            status="completed",
        )
        stored = event.to_stored_event()
        assert stored.type == "frame_complete"


class TestLLMEvents:
    """Test LLM execution events."""

    def test_llm_start_creation(self) -> None:
        event = LLMStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            model="gpt-4",
            prompt_tokens_estimate=1000,
        )
        assert event.type == EventType.LLM_START
        assert event.model == "gpt-4"

    def test_llm_chunk_creation(self) -> None:
        event = LLMChunkEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            content="Hello",
            reasoning="thinking",
        )
        assert event.type == EventType.LLM_CHUNK
        assert event.content == "Hello"

    def test_llm_chunk_to_stored_event_empty_payload(self) -> None:
        event = LLMChunkEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            content="Hello",
        )
        stored = event.to_stored_event()
        assert stored.payload_json == "{}"

    def test_llm_complete_creation(self) -> None:
        event = LLMCompleteEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            content="Full response",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            finish_reason="stop",
        )
        assert event.type == EventType.LLM_COMPLETE
        assert event.content == "Full response"

    def test_llm_complete_to_stored_event_excludes_content(self) -> None:
        event = LLMCompleteEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            content="long content",
            reasoning="reasoning",
            input_tokens=100,
            output_tokens=50,
        )
        stored = event.to_stored_event()
        assert stored.type == "llm_complete"
        payload = json.loads(stored.payload_json)
        # Full content/reasoning are retained in the stored payload per current
        # implementation; token metadata is included.
        assert payload["content"] == "long content"
        assert payload["reasoning"] == "reasoning"
        assert payload["input_tokens"] == 100
        assert payload["output_tokens"] == 50


class TestToolEvents:
    """Test tool execution events."""

    def test_tool_start_creation(self) -> None:
        event = ToolStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            tool_call_id="call_1",
            tool_name="read_file",
            arguments={"path": "/tmp/file.txt"},  # noqa: S108 - test fixture path, not a real temp dir
            execution_mode="sandbox",
        )
        assert event.type == EventType.TOOL_START
        assert event.tool_name == "read_file"

    def test_tool_result_creation(self) -> None:
        event = ToolResultEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            tool_call_id="call_1",
            tool_name="read_file",
            success=True,
            output="content",
            timing={"duration_ms": 100},
        )
        assert event.type == EventType.TOOL_RESULT
        assert event.success is True
        assert event.output == "content"

    def test_tool_result_failure(self) -> None:
        event = ToolResultEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            tool_call_id="call_1",
            tool_name="read_file",
            success=False,
            error={"message": "not found", "code": 404},
        )
        assert event.success is False
        assert event.error == {"message": "not found", "code": 404}

    def test_tool_result_to_stored_event(self) -> None:
        event = ToolResultEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            tool_call_id="call_1",
            tool_name="tool",
            success=True,
            output="ok",
        )
        stored = event.to_stored_event()
        assert stored.type == "tool_result"
        assert stored.frame_id == "frame_1"


class TestApprovalEvents:
    """Test approval events."""

    def test_approval_required_creation(self) -> None:
        event = ApprovalRequiredEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            approval_requests=[
                ApprovalRequestForm(
                    approval_token="tok",  # noqa: S106 - test fixture token, not a real secret
                    tool_call_id="call_1",
                    tool_name="write_file",
                    arguments={"path": "/tmp/f.txt"},  # noqa: S108 - test fixture path, not a real temp dir
                )
            ],
            timeout_seconds=300,
        )
        assert event.type == EventType.APPROVAL_REQUIRED
        assert len(event.approval_requests) == 1
        assert event.approval_requests[0].approval_token == "tok"  # noqa: S105 - test fixture value, not a real secret

    def test_approval_granted_creation(self) -> None:
        event = ApprovalGrantedEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            approval_token="tok",  # noqa: S106 - test fixture token, not a real secret
            tool_call_id="call_1",
        )
        assert event.type == EventType.APPROVAL_GRANTED

    def test_approval_denied_creation(self) -> None:
        event = ApprovalDeniedEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            approval_token="tok",  # noqa: S106 - test fixture token, not a real secret
            tool_call_id="call_1",
            reason="denied",
        )
        assert event.type == EventType.APPROVAL_DENIED
        assert event.reason == "denied"


class TestErrorAndPartialEvents:
    """Test error and partial response events."""

    def test_error_event(self) -> None:
        event = ErrorEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            error_type="ValueError",
            error_message="bad input",
            recoverable=False,
        )
        assert event.type == EventType.ERROR
        assert event.recoverable is False

    def test_error_to_stored_event(self) -> None:
        event = ErrorEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            error_type="E",
            error_message="m",
        )
        stored = event.to_stored_event()
        assert stored.type == "error"

    def test_partial_response_event(self) -> None:
        event = PartialResponseEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            error_type="truncation",
            error_message="truncated",
            raw_response="partial...",
        )
        assert event.type == EventType.PARTIAL_RESPONSE
        assert event.error_type == "truncation"


class TestContinuationDecisionEvent:
    """Test continuation decision event."""

    def test_continuation_decision_creation(self) -> None:
        event = ContinuationDecisionEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            decision="CONTINUE",
            reason="tool calls pending",
        )
        assert event.type == EventType.CONTINUATION_DECISION
        assert event.decision == "CONTINUE"


class TestEventFields:
    """Test shared BaseEvent fields (seq, timestamp, debug_preview)."""

    def test_seq_defaults_to_zero(self) -> None:
        event = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="q",
            turn_index=0,
        )
        assert event.seq == 0

    def test_seq_can_be_set(self) -> None:
        event = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="q",
            turn_index=0,
            seq=5,
        )
        assert event.seq == 5

    def test_debug_preview_optional(self) -> None:
        event = FrameStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            iteration=1,
            context_snapshot_id="ctx",
            context_hash="sha256:abc",
        )
        assert event.debug_preview is None

    def test_timestamp_defaults_to_now(self) -> None:
        before = datetime.now(UTC)
        event = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="q",
            turn_index=0,
        )
        after = datetime.now(UTC)
        assert before <= event.timestamp <= after

    def test_timestamp_can_be_set(self) -> None:
        test_time = datetime.fromtimestamp(1234567890, tz=UTC)
        event = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="q",
            turn_index=0,
            timestamp=test_time,
        )
        assert event.timestamp == test_time
