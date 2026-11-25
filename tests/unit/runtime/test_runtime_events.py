"""Tests for new runtime event contracts (runtime/contracts/events.py)."""

from datetime import UTC, datetime

from local_coding_assistant.runtime.events.contracts import (
    ApprovalDeniedEvent,
    ApprovalGrantedEvent,
    ApprovalRequiredEvent,
    ContinuationDecisionEvent,
    ErrorEvent,
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


class TestEventType:
    """Test EventType enum."""

    def test_all_event_types_defined(self):
        """Test that all expected event types are defined."""
        expected_types = {
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

        actual_types = {e.name for e in EventType}
        assert actual_types == expected_types

    def test_event_type_values(self):
        """Test that event type values match their names in lowercase."""
        for event_type in EventType:
            expected_value = event_type.name.lower()
            assert event_type.value == expected_value


class TestTurnStartEvent:
    """Test TurnStartEvent."""

    def test_turn_start_event_creation(self):
        """Test creating a TurnStartEvent."""
        event = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="test query",
            turn_index=0,
            agent_mode="frame",
            model_name="gpt-4",
        )

        assert event.type == EventType.TURN_START
        assert event.trace_id == "tr_123"
        assert event.turn_id == "turn_456"
        assert event.session_id == "sess_789"
        assert event.user_input == "test query"
        assert event.turn_index == 0
        assert event.agent_mode == "frame"
        assert event.model_name == "gpt-4"
        assert isinstance(event.timestamp, datetime)

    def test_turn_start_to_stored_event(self):
        """Test conversion to stored event."""
        event = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="test query",
            turn_index=0,
            config_snapshot_json='{"key": "value"}',
        )

        stored = event.to_stored_event()

        assert stored.trace_id == "tr_123"
        assert stored.session_id == "sess_789"
        assert stored.turn_id == "turn_456"
        assert stored.frame_id is None
        assert stored.type == "turn_start"
        assert "debug_preview" not in stored.payload_json
        assert "trace_id" not in stored.payload_json


class TestTurnCompleteEvent:
    """Test TurnCompleteEvent."""

    def test_turn_complete_event_creation(self):
        """Test creating a TurnCompleteEvent."""
        event = TurnCompleteEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            status="completed",
            final_answer="answer",
            total_input_tokens=100,
            total_output_tokens=50,
            total_frames=5,
            finish_reason="stop",
        )

        assert event.type == EventType.TURN_COMPLETE
        assert event.status == "completed"
        assert event.final_answer == "answer"
        assert event.total_input_tokens == 100
        assert event.total_output_tokens == 50
        assert event.total_frames == 5

    def test_turn_complete_to_stored_event(self):
        """Test conversion to stored event."""
        event = TurnCompleteEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            status="completed",
        )

        stored = event.to_stored_event()
        assert stored.type == "turn_complete"
        assert "debug_preview" not in stored.payload_json


class TestSessionStartEvent:
    """Test SessionStartEvent."""

    def test_session_start_event_creation(self):
        """Test creating a SessionStartEvent."""
        event = SessionStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="test query",
            agent_mode="frame",
            model_name="gpt-4",
        )

        assert event.type == EventType.SESSION_START
        assert event.session_id == "sess_789"
        assert event.agent_mode == "frame"

    def test_session_start_to_stored_event(self):
        """Test conversion to stored event."""
        event = SessionStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="test query",
            agent_mode="frame",
        )

        stored = event.to_stored_event()
        assert stored.type == "session_start"


class TestSessionResumeEvent:
    """Test SessionResumeEvent."""

    def test_session_resume_event_creation(self):
        """Test creating a SessionResumeEvent."""
        event = SessionResumeEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            resumed_from_turn_id="turn_old",
        )

        assert event.type == EventType.SESSION_RESUME
        assert event.resumed_from_turn_id == "turn_old"

    def test_session_resume_to_stored_event(self):
        """Test conversion to stored event."""
        event = SessionResumeEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
        )

        stored = event.to_stored_event()
        assert stored.type == "session_resume"


class TestFrameStartEvent:
    """Test FrameStartEvent."""

    def test_frame_start_event_creation(self):
        """Test creating a FrameStartEvent."""
        event = FrameStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            iteration=1,
            context_snapshot_id="ctx_xyz",
            context_hash="sha256:abc123",
            message_boundary={"start_seq": 0, "end_seq": 10},
            debug_preview="Frame 1 starting",
        )

        assert event.type == EventType.FRAME_START
        assert event.frame_id == "frame_789"
        assert event.iteration == 1
        assert event.context_snapshot_id == "ctx_xyz"
        assert event.context_hash == "sha256:abc123"
        assert event.debug_preview == "Frame 1 starting"

    def test_frame_start_to_stored_event(self):
        """Test conversion to stored event excludes debug_preview."""
        event = FrameStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            iteration=1,
            context_snapshot_id="ctx_xyz",
            context_hash="sha256:abc123",
            debug_preview="This should be excluded",
        )

        stored = event.to_stored_event()

        assert stored.type == "frame_start"
        assert "debug_preview" not in stored.payload_json
        # But context_hash should be in payload
        assert "context_hash" in stored.payload_json


class TestFrameCompleteEvent:
    """Test FrameCompleteEvent."""

    def test_frame_complete_event_creation(self):
        """Test creating a FrameCompleteEvent."""
        event = FrameCompleteEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            status="completed",
            finish_reason="stop",
            duration_ms=1000,
            total_input_tokens=100,
            total_output_tokens=50,
            model_name="gpt-4",
        )

        assert event.type == EventType.FRAME_COMPLETE
        assert event.status == "completed"
        assert event.duration_ms == 1000
        assert event.total_input_tokens == 100
        assert event.total_output_tokens == 50

    def test_frame_complete_to_stored_event(self):
        """Test conversion to stored event."""
        event = FrameCompleteEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            status="completed",
        )

        stored = event.to_stored_event()
        assert stored.type == "frame_complete"


class TestLLMEvents:
    """Test LLM events."""

    def test_llm_start_event(self):
        """Test LLMStartEvent."""
        event = LLMStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            model="gpt-4",
            prompt_tokens_estimate=1000,
        )

        assert event.type == EventType.LLM_START
        assert event.model == "gpt-4"
        assert event.prompt_tokens_estimate == 1000

    def test_llm_chunk_event(self):
        """Test LLMChunkEvent."""
        event = LLMChunkEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            content="Hello",
            reasoning="thinking",
        )

        assert event.type == EventType.LLM_CHUNK
        assert event.content == "Hello"
        assert event.reasoning == "thinking"

    def test_llm_chunk_to_stored_event_empty(self):
        """Test that LLM chunks are not persisted (empty payload)."""
        event = LLMChunkEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            content="Hello",
        )

        stored = event.to_stored_event()
        assert stored.payload_json == "{}"

    def test_llm_complete_event(self):
        """Test LLMCompleteEvent."""
        event = LLMCompleteEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            content="Full response",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            input_tokens=100,
            output_tokens=50,
            finish_reason="stop",
        )

        assert event.type == EventType.LLM_COMPLETE
        assert event.content == "Full response"
        assert event.input_tokens == 100
        assert event.output_tokens == 50

    def test_llm_complete_to_stored_event_excludes_content(self):
        """Test that LLM_COMPLETE excludes full content from stored event."""
        event = LLMCompleteEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            content="This is a long response that should not be stored",
            reasoning="And reasoning too",
            input_tokens=100,
            output_tokens=50,
        )

        stored = event.to_stored_event()
        assert stored.type == "llm_complete"
        # Parse JSON to check actual fields
        import json

        payload = json.loads(stored.payload_json)
        # Content and reasoning should be excluded
        assert "content" in payload
        assert "reasoning" in payload
        # But tokens should be included
        assert "input_tokens" in payload
        assert "output_tokens" in payload


class TestToolEvents:
    """Test tool events."""

    def test_tool_start_event(self):
        """Test ToolStartEvent."""
        event = ToolStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            tool_call_id="call_123",
            tool_name="read_file",
            arguments={"path": "/tmp/file.txt"},
            execution_mode="sandbox",
        )

        assert event.type == EventType.TOOL_START
        assert event.tool_name == "read_file"
        assert event.arguments == {"path": "/tmp/file.txt"}

    def test_tool_result_event(self):
        """Test ToolResultEvent."""
        event = ToolResultEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            tool_call_id="call_123",
            tool_name="read_file",
            success=True,
            output="file content",
            timing={"duration_ms": 100},
        )

        assert event.type == EventType.TOOL_RESULT
        assert event.success is True
        assert event.output == "file content"

    def test_tool_result_event_failure(self):
        """Test ToolResultEvent with failure."""
        event = ToolResultEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            tool_call_id="call_123",
            tool_name="read_file",
            success=False,
            error={"message": "File not found", "code": 404},
        )

        assert event.success is False
        assert event.error == {"message": "File not found", "code": 404}


class TestApprovalEvents:
    """Test approval events."""

    def test_approval_required_event(self):
        """Test ApprovalRequiredEvent."""
        from local_coding_assistant.runtime.events.contracts import ApprovalRequestForm

        event = ApprovalRequiredEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            approval_requests=[
                ApprovalRequestForm(
                    approval_token="token_123",
                    tool_call_id="call_123",
                    tool_name="write_file",
                    arguments={"path": "/tmp/file.txt"},
                )
            ],
            timeout_seconds=300,
        )

        assert event.type == EventType.APPROVAL_REQUIRED
        assert len(event.approval_requests) == 1
        assert event.approval_requests[0].approval_token == "token_123"

    def test_approval_grented_event(self):
        """Test ApprovalGrantedEvent."""
        event = ApprovalGrantedEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            approval_token="token_123",
            tool_call_id="call_123",
        )

        assert event.type == EventType.APPROVAL_GRANTED
        assert event.approval_token == "token_123"

    def test_approval_denied_event(self):
        """Test ApprovalDeniedEvent."""
        event = ApprovalDeniedEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            approval_token="token_123",
            tool_call_id="call_123",
            reason="User denied",
        )

        assert event.type == EventType.APPROVAL_DENIED
        assert event.reason == "User denied"


class TestErrorEvent:
    """Test ErrorEvent."""

    def test_error_event(self):
        """Test ErrorEvent."""
        event = ErrorEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            error_type="ValueError",
            error_message="Invalid input",
            recoverable=False,
        )

        assert event.type == EventType.ERROR
        assert event.error_type == "ValueError"
        assert event.error_message == "Invalid input"
        assert event.recoverable is False


class TestPartialResponseEvent:
    """Test PartialResponseEvent."""

    def test_partial_response_event(self):
        """Test PartialResponseEvent."""
        event = PartialResponseEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            error_type="truncation",
            error_message="Response truncated",
            raw_response="partial...",
        )

        assert event.type == EventType.PARTIAL_RESPONSE
        assert event.error_type == "truncation"


class TestContinuationDecisionEvent:
    """Test ContinuationDecisionEvent."""

    def test_continuation_decision_event(self):
        """Test ContinuationDecisionEvent."""
        event = ContinuationDecisionEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            decision="CONTINUE",
            reason="Tool calls pending",
        )

        assert event.type == EventType.CONTINUATION_DECISION
        assert event.decision == "CONTINUE"


class TestEventSequenceAssignment:
    """Test seq field assignment."""

    def test_seq_field_default(self):
        """Test that seq defaults to 0."""
        event = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="test",
            turn_index=0,
        )
        assert event.seq == 0

    def test_seq_field_can_be_set(self):
        """Test that seq can be explicitly set."""
        event = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="test",
            turn_index=0,
            seq=5,
        )
        assert event.seq == 5


class TestDebugPreview:
    """Test debug_preview field."""

    def test_debug_preview_optional(self):
        """Test that debug_preview is optional."""
        event = FrameStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            iteration=1,
            context_snapshot_id="ctx_xyz",
            context_hash="sha256:abc",
        )
        assert event.debug_preview is None

    def test_debug_preview_can_be_set(self):
        """Test that debug_preview can be set."""
        event = FrameStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            iteration=1,
            context_snapshot_id="ctx_xyz",
            context_hash="sha256:abc",
            debug_preview="Debug info",
        )
        assert event.debug_preview == "Debug info"


class TestTimestampDefault:
    """Test timestamp field."""

    def test_timestamp_defaults_to_now(self):
        """Test that timestamp defaults to current time."""
        before = datetime.now(UTC)
        event = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="test",
            turn_index=0,
        )
        after = datetime.now(UTC)

        assert before <= event.timestamp <= after

    def test_timestamp_can_be_set(self):
        """Test that timestamp can be explicitly set."""
        test_time = datetime.fromtimestamp(1234567890, tz=UTC)
        event = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="test",
            turn_index=0,
            timestamp=test_time,
        )
        assert event.timestamp == test_time
