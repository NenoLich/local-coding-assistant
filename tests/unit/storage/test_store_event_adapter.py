"""Tests for StoreEventAdapter event conversion and batching.

Covers the current adapter behavior in src/local_coding_assistant/storage/store_event_adapter.py:
- runtime event -> pending entity mutation (session/turn/snapshot/frame/message/event)
- sequence number assignment (monotonic per session)
- trace_id auto-generation
- prepare_batch() op assembly, merge semantics, completed-entity clearing
- close() final flush
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from local_coding_assistant.runtime.events.contracts import (
    ApprovalDeniedEvent,
    ApprovalGrantedEvent,
    ApprovalRequiredEvent,
    ErrorEvent,
    FrameCompleteEvent,
    FrameStartEvent,
    LLMCompleteEvent,
    SessionStartEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.storage.store_event_adapter import StoreEventAdapter

from .conftest import approval_requests  # re-exported for convenience


class TestStoreEventAdapter:
    """Test StoreEventAdapter event conversion and batching."""

    # ------------------------------------------------------------------ init

    def test_adapter_initialization(self, adapter: StoreEventAdapter) -> None:
        """Adapter initializes with empty state."""
        assert len(adapter._pending) == 0
        assert len(adapter._seq_counters) == 0
        assert adapter._current_turn_id is None
        assert adapter._current_frame_id is None

    # ------------------------------------------------------ session start

    def test_process_session_start_event(self, adapter: StoreEventAdapter) -> None:
        """SessionStartEvent creates a session upsert and queues the event."""
        event = SessionStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="hello",
            agent_mode="frame",
            model_name="gpt-4",
            metadata_json='{"key": "val"}',
        )

        adapter.process(event)

        key = f"{event.turn_id}:turn"
        assert key in adapter._pending
        pending = adapter._pending[key]

        assert pending.session_create is not None
        assert pending.session_create["op"] == "upsert_session"
        data = pending.session_create["data"]
        assert data["session_id"] == "sess_1"
        assert data["user_input"] == "hello"
        assert data["status"] == "idle"
        assert data["agent_mode"] == "frame"
        assert data["model_name"] == "gpt-4"
        assert data["metadata_json"] == '{"key": "val"}'

        assert len(pending.events) == 1
        assert pending.events[0]["op"] == "event"
        assert pending.events[0]["data"]["event_type"] == "session_start"

    # ------------------------------------------------------------------ turn

    def test_process_turn_start_event(self, adapter: StoreEventAdapter) -> None:
        """TurnStartEvent creates turn + snapshot + session update + user message."""
        event = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="test query",
            turn_index=0,
            agent_mode="frame",
            model_name="gpt-4",
            config_snapshot_json='{"model": "gpt-4"}',
            context_snapshot_json='{"id": "ctx_123", "turn_id": "turn_456", "schema_version": 1, "tool_set_ref": {}, "template_set_ref": {}, "repo_snapshot_ref": null, "memory_snapshot_ref": null, "message_boundary": null, "content_hash": "hash123", "created_at_ms": 1000}',
            context_profile_json='{"profile": 1}',
        )

        adapter.process(event)

        key = f"{event.turn_id}:turn"
        assert key in adapter._pending
        pending = adapter._pending[key]

        assert pending.turn_create is not None
        assert pending.turn_create["op"] == "upsert_turn"
        td = pending.turn_create["data"]
        assert td["turn_id"] == "turn_456"
        assert td["session_id"] == "sess_789"
        assert td["trace_id"] == "tr_123"
        assert td["turn_index"] == 0
        assert td["user_input"] == "test query"
        assert td["status"] == "running"
        assert td["config_snapshot_json"] == '{"model": "gpt-4"}'

        assert pending.snapshot_create is not None
        assert pending.snapshot_create["op"] == "upsert_context_snapshot"
        sd = pending.snapshot_create["data"]
        assert sd["snapshot_id"] == "ctx_123"
        assert sd["turn_id"] == "turn_456"
        assert sd["content_hash"] == "hash123"

        assert pending.session_update is not None
        assert pending.session_update["op"] == "upsert_session"
        assert pending.session_update["data"]["status"] == "running"
        assert (
            pending.session_update["data"]["context_profile_json"] == '{"profile": 1}'
        )

        assert pending.session_turns_delta == 1

        assert len(pending.messages) == 1
        assert pending.messages[0]["role"] == "user"
        assert pending.messages[0]["content"] == "test query"
        assert pending.messages[0]["dedupe_key"] == "user:turn_456"

        assert len(pending.events) == 1

    def test_process_turn_start_without_snapshot(
        self, adapter: StoreEventAdapter
    ) -> None:
        """TurnStartEvent with no context_snapshot_json creates no snapshot."""
        event = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="query",
            turn_index=0,
        )

        adapter.process(event)

        pending = adapter._pending[f"{event.turn_id}:turn"]
        assert pending.snapshot_create is None
        assert pending.turn_create is not None

    # ------------------------------------------------------ frame start

    def test_process_frame_start_event(self, adapter: StoreEventAdapter) -> None:
        """FrameStartEvent creates a frame (snapshot comes from TurnStartEvent)."""
        event = FrameStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            iteration=1,
            context_snapshot_id="ctx_xyz",
            context_hash="sha256:abc123",
            message_boundary={"start_seq": 0, "end_seq": 10},
            history_through_seq=24,
            prompt_hash="sha256:prompt",
        )

        adapter.process(event)

        key = f"{event.turn_id}:{event.frame_id}"
        assert key in adapter._pending
        pending = adapter._pending[key]

        assert pending.frame_create is not None
        assert pending.frame_create["op"] == "upsert_frame"
        fd = pending.frame_create["data"]
        assert fd["frame_id"] == "frame_789"
        assert fd["iteration"] == 1
        assert fd["status"] == "running_llm"
        assert fd["history_through_seq"] == 24
        assert fd["prompt_hash"] == "sha256:prompt"
        assert fd["context_snapshot_id"] == "ctx_xyz"

        assert pending.snapshot_create is None

        turn_pending = adapter._pending[f"{event.turn_id}:turn"]
        assert turn_pending.turn_frames_delta == 1
        assert turn_pending.session_frames_delta == 1

    def test_process_frame_start_second_iteration(
        self, adapter: StoreEventAdapter
    ) -> None:
        """Second frame iteration creates no context snapshot."""
        event = FrameStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            iteration=2,
            context_snapshot_id="ctx_xyz",
            context_hash="sha256:abc123",
        )

        adapter.process(event)

        pending = adapter._pending[f"{event.turn_id}:{event.frame_id}"]
        assert pending.frame_create is not None
        assert pending.snapshot_create is None

    # ------------------------------------------------------ frame complete

    def test_process_frame_complete_event(self, adapter: StoreEventAdapter) -> None:
        """FrameCompleteEvent updates frame status/timing, no tokens."""
        event = FrameCompleteEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            status="completed",
            finish_reason="stop",
            duration_ms=1000,
            total_input_tokens=30,
            total_output_tokens=20,
            model_name="gpt-4",
        )

        adapter.process(event)

        pending = adapter._pending[f"{event.turn_id}:{event.frame_id}"]
        assert pending.frame_update is not None
        assert pending.frame_update["op"] == "upsert_frame"
        assert pending.frame_update["data"]["status"] == "completed"
        assert pending.frame_update["data"]["duration_ms"] == 1000
        assert "input_tokens" not in pending.frame_update["data"]
        assert "output_tokens" not in pending.frame_update["data"]
        assert pending.frame_input_tokens_delta == 0
        assert pending.frame_output_tokens_delta == 0

    # ------------------------------------------------------ llm complete

    def test_process_llm_complete_event(self, adapter: StoreEventAdapter) -> None:
        """LLMCompleteEvent updates frame model/status, adds token deltas, assistant message."""
        event = LLMCompleteEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            content="response",
            input_tokens=90,
            output_tokens=60,
            model="gpt-4",
        )

        adapter.process(event)

        pending = adapter._pending[f"{event.turn_id}:{event.frame_id}"]
        assert pending.frame_update is not None
        assert pending.frame_update["data"]["model_name"] == "gpt-4"
        assert pending.frame_update["data"]["status"] == "running_tools"
        assert pending.frame_input_tokens_delta == 90
        assert pending.frame_output_tokens_delta == 60

        turn_pending = adapter._pending[f"{event.turn_id}:turn"]
        assert turn_pending.turn_input_tokens_delta == 90
        assert turn_pending.turn_output_tokens_delta == 60
        assert turn_pending.session_input_tokens_delta == 90
        assert turn_pending.session_output_tokens_delta == 60

        assert len(turn_pending.messages) == 1
        assert turn_pending.messages[0]["role"] == "assistant"
        assert turn_pending.messages[0]["content"] == "response"
        assert turn_pending.messages[0]["dedupe_key"] == "assistant:frame_789"

    def test_process_llm_complete_no_content_no_calls(
        self, adapter: StoreEventAdapter
    ) -> None:
        """LLMCompleteEvent with no content and no tool calls projects nothing."""
        event = LLMCompleteEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            content="",
            input_tokens=0,
            output_tokens=0,
        )

        adapter.process(event)

        pending = adapter._pending[f"{event.turn_id}:{event.frame_id}"]
        assert pending.frame_update is not None
        assert len(pending.messages) == 0

    # ------------------------------------------------------ tool result

    def test_process_tool_result_event_success(
        self, adapter: StoreEventAdapter
    ) -> None:
        """Successful ToolResultEvent projects a tool message, no frame update."""
        event = ToolResultEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            tool_call_id="call_123",
            tool_name="read_file",
            success=True,
            output="file contents",
        )

        adapter.process(event)

        turn_pending = adapter._pending[f"{event.turn_id}:turn"]
        assert turn_pending.frame_update is None
        assert len(turn_pending.messages) == 1
        msg = turn_pending.messages[0]
        assert msg["role"] == "tool"
        assert msg["content"] == "file contents"
        assert msg["tool_call_id"] == "call_123"
        assert msg["dedupe_key"] == "tool:turn_456:call_123"

    def test_process_tool_result_event_failure(
        self, adapter: StoreEventAdapter
    ) -> None:
        """Failed ToolResultEvent updates frame status and error message (current adapter behavior)."""
        event = ToolResultEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            tool_call_id="call_123",
            tool_name="read_file",
            success=False,
            error={"message": "File not found"},
        )

        adapter.process(event)

        pending = adapter._pending[f"{event.turn_id}:{event.frame_id}"]
        assert pending.frame_update is not None
        assert pending.frame_update["data"]["status"] == "running_tools"
        assert pending.frame_update["data"]["error_message"] == "File not found"

    def test_process_tool_result_event_file_baselines(
        self, adapter: StoreEventAdapter
    ) -> None:
        """ToolResultEvent queues file baseline upserts for read/snapshot/agent_create sources."""
        event = ToolResultEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            tool_call_id="call_123",
            tool_name="read_file",
            success=True,
            workspace_id="ws_1",
            file_observations=[
                {
                    "path": "a.py",
                    "hash": "h1",
                    "size": 10,
                    "mtime": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                    "source": {"value": "read"},
                },
                {
                    "path": "b.py",
                    "hash": "h2",
                    "size": 20,
                    "mtime": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                    "source": {"value": "snapshot"},
                },
                {
                    "path": "c.py",
                    "hash": "h3",
                    "size": 30,
                    "mtime": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                    "source": {"value": "write"},
                },
            ],
        )

        adapter.process(event)

        pending = adapter._pending[f"{event.turn_id}:{event.frame_id}"]
        # Only read/snapshot/agent_create sources are persisted (write skipped).
        assert len(pending.baseline_upserts) == 2
        for b in pending.baseline_upserts:
            assert b["op"] == "upsert_file_baseline"
            assert b["data"]["workspace_id"] == "ws_1"
            assert b["data"]["turn_id"] == "turn_456"
            assert b["data"]["frame_id"] == "frame_789"
            assert b["data"]["session_id"] == "sess_abc"

    def test_process_tool_result_event_baseline_requires_workspace_id(
        self, adapter: StoreEventAdapter
    ) -> None:
        """File baseline upserts require a workspace_id."""
        event = ToolResultEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            tool_call_id="call_123",
            tool_name="read_file",
            success=True,
            workspace_id=None,
            file_observations=[
                {
                    "path": "a.py",
                    "hash": "h1",
                    "size": 10,
                    "mtime": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                    "source": {"value": "read"},
                },
            ],
        )

        with pytest.raises(ValueError, match="workspace_id is required"):
            adapter.process(event)

    # ------------------------------------------------------ approval

    def test_process_approval_required_event(self, adapter: StoreEventAdapter) -> None:
        """ApprovalRequiredEvent updates frame to running_tools (current adapter behavior)."""
        event = ApprovalRequiredEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            approval_requests=approval_requests(count=1),
        )

        adapter.process(event)

        pending = adapter._pending[f"{event.turn_id}:{event.frame_id}"]
        assert pending.frame_update is not None
        assert pending.frame_update["data"]["status"] == "running_tools"

    def test_process_approval_granted_event(self, adapter: StoreEventAdapter) -> None:
        """ApprovalGrantedEvent resumes frame to running_tools, clears pending_call_id."""
        event = ApprovalGrantedEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            approval_token="token_123",  # noqa: S106 - test fixture token, not a real secret
            tool_call_id="call_123",
        )

        adapter.process(event)

        pending = adapter._pending[f"{event.turn_id}:{event.frame_id}"]
        assert pending.frame_update["data"]["status"] == "running_tools"
        assert pending.frame_update["data"]["pending_call_id"] is None

    def test_process_approval_denied_event(self, adapter: StoreEventAdapter) -> None:
        """ApprovalDeniedEvent resumes frame to running_tools, clears pending_call_id."""
        event = ApprovalDeniedEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            approval_token="token_123",  # noqa: S106 - test fixture token, not a real secret
            tool_call_id="call_123",
            reason="user denied",
        )

        adapter.process(event)

        pending = adapter._pending[f"{event.turn_id}:{event.frame_id}"]
        assert pending.frame_update["data"]["status"] == "running_tools"
        assert pending.frame_update["data"]["pending_call_id"] is None

    # ------------------------------------------------------ error

    def test_process_error_event(self, adapter: StoreEventAdapter) -> None:
        """ErrorEvent marks the frame failed with finished_at and error message."""
        event = ErrorEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            error_type="ValueError",
            error_message="Invalid input",
        )

        adapter.process(event)

        pending = adapter._pending[f"{event.turn_id}:{event.frame_id}"]
        assert pending.frame_update is not None
        assert pending.frame_update["data"]["status"] == "failed"
        assert pending.frame_update["data"]["error_message"] == "Invalid input"
        assert pending.frame_update["data"]["finished_at_ms"] > 0

    # ------------------------------------------------------ turn complete

    def test_process_turn_complete_event(self, adapter: StoreEventAdapter) -> None:
        """TurnCompleteEvent updates turn status/finished_at and session to idle."""
        event = TurnCompleteEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            status="completed",
            total_input_tokens=60,
            total_output_tokens=40,
            total_frames=5,
        )

        adapter.process(event)

        key = f"{event.turn_id}:turn"
        pending = adapter._pending[key]

        assert pending.turn_update is not None
        assert pending.turn_update["op"] == "upsert_turn"
        assert pending.turn_update["data"]["status"] == "completed"
        assert pending.turn_update["data"]["finished_at_ms"] > 0
        assert pending.turn_update["data"]["final_answer"] is None
        assert "total_input_tokens" not in pending.turn_update["data"]
        assert "total_output_tokens" not in pending.turn_update["data"]
        assert "total_frames" not in pending.turn_update["data"]

        assert pending.session_update is not None
        assert pending.session_update["data"]["status"] == "idle"

        assert pending.turn_input_tokens_delta == 0
        assert pending.turn_output_tokens_delta == 0
        assert pending.turn_frames_delta == 0
        assert pending.session_input_tokens_delta == 0
        assert pending.session_output_tokens_delta == 0
        assert pending.session_frames_delta == 0
        assert pending.session_turns_delta == 0

    def test_process_turn_complete_event_with_final_answer(
        self, adapter: StoreEventAdapter
    ) -> None:
        """TurnCompleteEvent persists final_answer in turn_update data."""
        event = TurnCompleteEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            status="completed",
            final_answer="The answer is 42",
            total_input_tokens=60,
            total_output_tokens=40,
            total_frames=5,
        )

        adapter.process(event)

        key = f"{event.turn_id}:turn"
        pending = adapter._pending[key]

        assert pending.turn_update is not None
        assert pending.turn_update["op"] == "upsert_turn"
        assert pending.turn_update["data"]["final_answer"] == "The answer is 42"

    # ------------------------------------------------------ sequences

    def test_sequence_number_assignment(self, adapter: StoreEventAdapter) -> None:
        """Sequence numbers increment per session and never reset per frame."""
        turn_start = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="query",
            turn_index=0,
        )
        frame1 = FrameStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_1",
            session_id="sess_789",
            iteration=1,
            context_snapshot_id="ctx_1",
            context_hash="sha256:abc",
        )
        frame2 = FrameStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_2",
            session_id="sess_789",
            iteration=2,
            context_snapshot_id="ctx_1",
            context_hash="sha256:abc",
        )

        adapter.process(turn_start)
        assert turn_start.seq == 1

        adapter.process(frame1)
        assert frame1.seq == 2

        adapter.process(frame2)
        assert frame2.seq == 3

    # ------------------------------------------------------ trace id

    def test_trace_id_generation(self, adapter: StoreEventAdapter) -> None:
        """A missing trace_id is auto-generated."""
        event = TurnStartEvent(
            trace_id="",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="query",
            turn_index=0,
        )

        adapter.process(event)

        assert event.trace_id.startswith("tr_")
        assert len(event.trace_id) > 3

    # ------------------------------------------------------ prepare_batch

    def test_prepare_batch_empty(self, adapter: StoreEventAdapter) -> None:
        """prepare_batch returns an empty list when nothing was processed."""
        assert adapter.prepare_batch() == []

    def test_prepare_batch_with_session_start(self, adapter: StoreEventAdapter) -> None:
        """prepare_batch emits session upsert + event for a SessionStartEvent."""
        event = SessionStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="query",
            agent_mode="frame",
        )

        adapter.process(event)
        batch = adapter.prepare_batch()

        ops = [b["op"] for b in batch]
        assert "upsert_session" in ops
        assert "event" in ops

        event_op = next(b for b in batch if b["op"] == "event")
        assert event_op["data"]["event_type"] == "session_start"
        assert event_op["data"]["session_id"] == "sess_789"
        assert event_op["data"]["turn_id"] == "turn_456"
        assert event_op["data"]["frame_id"] is None

    def test_prepare_batch_with_turn_start(self, adapter: StoreEventAdapter) -> None:
        """prepare_batch assembles turn start ops in the expected order/count."""
        event = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="query",
            turn_index=0,
            context_snapshot_json='{"id": "ctx_123", "turn_id": "turn_456", "schema_version": 1, "tool_set_ref": {}, "template_set_ref": {}, "repo_snapshot_ref": null, "memory_snapshot_ref": null, "message_boundary": null, "content_hash": "hash123", "created_at_ms": 1000}',
        )

        adapter.process(event)
        batch = adapter.prepare_batch()

        assert len(batch) == 6

        ops = [b["op"] for b in batch]
        assert "upsert_session" in ops
        assert "upsert_turn" in ops
        assert "upsert_context_snapshot" in ops
        assert "increment_session_totals" in ops
        assert "message" in ops
        assert "event" in ops

        message_op = next(b for b in batch if b["op"] == "message")
        assert message_op["data"]["role"] == "user"
        assert message_op["data"]["content"] == "query"
        assert message_op["data"]["dedupe_key"] == "user:turn_456"

        session_increment = next(
            b for b in batch if b["op"] == "increment_session_totals"
        )
        assert session_increment["data"] == {
            "session_id": event.session_id,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_turns": 1,
            "total_frames": 0,
        }

    def test_prepare_batch_frame_lifecycle(self, adapter: StoreEventAdapter) -> None:
        """prepare_batch covers a full turn/start/frame/frame-complete lifecycle."""
        turn_start = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_abc",
            user_input="query",
            turn_index=0,
            context_snapshot_json='{"id": "ctx_xyz", "turn_id": "turn_456", "schema_version": 1, "tool_set_ref": {}, "template_set_ref": {}, "repo_snapshot_ref": null, "memory_snapshot_ref": null, "message_boundary": null, "content_hash": "sha256:abc", "created_at_ms": 1000}',
        )
        start_event = FrameStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            iteration=1,
            context_snapshot_id="ctx_xyz",
            context_hash="sha256:abc",
        )
        complete_event = FrameCompleteEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            status="completed",
            duration_ms=1000,
        )

        adapter.process(turn_start)
        adapter.process(start_event)
        adapter.process(complete_event)
        batch = adapter.prepare_batch()

        assert len(batch) >= 3

        ops = [b["op"] for b in batch]
        assert "upsert_context_snapshot" in ops
        assert "upsert_frame" in ops
        assert "upsert_turn" in ops

        # Completed frame key is cleared after prepare_batch.
        assert f"{start_event.turn_id}:{start_event.frame_id}" not in adapter._pending

    def test_prepare_batch_orders_turn_keys_before_frame_keys(
        self, adapter: StoreEventAdapter
    ) -> None:
        """Turn keys are processed before frame keys in prepare_batch."""
        turn_start = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_abc",
            user_input="query",
            turn_index=0,
        )
        frame_start = FrameStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_789",
            session_id="sess_abc",
            iteration=1,
            context_snapshot_id="ctx_xyz",
            context_hash="sha256:abc",
        )

        adapter.process(turn_start)
        adapter.process(frame_start)
        batch = adapter.prepare_batch()

        turn_ops = [b for b in batch if b["op"] in ("upsert_turn", "upsert_session")]
        frame_ops = [b for b in batch if b["op"] == "upsert_frame"]
        assert turn_ops
        assert frame_ops
        assert batch.index(turn_ops[0]) < batch.index(frame_ops[0])

    def test_prepare_batch_clears_completed_entities(
        self, adapter: StoreEventAdapter
    ) -> None:
        """prepare_batch clears turn keys that have a finished turn_update."""
        event = TurnCompleteEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            status="completed",
        )

        adapter.process(event)
        adapter.prepare_batch()

        assert f"{event.turn_id}:turn" not in adapter._pending

    def test_prepare_batch_keeps_pending_entities(
        self, adapter: StoreEventAdapter
    ) -> None:
        """Incomplete entities remain pending after prepare_batch."""
        event = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="query",
            turn_index=0,
        )

        adapter.process(event)
        adapter.prepare_batch()

        assert f"{event.turn_id}:turn" in adapter._pending

    # ------------------------------------------------------ merge_data

    def test_merge_data_create_only(self) -> None:
        create = {"op": "upsert_session", "data": {"id": "123", "status": "running"}}
        merged = StoreEventAdapter._merge_data(create, None)
        assert merged["id"] == "123"
        assert merged["status"] == "running"

    def test_merge_data_update_only(self) -> None:
        update = {"op": "upsert_session", "data": {"id": "123", "status": "completed"}}
        merged = StoreEventAdapter._merge_data(None, update)
        assert merged["id"] == "123"
        assert merged["status"] == "completed"

    def test_merge_data_both(self) -> None:
        create = {
            "op": "upsert_session",
            "data": {"id": "123", "status": "running", "started_at_ms": 1000},
        }
        update = {
            "op": "upsert_session",
            "data": {"status": "completed", "finished_at_ms": 2000},
        }
        merged = StoreEventAdapter._merge_data(create, update)
        assert merged["id"] == "123"
        assert merged["status"] == "completed"
        assert merged["started_at_ms"] == 1000
        assert merged["finished_at_ms"] == 2000

    def test_merge_data_required_fields_preserved(self) -> None:
        create = {
            "op": "upsert_turn",
            "data": {"turn_id": "t1", "status": "running", "started_at_ms": 1000},
        }
        update = {
            "op": "upsert_turn",
            "data": {"status": "completed", "started_at_ms": 0},
        }
        merged = StoreEventAdapter._merge_data(
            create, update, required_fields={"turn_id", "status", "started_at_ms"}
        )
        assert merged["turn_id"] == "t1"
        assert merged["status"] == "completed"
        assert merged["started_at_ms"] == 1000

    # ------------------------------------------------------ close

    def test_close_flushes_remaining(self, adapter: StoreEventAdapter) -> None:
        """close prepares a final batch for remaining pending operations."""
        event = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="query",
            turn_index=0,
        )

        adapter.process(event)
        batch = adapter.close()

        assert len(batch) > 0
        assert any(b["op"] == "upsert_turn" for b in batch)

    # ------------------------------------------------------ multiple frames

    def test_multiple_frames_in_turn(self, adapter: StoreEventAdapter) -> None:
        """Multiple frames in a turn each get a frame-level entity plus a turn-level one."""
        frame1 = FrameStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_1",
            session_id="sess_abc",
            iteration=1,
            context_snapshot_id="ctx_xyz",
            context_hash="sha256:abc",
        )
        frame2 = FrameStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            frame_id="frame_2",
            session_id="sess_abc",
            iteration=2,
            context_snapshot_id="ctx_xyz",
            context_hash="sha256:abc",
        )

        adapter.process(frame1)
        adapter.process(frame2)

        assert f"{frame1.turn_id}:{frame1.frame_id}" in adapter._pending
        assert f"{frame2.turn_id}:{frame2.frame_id}" in adapter._pending
        assert f"{frame1.turn_id}:turn" in adapter._pending

    # ------------------------------------------------------ event conversion

    def test_event_to_stored_conversion(self, adapter: StoreEventAdapter) -> None:
        """Events are queued in stored format with trace/session/turn/frame/seq/type."""
        event = TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="query",
            turn_index=0,
        )

        adapter.process(event)
        batch = adapter.prepare_batch()

        event_ops = [op for op in batch if op["op"] == "event"]
        assert len(event_ops) == 1

        data = event_ops[0]["data"]
        assert data["trace_id"] == "tr_123"
        assert data["session_id"] == "sess_789"
        assert data["turn_id"] == "turn_456"
        assert data["frame_id"] is None
        assert data["event_type"] == "turn_start"
        assert data["seq"] == 1
        assert "data_json" in data
        assert "timestamp_ms" in data

    # ------------------------------------------------------ roundtrip

    def test_roundtrip_batch_matches_expected_entities(
        self, adapter: StoreEventAdapter
    ) -> None:
        """Process a full turn flow and verify prepare_batch ops match expected entities."""
        turn_start = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="build a thing",
            turn_index=0,
            agent_mode="frame",
            context_snapshot_json='{"id": "snap_1", "turn_id": "turn_1", "schema_version": 1, "tool_set_ref": {}, "template_set_ref": {}, "repo_snapshot_ref": null, "memory_snapshot_ref": null, "message_boundary": null, "content_hash": "chash", "created_at_ms": 1000}',
        )
        frame_start = FrameStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            iteration=1,
            context_snapshot_id="snap_1",
            context_hash="sha256:x",
        )
        llm_complete = LLMCompleteEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            content="here it is",
            input_tokens=25,
            output_tokens=17,
            model="gpt-4",
        )
        tool_result = ToolResultEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            tool_call_id="call_1",
            tool_name="read_file",
            success=True,
            output="data",
        )
        frame_complete = FrameCompleteEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            status="completed",
            finish_reason="stop",
            duration_ms=500,
        )
        turn_complete = TurnCompleteEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            status="completed",
        )

        for ev in (
            turn_start,
            frame_start,
            llm_complete,
            tool_result,
            frame_complete,
            turn_complete,
        ):
            adapter.process(ev)

        batch = adapter.prepare_batch()

        # Session upsert (turn start sets running), turn upsert, snapshot upsert.
        assert any(b["op"] == "upsert_session" for b in batch)
        assert any(b["op"] == "upsert_turn" for b in batch)
        assert any(b["op"] == "upsert_context_snapshot" for b in batch)
        # Frame upsert (create + complete merged), token increment.
        assert any(b["op"] == "upsert_frame" for b in batch)
        assert any(b["op"] == "increment_frame_tokens" for b in batch)
        # Session totals (turns + tokens).
        assert any(b["op"] == "increment_session_totals" for b in batch)
        # Projected messages: user (turn start), assistant (llm complete), tool (tool result).
        messages = [b for b in batch if b["op"] == "message"]
        assert [m["data"]["role"] for m in messages] == ["user", "assistant", "tool"]
        # One stored event per runtime event. prepare_batch orders turn keys
        # before frame keys, so turn events come before frame events.
        events = [b for b in batch if b["op"] == "event"]
        assert len(events) == 6
        assert [e["data"]["event_type"] for e in events] == [
            "turn_start",
            "turn_complete",
            "frame_start",
            "llm_complete",
            "tool_result",
            "frame_complete",
        ]

        # Completed turn + frame keys cleared; nothing left pending.
        assert adapter._pending == {}
