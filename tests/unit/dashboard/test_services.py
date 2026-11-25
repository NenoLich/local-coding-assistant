"""
Unit tests for the historical (store-backed) dashboard services.

The execution store is mocked with an AsyncMock; these tests seal the service
layer's mapping of stored rows to dashboard models, pagination, filtering, and
error handling.
"""

from datetime import UTC, datetime

import pytest

from local_coding_assistant.dashboard.exceptions import (
    NotFoundError,
    StoreUnavailableError,
)
from local_coding_assistant.dashboard.services.debug_service import DebugService
from local_coding_assistant.dashboard.services.event_service import EventService
from local_coding_assistant.dashboard.services.frame_service import (
    FrameService,
    _preview_for_event,
)
from local_coding_assistant.dashboard.services.message_service import MessageService
from local_coding_assistant.dashboard.services.session_service import SessionService
from local_coding_assistant.dashboard.services.turn_service import TurnService
from local_coding_assistant.storage.contracts import (
    MissingRefs,
    StoredContextSnapshot,
    StoredFrame,
    StoredSession,
    StoredTurn,
)


def _ms() -> int:
    return int(datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)


def _session(
    session_id="session-1", status="idle", model_name="gpt-4"
) -> StoredSession:
    return StoredSession(
        id=session_id,
        user_input="hello",
        status=status,
        started_at_ms=_ms(),
        updated_at_ms=_ms(),
        agent_mode="frame_agent",
        model_name=model_name,
        total_input_tokens=100,
        total_output_tokens=50,
        total_turns=2,
        total_frames=3,
    )


def _turn(
    turn_id="turn-1", session_id="session-1", turn_index=0, status="completed"
) -> StoredTurn:
    return StoredTurn(
        id=turn_id,
        session_id=session_id,
        trace_id="trace-1",
        turn_index=turn_index,
        user_input="hello",
        config_snapshot_json='{"context_config": {}}',
        status=status,
        started_at_ms=_ms(),
        finished_at_ms=_ms() + 5000,
        context_snapshot_id="snap-1",
        total_input_tokens=100,
        total_output_tokens=50,
        total_frames=2,
        final_answer="done",
    )


def _frame(
    frame_id="frame-1", turn_id="turn-1", iteration=0, status="completed"
) -> StoredFrame:
    return StoredFrame(
        id=frame_id,
        session_id="session-1",
        turn_id=turn_id,
        trace_id="trace-1",
        iteration=iteration,
        status=status,
        started_at_ms=_ms(),
        finished_at_ms=_ms() + 2000,
        duration_ms=2000,
        model_name="gpt-4",
        finish_reason="stop",
        error_message=None,
    )


def _snapshot(snapshot_id="snap-1") -> StoredContextSnapshot:
    return StoredContextSnapshot(
        id=snapshot_id,
        turn_id="turn-1",
        content_hash="hash-1",
        created_at_ms=_ms(),
    )


class TestEventService:
    @pytest.mark.asyncio
    async def test_get_events_by_turn(self, mock_store):
        mock_store.get_events_by_turn.return_value = [
            {
                "id": 101,
                "seq": 1,
                "type": "turn_start",
                "timestamp": _ms(),
                "frame_id": None,
                "data_json": '{"user_input": "first"}',
            },
            {
                "id": 202,
                "seq": 2,
                "type": "frame_start",
                "timestamp": _ms(),
                "frame_id": "frame-1",
                "data_json": '{"iteration": 1}',
            },
        ]
        service = EventService(mock_store)

        entries = await service.get_events_by_turn("turn-1")

        assert [e.event_id for e in entries] == [101, 202]
        assert [e.seq for e in entries] == [1, 2]
        assert entries[0].type == "turn_start"
        assert entries[0].preview == {"user_input": "first"}
        assert entries[0].payload == {"user_input": "first"}
        assert entries[0].has_payload is True

    @pytest.mark.asyncio
    async def test_get_events_by_turn_type_filter(self, mock_store):
        mock_store.get_events_by_turn.return_value = [
            {
                "id": 1,
                "seq": 1,
                "type": "turn_start",
                "timestamp": _ms(),
                "frame_id": None,
                "data_json": "{}",
            },
            {
                "id": 2,
                "seq": 2,
                "type": "llm_complete",
                "timestamp": _ms(),
                "frame_id": "f",
                "data_json": "{}",
            },
        ]
        service = EventService(mock_store)

        entries = await service.get_events_by_turn(
            "turn-1", event_type=["llm_complete"]
        )
        assert len(entries) == 1
        assert entries[0].type == "llm_complete"

    @pytest.mark.asyncio
    async def test_get_event_payload_uses_unique_id(self, mock_store):
        mock_store.get_events_by_turn.return_value = [
            {
                "id": 101,
                "seq": 1,
                "type": "turn_start",
                "timestamp": _ms(),
                "frame_id": None,
                "data_json": '{"a": 1}',
            },
            {
                "id": 202,
                "seq": 1,
                "type": "frame_start",
                "timestamp": _ms(),
                "frame_id": "f",
                "data_json": '{"b": 2}',
            },
        ]
        service = EventService(mock_store)

        assert await service.get_event_payload("turn-1", 202) == {"b": 2}
        assert await service.get_event_payload("turn-1", 999) is None

    @pytest.mark.asyncio
    async def test_get_events_by_frame(self, mock_store):
        mock_store.get_frame_events.return_value = [
            {
                "id": 1,
                "seq": 1,
                "type": "frame_start",
                "timestamp": _ms(),
                "frame_id": "f",
                "data_json": '{"iteration": 1}',
            },
        ]
        service = EventService(mock_store)

        entries = await service.get_events_by_frame("f")
        assert len(entries) == 1
        assert entries[0].frame_id == "f"

    @pytest.mark.asyncio
    async def test_store_unavailable_raises(self):
        service = EventService(None)
        with pytest.raises(StoreUnavailableError):
            await service.get_events_by_turn("turn-1")


class TestSessionService:
    @pytest.mark.asyncio
    async def test_list_sessions_pagination(self, mock_store):
        mock_store.list_sessions.return_value = [_session(f"s{i}") for i in range(25)]
        service = SessionService(mock_store)

        result = await service.list_sessions(limit=10, offset=0)
        assert result["total"] == 25
        assert result["limit"] == 10
        assert result["offset"] == 0
        assert result["has_next"] is True
        assert result["has_prev"] is False
        assert len(result["items"]) == 10

    @pytest.mark.asyncio
    async def test_list_sessions_status_filter(self, mock_store):
        mock_store.list_sessions.return_value = [
            _session(status="running"),
            _session(status="idle"),
        ]
        service = SessionService(mock_store)

        await service.list_sessions(status="running")
        mock_store.list_sessions.assert_called_once_with(limit=10000, status="running")

    @pytest.mark.asyncio
    async def test_get_session_detail_not_found(self, mock_store):
        mock_store.get_session.return_value = None
        service = SessionService(mock_store)

        with pytest.raises(NotFoundError):
            await service.get_session_detail("nope")


class TestTurnService:
    @pytest.mark.asyncio
    async def test_get_turn_detail_not_found(self, mock_store):
        mock_store.get_turn.return_value = None
        service = TurnService(mock_store)

        with pytest.raises(NotFoundError):
            await service.get_turn_detail("nope")

    @pytest.mark.asyncio
    async def test_get_turn_detail_builds_frames(self, mock_store):
        mock_store.get_turn.return_value = _turn()
        mock_store.get_frame_ids_by_turn.return_value = ["frame-1"]
        mock_store.get_frame.return_value = _frame()
        mock_store.get_frame_events.return_value = [
            {
                "id": 1,
                "seq": 1,
                "type": "llm_complete",
                "timestamp": _ms(),
                "frame_id": "frame-1",
                "data_json": '{"input_tokens": 10, "output_tokens": 5, "cached_tokens": 2}',
            },
        ]
        mock_store.get_all_messages.return_value = []
        mock_store.get_context_snapshot.return_value = None
        service = TurnService(mock_store)

        detail = await service.get_turn_detail("turn-1")
        assert detail.turn.turn_id == "turn-1"
        assert len(detail.frames) == 1
        assert detail.frames[0].input_tokens == 10
        assert detail.messages == []


class TestFrameService:
    @pytest.mark.asyncio
    async def test_get_frame_detail_not_found(self, mock_store):
        mock_store.get_frame.return_value = None
        service = FrameService(mock_store)

        with pytest.raises(NotFoundError):
            await service.get_frame_detail("nope")

    @pytest.mark.asyncio
    async def test_get_frame_detail_builds_events_and_totals(self, mock_store):
        mock_store.get_frame.return_value = _frame()
        mock_store.get_frame_events.return_value = [
            {
                "id": 1,
                "seq": 1,
                "type": "llm_complete",
                "timestamp": _ms(),
                "frame_id": "frame-1",
                "data_json": '{"input_tokens": 10, "output_tokens": 5, "cached_tokens": 2}',
            },
        ]
        service = FrameService(mock_store)

        detail = await service.get_frame_detail("frame-1")
        assert detail.frame.frame_id == "frame-1"
        assert len(detail.events) == 1
        assert detail.frame.input_tokens == 10
        assert detail.frame.output_tokens == 5
        assert detail.frame.cache_tokens == 2

    @pytest.mark.asyncio
    async def test_list_frames_by_turn(self, mock_store):
        mock_store.get_frame_ids_by_turn.return_value = ["frame-1"]
        mock_store.get_frame.return_value = _frame()
        mock_store.get_frame_events.return_value = []
        service = FrameService(mock_store)

        frames = await service.list_frames_by_turn("turn-1")
        assert len(frames) == 1
        assert frames[0].iteration == 0


class TestMessageService:
    @pytest.mark.asyncio
    async def test_list_messages_by_session_turn_filter(self, mock_store):
        turn_0 = _turn(turn_id="turn-1", turn_index=0)
        turn_1 = _turn(turn_id="turn-2", turn_index=1)
        mock_store.get_all_messages.return_value = [
            {
                "seq": 1,
                "role": "user",
                "content": "hi",
                "turn_id": "turn-1",
                "frame_id": None,
                "created_at": _ms(),
                "metadata_json": None,
            },
            {
                "seq": 2,
                "role": "assistant",
                "content": "hello",
                "turn_id": "turn-2",
                "frame_id": None,
                "created_at": _ms(),
                "metadata_json": None,
            },
        ]
        mock_store.list_turns_by_session.return_value = [turn_0, turn_1]
        service = MessageService(mock_store)

        messages = await service.list_messages_by_session(
            "session-1", turn_from=1, turn_to=1
        )
        assert len(messages) == 1
        assert messages[0].seq == 2

    @pytest.mark.asyncio
    async def test_get_message_detail_not_found(self, mock_store):
        mock_store.get_all_messages.return_value = []
        mock_store.list_turns_by_session.return_value = []
        service = MessageService(mock_store)

        with pytest.raises(NotFoundError):
            await service.get_message_detail("session-1", 999)


class TestDebugService:
    @pytest.mark.asyncio
    async def test_resolve_snapshot_not_found(self, mock_store):
        mock_store.get_stored_context_snapshot.return_value = None
        service = DebugService(mock_store)

        with pytest.raises(NotFoundError):
            await service.resolve_snapshot("nope")

    @pytest.mark.asyncio
    async def test_check_missing_refs(self, mock_store):
        snap = _snapshot()
        snap.tool_set_ref = '{"tool_ids": [1, 2]}'
        snap.template_set_ref = '{"template_ids": [3]}'
        snap.repo_snapshot_ref = '{"repo_snapshot_hash": "abc"}'
        snap.memory_snapshot_ref = '{"memory_ids": []}'
        mock_store.get_stored_context_snapshot.return_value = snap
        mock_store.check_context_refs.return_value = MissingRefs(tool_ids=[2])
        service = DebugService(mock_store)

        missing = await service.check_missing_refs("snap-1")
        assert missing.tool_ids == [2]
        assert missing.is_empty is False

    @pytest.mark.asyncio
    async def test_check_missing_refs_not_found(self, mock_store):
        mock_store.get_stored_context_snapshot.return_value = None
        service = DebugService(mock_store)

        with pytest.raises(NotFoundError):
            await service.check_missing_refs("nope")


class TestPreviewForEvent:
    def test_llm_complete_preview_truncates_content(self):
        data = {
            "model": "gpt-4",
            "provider": "openai",
            "finish_reason": "stop",
            "input_tokens": 100,
            "output_tokens": 50,
            "tool_calls": [{"id": "tc-1"}],
            "content": "x" * 300,
        }
        preview = _preview_for_event("llm_complete", data)
        assert preview["model"] == "gpt-4"
        assert preview["provider"] == "openai"
        assert preview["finish_reason"] == "stop"
        assert preview["input_tokens"] == 100
        assert preview["output_tokens"] == 50
        assert preview["tool_calls"] == [{"id": "tc-1"}]
        assert preview["content"] == "x" * 200

    def test_llm_complete_empty_content_omitted(self):
        preview = _preview_for_event("llm_complete", {"model": "gpt-4", "content": ""})
        assert preview == {"model": "gpt-4"}

    def test_tool_result_preview(self):
        data = {
            "tool_name": "echo",
            "success": True,
            "execution_time_ms": 1500,
            "error": None,
        }
        preview = _preview_for_event("tool_result", data)
        assert preview["tool_name"] == "echo"
        assert preview["success"] is True
        assert preview["execution_time_ms"] == 1500
        assert preview["error"] is None

    def test_turn_complete_preview(self):
        data = {
            "status": "completed",
            "final_answer": "done",
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "total_frames": 2,
            "finish_reason": "stop",
        }
        assert _preview_for_event("turn_complete", data) == data

    def test_frame_complete_preview(self):
        data = {
            "status": "completed",
            "model_name": "gpt-4",
            "finish_reason": "stop",
            "duration_ms": 2000,
            "error_message": "boom",
        }
        assert _preview_for_event("frame_complete", data) == data

    def test_turn_start_preview(self):
        data = {
            "user_input": "hello",
            "turn_index": 0,
            "agent_mode": "frame_agent",
            "model_name": "gpt-4",
        }
        assert _preview_for_event("turn_start", data) == data

    def test_session_start_preview(self):
        data = {
            "user_input": "hello",
            "agent_mode": "frame_agent",
            "model_name": "gpt-4",
        }
        assert _preview_for_event("session_start", data) == data

    def test_error_preview(self):
        data = {
            "error_type": "runtime_error",
            "error_message": "boom",
            "recoverable": False,
        }
        assert _preview_for_event("error", data) == data

    def test_continuation_decision_preview(self):
        data = {
            "decision": "continue",
            "action": "next",
            "stop_reason": None,
            "reason": "ok",
        }
        assert _preview_for_event("continuation_decision", data) == data

    def test_approval_required_preview_counts_requests(self):
        data = {
            "approval_requests": [{"token": "a"}, {"token": "b"}, {"token": "c"}],
        }
        assert _preview_for_event("approval_required", data) == {"approval_requests": 3}

    def test_approval_required_no_requests(self):
        assert _preview_for_event("approval_required", {}) == {"approval_requests": 0}

    def test_approval_granted_preview(self):
        data = {"approval_token": "tok", "tool_call_id": "tc-1"}
        preview = _preview_for_event("approval_granted", data)
        assert preview["approval_token"] == "tok"  # noqa: S105 - test fixture value
        assert preview["tool_call_id"] == "tc-1"

    def test_approval_denied_preview(self):
        data = {
            "approval_token": "tok",
            "tool_call_id": "tc-1",
            "reason": "user denied",
        }
        preview = _preview_for_event("approval_denied", data)
        assert preview["approval_token"] == "tok"  # noqa: S105 - test fixture value
        assert preview["tool_call_id"] == "tc-1"
        assert preview["reason"] == "user denied"

    def test_unknown_event_type_returns_empty(self):
        assert _preview_for_event("whatever", {"foo": "bar"}) == {}
        assert _preview_for_event("", {}) == {}

    def test_missing_keys_are_omitted(self):
        assert _preview_for_event("llm_complete", {"model": "gpt-4"}) == {
            "model": "gpt-4"
        }
