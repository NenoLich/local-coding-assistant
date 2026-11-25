"""Unit tests for the ``serve`` command surface.

Covers ``ServerEventSubscriber.on_event``, ``list_sessions``,
``_RequestEventSubscriber.event_to_sse``, ``create_session`` and
``get_session`` from ``src/local_coding_assistant/cli/commands/serve.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from local_coding_assistant.cli.commands.serve import (
    CreateSessionRequest,
    ServerEventSubscriber,
    SessionResponse,
    _RequestEventSubscriber,
    create_session,
    get_session,
    list_sessions,
)
from local_coding_assistant.runtime.events.contracts import EventType, TurnStartEvent
from local_coding_assistant.storage.contracts import (
    StoredFrame,
    StoredSession,
    StoredTurn,
)


class _FakeTurn:
    """Minimal runtime turn object used by ``_session_response``."""

    def __init__(self) -> None:
        self.turn_id = "turn-1"
        self.trace_id = "trace-1"
        self.status = "completed"
        self.started_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        self.finished_at = datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC)

    def get_final_answer(self) -> str:
        return "All done"


class _FakeSession:
    """Minimal runtime session object returned by ``start_session``."""

    def __init__(self) -> None:
        self.id = "sess-1"

    def get_last_turn(self) -> _FakeTurn:
        return _FakeTurn()


class _FakeRuntime:
    """Stand-in for RuntimeManager used by create_session."""

    def __init__(self, session: _FakeSession) -> None:
        self.session = session
        self.captured_options = None
        self.event_pipeline = None

    def apply_run_options(self, options) -> None:
        self.captured_options = options

    async def start_session(self, user_input, trace_id, session_metadata):
        return self.session


def _make_session(**overrides) -> StoredSession:
    data = dict(
        id="sess-1",
        user_input="hello",
        status="idle",
        started_at_ms=1769265600000,
        updated_at_ms=1769265660000,
        agent_mode="frame_agent",
        model_name="mock-model",
        total_input_tokens=10,
        total_output_tokens=5,
        total_turns=2,
        total_frames=3,
        metadata_json='{"key": "value"}',
    )
    data.update(overrides)
    return StoredSession(**data)


def _make_turn(**overrides) -> StoredTurn:
    data = dict(
        id="turn-1",
        session_id="sess-1",
        trace_id="trace-1",
        turn_index=0,
        user_input="hello",
        config_snapshot_json="{}",
        status="completed",
        started_at_ms=1769265600000,
        finished_at_ms=1769265605000,
        total_input_tokens=10,
        total_output_tokens=5,
        total_frames=3,
        final_answer="All done",
    )
    data.update(overrides)
    return StoredTurn(**data)


def _make_frame(**overrides) -> StoredFrame:
    data = dict(
        id="frame-1",
        session_id="sess-1",
        turn_id="turn-1",
        trace_id="trace-1",
        iteration=1,
        status="completed",
        started_at_ms=1769265600000,
        finished_at_ms=1769265601000,
        duration_ms=1000,
        input_tokens=5,
        output_tokens=3,
        model_name="mock-model",
        finish_reason="stop",
        error_message=None,
    )
    data.update(overrides)
    return StoredFrame(**data)


class TestServerEventSubscriberOnEvent:
    """ServerEventSubscriber.on_event()"""

    def _event(self, **overrides) -> TurnStartEvent:
        data = dict(
            trace_id="trace-1",
            turn_id="turn-1",
            frame_id="frame-1",
            session_id="sess-1",
            user_input="hello world",
            turn_index=3,
            agent_mode="frame_agent",
            model_name="mock-model",
        )
        data.update(overrides)
        return TurnStartEvent(**data)

    async def test_broadcasts_to_registered_websockets(self) -> None:
        subscriber = ServerEventSubscriber()
        ws = MagicMock()
        ws.send_json = AsyncMock()
        subscriber.register_websocket("sess-1", ws)

        event = self._event()
        await subscriber.on_event(event)

        expected = {
            "type": "event",
            "data": {
                "event_type": event.type.value,
                "session_id": "sess-1",
                "turn_id": "turn-1",
                "frame_id": "frame-1",
                "timestamp": event.timestamp.isoformat(),
                "data": {
                    "session_id": "sess-1",
                    "user_input": "hello world",
                    "turn_index": 3,
                    "config_snapshot_json": None,
                    "agent_mode": "frame_agent",
                    "model_name": "mock-model",
                    "context_snapshot_id": None,
                    "context_snapshot_json": None,
                    "context_profile_json": None,
                },
            },
        }
        ws.send_json.assert_awaited_once_with(expected)

    async def test_no_session_id_returns_early(self) -> None:
        subscriber = ServerEventSubscriber()
        ws = MagicMock()
        ws.send_json = AsyncMock()
        subscriber.register_websocket("sess-1", ws)

        event = self._event()
        event.session_id = None  # type: ignore[assignment]
        await subscriber.on_event(event)

        ws.send_json.assert_not_awaited()

    async def test_unregistered_session_returns_early(self) -> None:
        subscriber = ServerEventSubscriber()
        ws = MagicMock()
        ws.send_json = AsyncMock()

        event = self._event()
        await subscriber.on_event(event)

        ws.send_json.assert_not_awaited()

    async def test_disconnected_websocket_is_unregistered(self) -> None:
        subscriber = ServerEventSubscriber()
        ws = MagicMock()
        ws.send_json = AsyncMock(side_effect=RuntimeError("gone"))
        subscriber.register_websocket("sess-1", ws)

        event = self._event()
        await subscriber.on_event(event)

        assert "sess-1" not in subscriber._websockets


class TestEventToSse:
    """_RequestEventSubscriber.event_to_sse()"""

    async def test_formats_event_as_sse_message(self) -> None:
        subscriber = _RequestEventSubscriber()
        event = TurnStartEvent(
            trace_id="trace-1",
            turn_id="turn-1",
            frame_id="frame-1",
            session_id="sess-1",
            user_input="hello",
            turn_index=0,
        )

        sse = subscriber.event_to_sse(event)

        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        payload = json.loads(sse[len("data: ") : -2])
        assert payload["type"] == "event"
        assert payload["data"]["event_type"] == EventType.TURN_START
        assert payload["data"]["session_id"] == "sess-1"
        assert payload["data"]["turn_id"] == "turn-1"
        assert payload["data"]["frame_id"] == "frame-1"
        assert payload["data"]["timestamp"] == event.timestamp.isoformat()
        assert "session_id" not in payload["data"]["data"]
        assert "trace_id" not in payload["data"]["data"]

    async def test_no_session_id_still_renders(self) -> None:
        subscriber = _RequestEventSubscriber()
        event = self._event_without_session()
        sse = subscriber.event_to_sse(event)

        assert sse.startswith("data: ")
        payload = json.loads(sse[len("data: ") : -2])
        assert payload["data"]["session_id"] is None

    def _event_without_session(self) -> TurnStartEvent:
        event = TurnStartEvent(
            trace_id="trace-1",
            turn_id="turn-1",
            frame_id=None,
            session_id="",
            user_input="hi",
            turn_index=0,
        )
        event.session_id = None  # type: ignore[assignment]
        return event


class TestListSessions:
    """list_sessions()"""

    async def test_returns_sorted_session_list(self) -> None:
        newer = _make_session(id="sess-2", updated_at_ms=1769265700000)
        older = _make_session(id="sess-1", updated_at_ms=1769265600000)
        store = MagicMock()
        store.list_sessions = AsyncMock(return_value=[older, newer])

        with patch("local_coding_assistant.cli.commands.serve._store", store):
            response = await list_sessions(limit=10, status="idle")

        store.list_sessions.assert_awaited_once_with(limit=10, status="idle")
        body = json.loads(response.body)
        assert [s["session_id"] for s in body] == ["sess-2", "sess-1"]
        assert body[0] == {
            "session_id": "sess-2",
            "user_input": "hello",
            "status": "idle",
            "started_at": "2026-01-24T14:40:00+00:00",
            "updated_at": "2026-01-24T14:41:40+00:00",
            "agent_mode": "frame_agent",
            "model_name": "mock-model",
            "input_tokens": 10,
            "output_tokens": 5,
            "total_turns": 2,
            "total_frames": 3,
        }

    async def test_default_limit_and_no_status_filter(self) -> None:
        store = MagicMock()
        store.list_sessions = AsyncMock(return_value=[])

        with patch("local_coding_assistant.cli.commands.serve._store", store):
            await list_sessions()

        assert store.list_sessions.await_count == 1
        kwargs = store.list_sessions.await_args.kwargs
        assert set(kwargs) == {"limit", "status"}

    async def test_raises_when_store_unavailable(self) -> None:
        with patch("local_coding_assistant.cli.commands.serve._store", None):
            with pytest.raises(HTTPException) as exc:
                await list_sessions()
        assert exc.value.status_code == 503


class TestCreateSession:
    """create_session()"""

    async def test_creates_session_without_stream(self) -> None:
        runtime = _FakeRuntime(_FakeSession())
        with patch(
            "local_coding_assistant.cli.commands.serve._runtime_manager", runtime
        ):
            result = await create_session(CreateSessionRequest(user_input="hello"))

        assert isinstance(result, SessionResponse)
        assert result.session_id == "sess-1"
        assert result.turn_id == "turn-1"
        assert result.trace_id == "trace-1"
        assert result.status == "completed"
        assert result.started_at == "2026-01-01T12:00:00+00:00"
        assert result.updated_at == "2026-01-01T12:00:05+00:00"
        assert result.final_answer == "All done"
        assert runtime.captured_options.agent_mode == "frame_agent"
        assert runtime.captured_options.auto_approve is None

    async def test_stream_returns_streaming_response(self) -> None:
        runtime = _FakeRuntime(_FakeSession())
        with patch(
            "local_coding_assistant.cli.commands.serve._runtime_manager", runtime
        ):
            result = await create_session(
                CreateSessionRequest(user_input="hello", stream=True)
            )

        assert isinstance(result, StreamingResponse)

    async def test_raises_when_runtime_unavailable(self) -> None:
        with patch("local_coding_assistant.cli.commands.serve._runtime_manager", None):
            with pytest.raises(HTTPException) as exc:
                await create_session(CreateSessionRequest(user_input="hello"))
        assert exc.value.status_code == 503


class TestGetSession:
    """get_session()"""

    async def test_returns_session_with_turns_and_frames(self) -> None:
        session = _make_session()
        turn = _make_turn()
        frame = _make_frame()
        store = MagicMock()
        store.get_session = AsyncMock(return_value=session)
        store.list_turns_by_session = AsyncMock(return_value=[turn])
        store.get_frame_ids_by_turn = AsyncMock(return_value=["frame-1"])
        store.get_frame = AsyncMock(return_value=frame)

        with patch("local_coding_assistant.cli.commands.serve._store", store):
            response = await get_session("sess-1")

        store.get_session.assert_awaited_once_with("sess-1")
        body = json.loads(response.body)
        assert body["session"]["id"] == "sess-1"
        assert body["session"]["status"] == "idle"
        assert body["session"]["metadata_json"] == '{"key": "value"}'
        assert len(body["turns"]) == 1
        assert body["turns"][0]["turn_id"] == "turn-1"
        assert body["turns"][0]["status"] == "completed"
        assert len(body["turns"][0]["frames"]) == 1
        assert body["turns"][0]["frames"][0]["frame_id"] == "frame-1"
        assert body["turns"][0]["frames"][0]["iteration"] == 1

    async def test_raises_404_when_session_missing(self) -> None:
        store = MagicMock()
        store.get_session = AsyncMock(return_value=None)

        with patch("local_coding_assistant.cli.commands.serve._store", store):
            with pytest.raises(HTTPException) as exc:
                await get_session("missing")
        assert exc.value.status_code == 404

    async def test_raises_503_when_store_unavailable(self) -> None:
        with patch("local_coding_assistant.cli.commands.serve._store", None):
            with pytest.raises(HTTPException) as exc:
                await get_session("sess-1")
        assert exc.value.status_code == 503
