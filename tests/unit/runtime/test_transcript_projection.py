"""Tests for the event-driven transcript projection.

Covers:
- MessageService.build_message: pure event -> (message, dedupe_key) mapping,
  including failure/cancellation as data, P5 (assistant tool_calls) and
  P6 (no empty assistant messages)
- TranscriptSubscriber: live session mirror and idempotency
- StoreEventAdapter + ExecutionStore: atomic event+message projection,
  SQL per-session seq assignment, and INSERT OR IGNORE idempotency
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_coding_assistant.runtime.context.profile import (
    ContextProfile,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.engine.subscribers.transcript_subscriber import (
    TranscriptSubscriber,
)
from local_coding_assistant.runtime.events.contracts import (
    FrameStartEvent,
    LLMCompleteEvent,
    SessionStartEvent,
    ToolResultEvent,
    TurnStartEvent,
)
from local_coding_assistant.runtime.events.projector import MessageService
from local_coding_assistant.runtime.session import Message, SessionState
from local_coding_assistant.runtime.session.session_manager import _message_from_row
from local_coding_assistant.storage.execution_store import ExecutionStore
from local_coding_assistant.storage.store_event_adapter import StoreEventAdapter
from local_coding_assistant.tools_platform.errors import (
    ToolErrorCategory,
    ToolErrorCode,
)


def _context_profile() -> ContextProfile:
    return ContextProfile(
        runtime_context_config=RuntimeContextConfig(),
        tool_set_ref=ToolSetRef(),
        template_set_ref=TemplateSetRef(),
    )


def _turn_start_event(**kwargs: Any) -> TurnStartEvent:
    defaults: dict[str, Any] = {
        "trace_id": "tr_1",
        "turn_id": "turn_1",
        "session_id": "sess_1",
        "user_input": "hello agent",
        "turn_index": 0,
    }
    defaults.update(kwargs)
    return TurnStartEvent(**defaults)


def _llm_complete_event(**kwargs: Any) -> LLMCompleteEvent:
    defaults: dict[str, Any] = {
        "trace_id": "tr_1",
        "turn_id": "turn_1",
        "frame_id": "frame_1",
        "session_id": "sess_1",
        "content": "I will call a tool",
        "model": "test_model",
        "total_tokens": 42,
    }
    defaults.update(kwargs)
    return LLMCompleteEvent(**defaults)


def _tool_result_event(**kwargs: Any) -> ToolResultEvent:
    defaults: dict[str, Any] = {
        "trace_id": "tr_1",
        "turn_id": "turn_1",
        "frame_id": "frame_1",
        "session_id": "sess_1",
        "tool_call_id": "call_1",
        "tool_name": "tool_a",
        "success": True,
        "output": "tool output",
        "arguments": {"p": 1},
    }
    defaults.update(kwargs)
    return ToolResultEvent(**defaults)


def _interrupted_error() -> dict[str, Any]:
    return {
        "code": ToolErrorCode.INTERRUPTED.value,
        "category": ToolErrorCategory.CONCURRENCY.value,
        "message": "Tool execution interrupted",
        "details": {},
    }


class TestBuildMessage:
    """MessageService.build_message: pure event -> (message, key)."""

    def test_turn_start_projects_user_message(self):
        message, key = MessageService().build_message(_turn_start_event())
        assert message is not None
        assert message.role == "user"
        assert message.content == "hello agent"
        assert key == "user:turn_1"

    def test_llm_complete_projects_assistant_with_tool_calls(self):
        event = _llm_complete_event(
            tool_calls=[{"id": "call_1", "name": "tool_a", "arguments": {"p": 1}}]
        )
        message, key = MessageService().build_message(event)
        assert message is not None
        assert message.role == "assistant"
        assert message.content == "I will call a tool"
        # P5: assistant messages carry their tool calls
        assert message.tool_calls == [
            {"id": "call_1", "name": "tool_a", "arguments": {"p": 1}}
        ]
        assert key == "assistant:frame_1"

    def test_llm_complete_preserves_thought_signature_in_tool_calls(self):
        """extra_content (thought_signature) must survive the projector so the
        next LLM call receives it for every tool call."""
        event = _llm_complete_event(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": {"location": "SF"},
                    "extra_content": {"thought_signature": "sig_abc123"},
                },
                {
                    "id": "call_2",
                    "name": "get_weather",
                    "arguments": {"location": "NY"},
                },
            ]
        )
        message, _key = MessageService().build_message(event)
        assert message is not None
        assert message.tool_calls == [
            {
                "id": "call_1",
                "name": "get_weather",
                "arguments": {"location": "SF"},
                "extra_content": {"thought_signature": "sig_abc123"},
            },
            {
                "id": "call_2",
                "name": "get_weather",
                "arguments": {"location": "NY"},
            },
        ]

    def test_llm_complete_without_content_still_writes_tool_calls(self):
        event = _llm_complete_event(
            content="", tool_calls=[{"id": "call_1", "name": "tool_a", "arguments": {}}]
        )
        message, _key = MessageService().build_message(event)
        assert message is not None
        assert message.content is None
        assert message.tool_calls is not None

    def test_empty_generation_writes_nothing(self):
        """P6: a failed/transient generation produces no message."""
        message = MessageService().build_message(
            _llm_complete_event(content="", tool_calls=[])
        )
        assert message is None

    def test_tool_success_projects_output(self):
        message, key = MessageService().build_message(_tool_result_event())
        assert message is not None
        assert message.role == "tool"
        assert message.tool_call_id == "call_1"
        assert message.content == "tool output"
        assert message.tool_calls is None
        assert key == "tool:turn_1:call_1"

    def test_tool_failure_projects_error_text(self):
        message, _key = MessageService().build_message(
            _tool_result_event(
                success=False,
                output=None,
                error={
                    "code": ToolErrorCode.BACKEND_FAILURE.value,
                    "category": ToolErrorCategory.INFRASTRUCTURE.value,
                    "message": "disk on fire",
                },
            )
        )
        assert message is not None
        assert message.content == "Error: disk on fire"

    def test_tool_cancellation_projects_interruption_note(self):
        """Cancellation is data: the tool did not finish."""
        message, _key = MessageService().build_message(
            _tool_result_event(success=False, output=None, error=_interrupted_error())
        )
        assert message is not None
        assert message.content == "Tool call interrupted: Tool execution interrupted"

    def test_structured_output_is_json_rendered(self):
        message, _key = MessageService().build_message(
            _tool_result_event(output={"files": ["a.py", "b.py"]})
        )
        assert message is not None
        assert json.loads(message.content) == {"files": ["a.py", "b.py"]}

    def test_unknown_event_projects_nothing(self):
        assert (
            MessageService().build_message(
                FrameStartEvent(
                    trace_id="tr_1",
                    turn_id="turn_1",
                    frame_id="frame_1",
                    session_id="sess_1",
                    iteration=1,
                    context_snapshot_id="ctx_1",
                    context_hash="sha256:abc",
                )
            )
            is None
        )


class TestTranscriptSubscriber:
    """Live session mirror with idempotency."""

    @staticmethod
    def _session() -> SessionState:
        return SessionState(id="sess_1", context_profile=_context_profile())

    @pytest.mark.asyncio
    async def test_events_append_to_live_session_in_order(self):
        session = self._session()
        subscriber = TranscriptSubscriber(lambda sid: session)

        await subscriber.on_event(_turn_start_event())
        await subscriber.on_event(
            _llm_complete_event(
                tool_calls=[{"id": "call_1", "name": "tool_a", "arguments": {"p": 1}}]
            )
        )
        await subscriber.on_event(_tool_result_event())

        assert [m.role for m in session.history] == ["user", "assistant", "tool"]
        assert [m.seq for m in session.history] == [1, 2, 3]
        assert session.history[2].tool_call_id == "call_1"

    @pytest.mark.asyncio
    async def test_repeated_event_does_not_duplicate(self):
        session = self._session()
        subscriber = TranscriptSubscriber(lambda sid: session)

        event = _tool_result_event()
        await subscriber.on_event(event)
        await subscriber.on_event(event)

        assert len(session.history) == 1

    @pytest.mark.asyncio
    async def test_tool_message_deduped_against_loaded_history(self):
        """A cold-started session (history from store) is not duplicated."""
        session = self._session()
        session.append_transcript_message(
            Message(role="tool", content="tool output", tool_call_id="call_1")
        )
        subscriber = TranscriptSubscriber(lambda sid: session)

        await subscriber.on_event(_tool_result_event())

        assert len(session.history) == 1

    @pytest.mark.asyncio
    async def test_unknown_session_is_ignored(self):
        subscriber = TranscriptSubscriber(lambda sid: None)
        await subscriber.on_event(_turn_start_event())
        assert subscriber._seen_keys == {}


class TestStoreProjection:
    """Adapter + store: atomic projection, seq assignment, idempotency."""

    def test_adapter_projects_message_into_same_batch_as_event(self):
        adapter = StoreEventAdapter()
        adapter.process(_turn_start_event())

        batch = adapter.prepare_batch()
        event_ops = [op for op in batch if op["op"] == "event"]
        message_ops = [op for op in batch if op["op"] == "message"]

        assert len(event_ops) == 1
        assert len(message_ops) == 1
        assert message_ops[0]["data"] == {
            "session_id": "sess_1",
            "turn_id": "turn_1",
            "frame_id": "turn_start",
            "seq": None,
            "role": "user",
            "content": "hello agent",
            "tool_call_id": None,
            "tool_name": None,
            "metadata_json": None,
            "dedupe_key": "user:turn_1",
            "created_at_ms": message_ops[0]["data"]["created_at_ms"],
        }

    @pytest.mark.asyncio
    async def test_full_turn_events_persist_transcript(self, tmp_path: Path):
        store = ExecutionStore(
            tmp_path / "proj.db", flush_interval_ms=100, batch_size=50
        )
        await store.init()
        try:
            session_id = str(uuid.uuid4())
            await store.create_session(
                session_id=session_id,
                user_input="hello agent",
                agent_mode="frame",
                model_name=None,
            )
            await store.process_event(
                SessionStartEvent(
                    trace_id="tr_1",
                    turn_id="turn_1",
                    session_id=session_id,
                    user_input="hello agent",
                    agent_mode="frame",
                )
            )
            await store.process_event(
                _turn_start_event(session_id=session_id, turn_id="turn_1")
            )
            await store.process_event(
                FrameStartEvent(
                    trace_id="tr_1",
                    turn_id="turn_1",
                    frame_id="frame_1",
                    session_id=session_id,
                    iteration=1,
                    context_snapshot_id="ctx_1",
                    context_hash="sha256:abc",
                )
            )
            await store.process_event(
                _llm_complete_event(
                    session_id=session_id,
                    tool_calls=[
                        {"id": "call_1", "name": "tool_a", "arguments": {"p": 1}}
                    ],
                )
            )
            await store.process_event(_tool_result_event(session_id=session_id))

            messages = await store.get_all_messages(session_id)
            assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
            # SQL-assigned, gapless per-session sequence numbers
            assert [m["seq"] for m in messages] == [1, 2, 3]
            assert messages[0]["content"] == "hello agent"
            assert messages[1]["dedupe_key"] == "assistant:frame_1"
            assert json.loads(messages[1]["metadata_json"])["tool_calls"] == [
                {"id": "call_1", "name": "tool_a", "arguments": {"p": 1}}
            ]
            assert messages[2]["content"] == "tool output"
            assert messages[2]["dedupe_key"] == "tool:turn_1:call_1"

            # Events and their messages are committed together (same batch)
            events = await store.event_repo.get_by_session(session_id)
            assert events, "source events must be persisted"

            # Re-processing the same events must not duplicate messages
            await store.process_event(_tool_result_event(session_id=session_id))
            messages_after = await store.get_all_messages(session_id)
            assert len(messages_after) == 3
        finally:
            await store.close()


class TestRuntimeManagerWiring:
    """The transcript subscriber is registered downstream of SessionManager."""

    def test_runtime_manager_registers_transcript_subscriber(self):
        from unittest.mock import MagicMock

        from local_coding_assistant.runtime.events.pipeline import EventPipeline
        from local_coding_assistant.runtime.manager import RuntimeManager

        pipeline = EventPipeline()
        manager = RuntimeManager(
            llm_service=MagicMock(),
            tool_gateway=MagicMock(),
            event_pipeline=pipeline,
            store=MagicMock(),
            config_manager=MagicMock(),
            repository_context_service=MagicMock(),
        )

        assert any(isinstance(s, TranscriptSubscriber) for s in pipeline.subscribers)
        assert manager._session_manager is not None


class TestMessageFromRow:
    """History loading restores assistant tool_calls (P5) after a restart."""

    def test_tool_calls_restored_from_metadata(self):
        message = _message_from_row(
            {
                "role": "assistant",
                "content": "I will call a tool",
                "tool_call_id": None,
                "seq": 7,
                "metadata_json": json.dumps(
                    {"tool_calls": [{"id": "c1", "name": "tool_a", "arguments": {}}]}
                ),
            }
        )
        assert message.role == "assistant"
        assert message.seq == 7
        assert message.tool_calls == [{"id": "c1", "name": "tool_a", "arguments": {}}]

    def test_row_without_metadata_has_no_tool_calls(self):
        message = _message_from_row(
            {"role": "user", "content": "hi", "tool_call_id": None, "seq": 1}
        )
        assert message.tool_calls is None
