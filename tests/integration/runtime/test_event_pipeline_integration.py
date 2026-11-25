"""Integration tests for the runtime event pipeline fan-out.

Covers:
* multi-subscriber fan-out (StoreSubscriber + in-memory spies)
* per-session, monotonic sequence-number assignment shared across subscribers
* subscriber isolation (a failing subscriber never blocks the others)
* concurrent (non-serial) dispatch of subscribers
* unsubscribe / close lifecycle
* durable persistence through StoreSubscriber against a real ExecutionStore,
  including seq continuation across pipeline restarts
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from local_coding_assistant.runtime.engine.subscribers.store_subscriber import (
    StoreSubscriber,
)
from local_coding_assistant.runtime.events.contracts import (
    FrameCompleteEvent,
    FrameStartEvent,
    LLMCompleteEvent,
    LLMStartEvent,
    RuntimeEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.runtime.events.pipeline import EventPipeline
from local_coding_assistant.storage.execution_store import ExecutionStore

SESSION_ID = "sess_pipeline"
TURN_ID = "turn_pipeline"
FRAME_ID = "frame_pipeline"
TRACE_ID = "trace_pipeline"


async def _seed_store_entities(
    store: ExecutionStore,
    session_id: str = SESSION_ID,
    turn_id: str = TURN_ID,
    frame_id: str = FRAME_ID,
    turn_index: int = 0,
    iteration: int = 1,
) -> None:
    """Pre-create session/turn/frame rows so events satisfy FK constraints."""
    await store.create_session(session_id, "hello world", "frame", "gpt-4")
    await store.create_turn(turn_id, session_id, TRACE_ID, turn_index, "hello world")
    await store.create_frame(
        frame_id,
        session_id,
        turn_id,
        TRACE_ID,
        iteration,
        context_snapshot_id="ctx_snap",
    )


class _SpySubscriber:
    """Recording subscriber with optional failure / delay for isolation tests."""

    def __init__(self, name: str = "spy") -> None:
        self.name = name
        self.events: list[RuntimeEvent] = []
        self.fail = False
        self.delay_seconds: float = 0.0

    async def on_event(self, event: RuntimeEvent) -> None:
        if self.fail:
            raise RuntimeError(f"{self.name} boom")
        if self.delay_seconds:
            import asyncio

            await asyncio.sleep(self.delay_seconds)
        self.events.append(event)

    async def close(self) -> None:
        return None


def _turn_start(
    session_id: str = SESSION_ID,
    turn_id: str = TURN_ID,
    turn_index: int = 0,
) -> TurnStartEvent:
    return TurnStartEvent(
        trace_id=TRACE_ID,
        turn_id=turn_id,
        session_id=session_id,
        user_input="hello world",
        turn_index=turn_index,
        agent_mode="frame",
        model_name="gpt-4",
    )


def _frame_start(
    session_id: str = SESSION_ID,
    turn_id: str = TURN_ID,
    frame_id: str = FRAME_ID,
    iteration: int = 1,
) -> FrameStartEvent:
    return FrameStartEvent(
        trace_id=TRACE_ID,
        turn_id=turn_id,
        frame_id=frame_id,
        session_id=session_id,
        iteration=iteration,
        context_snapshot_id="ctx_snap",
        context_hash="sha256:abc",
    )


def _llm_start(
    session_id: str = SESSION_ID,
    turn_id: str = TURN_ID,
    frame_id: str = FRAME_ID,
) -> LLMStartEvent:
    return LLMStartEvent(
        trace_id=TRACE_ID,
        turn_id=turn_id,
        frame_id=frame_id,
        session_id=session_id,
        model="gpt-4",
        prompt_tokens_estimate=100,
    )


def _llm_complete(
    session_id: str = SESSION_ID,
    turn_id: str = TURN_ID,
    frame_id: str = FRAME_ID,
) -> LLMCompleteEvent:
    return LLMCompleteEvent(
        trace_id=TRACE_ID,
        turn_id=turn_id,
        frame_id=frame_id,
        session_id=session_id,
        content="done",
        model="gpt-4",
        input_tokens=10,
        output_tokens=5,
        finish_reason="stop",
    )


def _tool_start(
    session_id: str = SESSION_ID,
    turn_id: str = TURN_ID,
    frame_id: str = FRAME_ID,
    tool_call_id: str = "call_1",
    tool_name: str = "read_file",
) -> ToolStartEvent:
    return ToolStartEvent(
        trace_id=TRACE_ID,
        turn_id=turn_id,
        frame_id=frame_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments={"path": "file.txt"},
        execution_mode="sandbox",
    )


def _tool_result(
    session_id: str = SESSION_ID,
    turn_id: str = TURN_ID,
    frame_id: str = FRAME_ID,
    tool_call_id: str = "call_1",
    tool_name: str = "read_file",
    success: bool = True,
) -> ToolResultEvent:
    return ToolResultEvent(
        trace_id=TRACE_ID,
        turn_id=turn_id,
        frame_id=frame_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        success=success,
        output="content",
        timing={"duration_ms": 10},
    )


def _frame_complete(
    session_id: str = SESSION_ID,
    turn_id: str = TURN_ID,
    frame_id: str = FRAME_ID,
) -> FrameCompleteEvent:
    return FrameCompleteEvent(
        trace_id=TRACE_ID,
        turn_id=turn_id,
        frame_id=frame_id,
        session_id=session_id,
        status="completed",
        finish_reason="stop",
        duration_ms=50,
        total_input_tokens=10,
        total_output_tokens=5,
    )


def _turn_complete(
    session_id: str = SESSION_ID, turn_id: str = TURN_ID
) -> TurnCompleteEvent:
    return TurnCompleteEvent(
        trace_id=TRACE_ID,
        turn_id=turn_id,
        session_id=session_id,
        status="completed",
        final_answer="done",
        total_frames=1,
    )


@pytest.fixture
async def execution_store(tmp_path: Any) -> AsyncIterator[ExecutionStore]:
    """ExecutionStore backed by a temporary SQLite file."""
    store = ExecutionStore(tmp_path / "event_pipeline.db")
    await store.init()
    yield store
    await store.close()


class TestFanOut:
    async def test_all_subscribers_receive_all_events(self) -> None:
        pipeline = EventPipeline()
        subscribers = [_SpySubscriber(f"sub_{i}") for i in range(3)]
        for subscriber in subscribers:
            pipeline.subscribe(subscriber)

        events = [_turn_start(), _frame_start(), _llm_start(), _llm_complete()]
        for event in events:
            await pipeline.emit(event)

        expected_names = [type(event).__name__ for event in events]
        for subscriber in subscribers:
            assert [
                type(event).__name__ for event in subscriber.events
            ] == expected_names

    async def test_all_subscribers_observe_same_seq(self) -> None:
        pipeline = EventPipeline()
        subscribers = [_SpySubscriber("a"), _SpySubscriber("b")]
        for subscriber in subscribers:
            pipeline.subscribe(subscriber)

        event = _turn_start()
        await pipeline.emit(event)

        assert subscribers[0].events[0].seq == subscribers[1].events[0].seq
        assert subscribers[0].events[0].seq == 1

    async def test_emit_with_no_subscribers_does_not_raise(self) -> None:
        pipeline = EventPipeline()
        await pipeline.emit(_turn_start())
        await pipeline.close()


class TestSeqAssignment:
    async def test_seq_is_per_session_and_monotonic(self) -> None:
        pipeline = EventPipeline()
        spy = _SpySubscriber()
        pipeline.subscribe(spy)

        await pipeline.emit(_turn_start(session_id="sess_a"))
        await pipeline.emit(_turn_start(session_id="sess_b"))
        await pipeline.emit(_frame_start(session_id="sess_a"))
        await pipeline.emit(_frame_start(session_id="sess_b"))

        a_seqs = [e.seq for e in spy.events if e.session_id == "sess_a"]
        b_seqs = [e.seq for e in spy.events if e.session_id == "sess_b"]
        assert a_seqs == [1, 2]
        assert b_seqs == [1, 2]
        # global emission order is preserved within each session
        assert [e.session_id for e in spy.events] == [
            "sess_a",
            "sess_b",
            "sess_a",
            "sess_b",
        ]

    async def test_pipeline_without_store_still_assigns_seq(self) -> None:
        pipeline = EventPipeline(execution_store=None)
        spy = _SpySubscriber()
        pipeline.subscribe(spy)
        await pipeline.emit(_turn_start())
        assert spy.events[0].seq == 1


class TestSubscriberIsolation:
    async def test_failing_subscriber_does_not_block_others(self) -> None:
        pipeline = EventPipeline()
        bad = _SpySubscriber("bad")
        bad.fail = True
        good = _SpySubscriber("good")
        pipeline.subscribe(bad)
        pipeline.subscribe(good)

        await pipeline.emit(_turn_start())

        assert len(good.events) == 1
        assert len(bad.events) == 0

    async def test_slow_subscriber_does_not_prevent_others(self) -> None:
        pipeline = EventPipeline()
        slow = _SpySubscriber("slow")
        slow.delay_seconds = 0.2
        fast = _SpySubscriber("fast")
        pipeline.subscribe(slow)
        pipeline.subscribe(fast)

        await pipeline.emit(_turn_start())

        assert len(fast.events) == 1
        assert len(slow.events) == 1

    async def test_unsubscribe_stops_delivery(self) -> None:
        pipeline = EventPipeline()
        a = _SpySubscriber("a")
        b = _SpySubscriber("b")
        pipeline.subscribe(a)
        pipeline.subscribe(b)
        pipeline.unsubscribe(a)

        await pipeline.emit(_turn_start())

        assert len(b.events) == 1
        assert len(a.events) == 0

    async def test_close_clears_subscribers(self) -> None:
        pipeline = EventPipeline()
        pipeline.subscribe(_SpySubscriber())
        await pipeline.close()
        assert pipeline.subscribers == []


class TestPersistence:
    async def test_store_subscriber_persists_events_with_assigned_seq(
        self, execution_store: ExecutionStore
    ) -> None:
        await _seed_store_entities(execution_store)

        pipeline = EventPipeline(execution_store=execution_store)
        pipeline.subscribe(StoreSubscriber(execution_store))
        spy = _SpySubscriber()
        pipeline.subscribe(spy)

        events = [
            _turn_start(),
            _frame_start(),
            _llm_start(),
            _llm_complete(),
            _tool_start(),
            _tool_result(),
            _frame_complete(),
            _turn_complete(),
        ]
        for event in events:
            await pipeline.emit(event)
        await execution_store.flush()

        assert [event.seq for event in spy.events] == list(range(1, len(events) + 1))
        assert await execution_store.get_max_seq(SESSION_ID) == len(events)

        stored = await execution_store.event_repo.get_by_session(SESSION_ID)
        assert [row["type"] for row in stored] == [event.type.value for event in events]
        assert [row["seq"] for row in stored] == [event.seq for event in events]

    async def test_seq_continues_from_store_max_seq_on_restart(
        self, execution_store: ExecutionStore
    ) -> None:
        await _seed_store_entities(
            execution_store, turn_id="turn_1", frame_id="frame_1"
        )

        first = EventPipeline(execution_store=execution_store)
        first.subscribe(StoreSubscriber(execution_store))
        await first.emit(_turn_start(turn_id="turn_1"))
        await first.emit(_turn_start(turn_id="turn_1"))
        await execution_store.flush()
        assert await execution_store.get_max_seq(SESSION_ID) == 2

        await execution_store.create_turn(
            "turn_3", SESSION_ID, TRACE_ID, 1, "hello world"
        )
        await execution_store.create_frame(
            "frame_3",
            SESSION_ID,
            "turn_3",
            TRACE_ID,
            2,
            context_snapshot_id="ctx_snap",
        )

        second = EventPipeline(execution_store=execution_store)
        second.subscribe(StoreSubscriber(execution_store))
        spy = _SpySubscriber()
        second.subscribe(spy)
        await second.emit(_turn_start(turn_id="turn_3"))
        await execution_store.flush()

        assert spy.events[0].seq == 3
        assert await execution_store.get_max_seq(SESSION_ID) == 3
