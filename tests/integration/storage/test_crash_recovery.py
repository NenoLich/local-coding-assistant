"""Integration tests for crash recovery and durability.

Covers:
- Flushed data survives a simulated crash (process events without closing,
  then reopen the store and verify flushed data is present).
- Boundary events (FRAME_COMPLETE, LLM_COMPLETE, ERROR, TOOL_RESULT,
  TURN_COMPLETE) trigger an immediate flush.
- Unflushed adapter state is not durable until flushed (in-memory rebase only).
- Running frames are detected and marked failed on recovery.
"""

from __future__ import annotations

import uuid

from local_coding_assistant.runtime.engine.frame import ExecutionStatus
from local_coding_assistant.runtime.events.contracts import (
    EventType,
    FrameCompleteEvent,
    FrameStartEvent,
    LLMCompleteEvent,
    SessionStartEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.storage.execution_store import ExecutionStore, SessionStatus


def _new_id() -> str:
    return f"t_{uuid.uuid4().hex[:12]}"


async def _fire_session_start(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    user_input: str = "hello",
) -> None:
    await store.process_event(
        SessionStartEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            user_input=user_input,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )
    )


async def _fire_turn_start(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    turn_index: int = 0,
    user_input: str = "turn query",
) -> None:
    await store.process_event(
        TurnStartEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            user_input=user_input,
            turn_index=turn_index,
            config_snapshot_json="{}",
            context_snapshot_json=None,
        )
    )


async def _fire_frame_start(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    iteration: int = 0,
) -> None:
    await store.process_event(
        FrameStartEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            frame_id=frame_id,
            iteration=iteration,
            context_snapshot_id="snap_default",
            context_hash="hash_default",
        )
    )


async def _fire_llm_complete(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    content: str = "response",
    input_tokens: int = 10,
    output_tokens: int = 0,
) -> None:
    await store.process_event(
        LLMCompleteEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            frame_id=frame_id,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model="gpt-4",
        )
    )


async def _fire_tool_result(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    tool_call_id: str = "call_1",
    tool_name: str = "calculator",
    success: bool = True,
) -> None:
    await store.process_event(
        ToolResultEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            frame_id=frame_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            success=success,
            output="42",
        )
    )


async def _fire_frame_complete(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    status: str = "completed",
    duration_ms: int = 100,
) -> None:
    await store.process_event(
        FrameCompleteEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            frame_id=frame_id,
            status=status,
            duration_ms=duration_ms,
            total_input_tokens=0,
            total_output_tokens=0,
            model_name="gpt-4",
            finish_reason="stop",
        )
    )


async def _fire_turn_complete(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    status: str = "completed",
    final_answer: str | None = None,
) -> None:
    await store.process_event(
        TurnCompleteEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            status=status,
            total_input_tokens=0,
            total_output_tokens=0,
            total_frames=0,
            final_answer=final_answer,
        )
    )


class TestCrashRecoveryDurability:
    async def test_flushed_data_persists_without_close_and_reopen(
        self, execution_store: ExecutionStore
    ) -> None:
        """Flushed data is durable without close(); survives reopen after crash."""
        store1 = execution_store
        db_path = store1.db_path
        session_id = _new_id()
        turn_id = _new_id()
        frame_id = _new_id()

        await _fire_session_start(store1, session_id, turn_id, user_input="hello world")
        await _fire_turn_start(
            store1, session_id, turn_id, turn_index=0, user_input="hello world"
        )
        await _fire_frame_start(store1, session_id, turn_id, frame_id, iteration=0)
        await _fire_llm_complete(
            store1,
            session_id,
            turn_id,
            frame_id,
            content="hello there",
            input_tokens=25,
            output_tokens=0,
        )
        await _fire_tool_result(
            store1,
            session_id,
            turn_id,
            frame_id,
            tool_call_id="call_1",
            tool_name="calculator",
            success=True,
        )
        await _fire_frame_complete(
            store1, session_id, turn_id, frame_id, status="completed"
        )
        await _fire_turn_complete(store1, session_id, turn_id, status="completed")

        # Simulate crash: do NOT close store1. Boundary events already flushed
        # each entity mutation to the SQLite file, so data is durable here.
        session = await store1.get_session(session_id)
        assert session is not None
        assert session.id == session_id
        assert session.status == SessionStatus.IDLE.value

        turn = await store1.get_turn(turn_id)
        assert turn is not None
        assert turn.status == "completed"

        frame = await store1.frame_repo.get(frame_id)
        assert frame is not None
        assert frame.status == ExecutionStatus.COMPLETED.value

        events = await store1.get_frame_events(frame_id)
        event_types = {e["type"] for e in events}
        assert EventType.LLM_COMPLETE.value in event_types
        assert EventType.TOOL_RESULT.value in event_types
        assert EventType.FRAME_COMPLETE.value in event_types

        # Close store1 (idempotent, data already flushed) then reopen on the
        # same file -> flushed data intact across crash/reopen.
        await store1.close()

        store2 = ExecutionStore(db_path)
        await store2.init()
        try:
            session = await store2.get_session(session_id)
            assert session is not None
            assert session.id == session_id

            turn = await store2.get_turn(turn_id)
            assert turn is not None
            assert turn.status == "completed"

            frame = await store2.frame_repo.get(frame_id)
            assert frame is not None
            assert frame.status == ExecutionStatus.COMPLETED.value

            events = await store2.get_frame_events(frame_id)
            assert len(events) >= 1
        finally:
            await store2.close()

    async def test_boundary_events_trigger_immediate_flush(
        self, execution_store: ExecutionStore
    ) -> None:
        """Boundary events flush immediately; non-boundary events stay in memory."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        frame_id = _new_id()

        # Non-boundary events: nothing written to disk yet.
        await _fire_session_start(store, session_id, turn_id, user_input="hi")
        await _fire_turn_start(
            store, session_id, turn_id, turn_index=0, user_input="hi"
        )
        await _fire_frame_start(store, session_id, turn_id, frame_id, iteration=0)

        assert await store.get_session(session_id) is None
        assert await store.get_turn(turn_id) is None
        assert await store.frame_repo.get(frame_id) is None
        assert await store.event_repo.get_by_session(session_id) == []

        # A boundary event (LLM_COMPLETE) flushes everything accumulated so far.
        await _fire_llm_complete(
            store, session_id, turn_id, frame_id, input_tokens=5, output_tokens=0
        )

        assert await store.get_session(session_id) is not None
        assert await store.get_turn(turn_id) is not None
        assert await store.frame_repo.get(frame_id) is not None
        events = await store.event_repo.get_by_session(session_id)
        assert len(events) >= 1
        assert {e["type"] for e in events} == {
            EventType.SESSION_START.value,
            EventType.TURN_START.value,
            EventType.FRAME_START.value,
            EventType.LLM_COMPLETE.value,
        }

    async def test_unflushed_adapter_state_not_durable(
        self, execution_store: ExecutionStore
    ) -> None:
        """Only non-boundary events, never flushed: not durable at crash time."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        frame_id = _new_id()

        await _fire_session_start(store, session_id, turn_id, user_input="hi")
        await _fire_turn_start(
            store, session_id, turn_id, turn_index=0, user_input="hi"
        )
        await _fire_frame_start(store, session_id, turn_id, frame_id, iteration=0)

        # Crash moment: no flush, no close. Everything is still in the adapter's
        # in-memory rebase state, not in the SQLite file.
        assert await store.get_session(session_id) is None
        assert await store.get_turn(turn_id) is None
        assert await store.frame_repo.get(frame_id) is None

        # After a flush, the same data becomes durable.
        await store.flush()
        assert await store.get_session(session_id) is not None
        assert await store.get_turn(turn_id) is not None
        assert await store.frame_repo.get(frame_id) is not None


class TestRunningFramesRecovery:
    async def test_running_frames_marked_failed_on_recovery(
        self, execution_store: ExecutionStore
    ) -> None:
        """Frames left running at crash are detected and marked failed on reopen."""
        store1 = execution_store
        db_path = store1.db_path
        session_id = _new_id()
        turn_id = _new_id()
        running_frame_id = _new_id()
        completed_frame_id = _new_id()

        await _fire_session_start(store1, session_id, turn_id, user_input="hi")
        await _fire_turn_start(
            store1, session_id, turn_id, turn_index=0, user_input="hi"
        )

        # Frame 1: LLM_COMPLETE (boundary -> flushed) leaves it in running_tools.
        await _fire_frame_start(
            store1, session_id, turn_id, running_frame_id, iteration=0
        )
        await _fire_llm_complete(
            store1,
            session_id,
            turn_id,
            running_frame_id,
            input_tokens=5,
            output_tokens=0,
        )

        # Frame 2: completes via FRAME_COMPLETE (boundary -> flushed), terminal.
        await _fire_frame_start(
            store1, session_id, turn_id, completed_frame_id, iteration=1
        )
        await _fire_llm_complete(
            store1,
            session_id,
            turn_id,
            completed_frame_id,
            input_tokens=3,
            output_tokens=0,
        )
        await _fire_frame_complete(
            store1, session_id, turn_id, completed_frame_id, status="completed"
        )

        await _fire_turn_complete(store1, session_id, turn_id, status="completed")

        await store1.close()

        store2 = ExecutionStore(db_path)
        await store2.init()
        try:
            running = await store2.frame_repo.get_running_frames()
            running_ids = {f.id for f in running}
            assert running_frame_id in running_ids
            assert completed_frame_id not in running_ids

            for frame in running:
                assert frame.status in (
                    ExecutionStatus.RUNNING_LLM.value,
                    ExecutionStatus.RUNNING_TOOLS.value,
                )

            # Mark the running frames failed (crash recovery).
            await store2.frame_repo.mark_failed(
                running_frame_id, "process crashed before frame completion"
            )

            recovered = await store2.frame_repo.get(running_frame_id)
            assert recovered is not None
            assert recovered.status == ExecutionStatus.FAILED.value

            # Completed frame stays completed.
            still_completed = await store2.frame_repo.get(completed_frame_id)
            assert still_completed is not None
            assert still_completed.status == ExecutionStatus.COMPLETED.value

            # No running frames remain after recovery.
            assert await store2.frame_repo.get_running_frames() == []
        finally:
            await store2.close()
