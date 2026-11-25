"""Integration tests for full session lifecycle through ExecutionStore.

End-to-end flows: session creation, turns, frames, events, messages,
context snapshots, status transitions, totals consistency, message projection,
and round-trip persistence (close -> reopen -> verify).
"""

from __future__ import annotations

import json
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
    agent_mode: str = "frame_agent",
    model_name: str = "gpt-4",
) -> None:
    await store.process_event(
        SessionStartEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            user_input=user_input,
            agent_mode=agent_mode,
            model_name=model_name,
        )
    )


async def _fire_turn_start(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    turn_index: int,
    user_input: str = "turn query",
    config_snapshot_json: str | None = "{}",
    context_snapshot_json: str | None = None,
) -> None:
    await store.process_event(
        TurnStartEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            user_input=user_input,
            turn_index=turn_index,
            config_snapshot_json=config_snapshot_json,
            context_snapshot_json=context_snapshot_json,
        )
    )


async def _fire_frame_start(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    iteration: int,
    context_snapshot_id: str | None = None,
) -> None:
    await store.process_event(
        FrameStartEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            frame_id=frame_id,
            iteration=iteration,
            context_snapshot_id=context_snapshot_id or "snap_default",
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
    model: str = "gpt-4",
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
            model=model,
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
    output: str = "42",
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
            output=output,
        )
    )


async def _fire_frame_complete(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    status: str = "completed",
    duration_ms: int = 100,
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    finish_reason: str = "stop",
) -> None:
    await store.process_event(
        FrameCompleteEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            frame_id=frame_id,
            status=status,
            duration_ms=duration_ms,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            model_name="gpt-4",
            finish_reason=finish_reason,
        )
    )


async def _fire_turn_complete(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    status: str = "completed",
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    total_frames: int = 0,
    final_answer: str | None = None,
) -> None:
    await store.process_event(
        TurnCompleteEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            status=status,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_frames=total_frames,
            final_answer=final_answer,
        )
    )


class TestFullSessionLifecycleRoundtrip:
    async def test_full_session_lifecycle_roundtrip(
        self, execution_store: ExecutionStore
    ) -> None:
        """Round-trip: full session through events, close, reopen, verify all data."""
        store1 = execution_store
        db_path = store1.db_path

        session_id = _new_id()
        turn_id = _new_id()
        frame_id = _new_id()
        snapshot_id = _new_id()

        context_snapshot_json = json.dumps(
            {
                "id": snapshot_id,
                "schema_version": 1,
                "tool_set_ref": {
                    "tool_ids": [1],
                    "tool_names": ["calculator"],
                    "version": "1.0.0",
                },
                "template_set_ref": {
                    "template_ids": [],
                    "template_paths": [],
                    "version": "1.0.0",
                },
                "content_hash": "snap_hash_abc",
                "created_at_ms": 1000,
            }
        )

        await store1.register_tool(
            name="calculator", schema_json="{}", risk_class="low"
        )

        await _fire_session_start(store1, session_id, turn_id, user_input="hello world")
        await _fire_turn_start(
            store1,
            session_id,
            turn_id,
            turn_index=0,
            user_input="hello world",
            context_snapshot_json=context_snapshot_json,
        )
        await _fire_frame_start(
            store1,
            session_id,
            turn_id,
            frame_id,
            iteration=0,
            context_snapshot_id=snapshot_id,
        )
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
            output="42",
        )
        await _fire_frame_complete(
            store1,
            session_id,
            turn_id,
            frame_id,
            status="completed",
            duration_ms=200,
            total_input_tokens=0,
            total_output_tokens=0,
            finish_reason="stop",
        )
        await _fire_turn_complete(
            store1,
            session_id,
            turn_id,
            status="completed",
            total_input_tokens=0,
            total_output_tokens=0,
            total_frames=0,
            final_answer="hello there",
        )

        await store1.close()

        store2 = ExecutionStore(db_path)
        await store2.init()
        try:
            session = await store2.get_session(session_id)
            assert session is not None
            assert session.id == session_id
            assert session.user_input == "hello world"
            assert session.status == SessionStatus.IDLE.value
            assert session.agent_mode == "frame_agent"
            assert session.model_name == "gpt-4"
            assert session.total_input_tokens == 25
            assert session.total_turns == 1
            assert session.total_frames == 1

            turn = await store2.get_turn(turn_id)
            assert turn is not None
            assert turn.id == turn_id
            assert turn.session_id == session_id
            assert turn.turn_index == 0
            assert turn.status == "completed"
            assert turn.total_input_tokens == 25
            assert turn.total_frames == 1
            assert turn.final_answer == "hello there"

            frame = await store2.frame_repo.get(frame_id)
            assert frame is not None
            assert frame.id == frame_id
            assert frame.session_id == session_id
            assert frame.turn_id == turn_id
            assert frame.iteration == 0
            assert frame.status == ExecutionStatus.COMPLETED.value
            assert frame.input_tokens == 25
            assert frame.context_snapshot_id == snapshot_id

            snapshot = await store2.get_context_snapshot_by_turn(turn_id)
            assert snapshot is not None
            assert snapshot.id == snapshot_id
            assert snapshot.turn_id == turn_id
            assert snapshot.content_hash == "snap_hash_abc"

            events = await store2.get_frame_events(frame_id)
            event_types = {e["type"] for e in events}
            assert EventType.FRAME_START.value in event_types
            assert EventType.LLM_COMPLETE.value in event_types
            assert EventType.TOOL_RESULT.value in event_types
            assert EventType.FRAME_COMPLETE.value in event_types
            frame_seqs = sorted({e["seq"] for e in events})
            # Setup consumes seqs 1 (session_start), 2 (turn_start), 3 (frame_start),
            # so the frame's events continue the per-session counter at 3..6.
            assert frame_seqs == [3, 4, 5, 6]

            messages = await store2.get_all_messages(session_id)
            assert len(messages) == 3
            roles = {m["role"] for m in messages}
            assert roles == {"user", "assistant", "tool"}

            batch = await store2.load_resume_batch(session_id)
            assert batch is not None
            assert batch.session.id == session_id
            assert batch.turn.id == turn_id
            assert batch.snapshot is not None
            assert batch.snapshot.id == snapshot_id
            assert len(batch.frames) == 1
            assert batch.frames[0].id == frame_id
            assert batch.missing_refs.is_empty is True

            replay = await store2.load_replay_batch(session_id)
            assert replay is not None
            assert replay.session.id == session_id
            assert len(replay.turns) == 1
            assert replay.turns[0].turn.id == turn_id
            assert len(replay.turns[0].frames) == 1
            assert frame_id in replay.turns[0].frame_events
        finally:
            await store2.close()

    async def test_multi_turn_session_totals_consistency(
        self, execution_store: ExecutionStore
    ) -> None:
        """Multi-turn session: session totals == sum of turn totals == sum of frame tokens."""
        store = execution_store
        session_id = _new_id()

        turn_ids = []
        frame_ids = []
        tokens_per_turn = [25, 40]

        for turn_index, tokens in enumerate(tokens_per_turn):
            turn_id = _new_id()
            frame_id = _new_id()
            turn_ids.append(turn_id)
            frame_ids.append(frame_id)

            await _fire_session_start(
                store, session_id, turn_id, user_input=f"turn {turn_index}"
            )
            await _fire_turn_start(
                store,
                session_id,
                turn_id,
                turn_index=turn_index,
                user_input=f"turn {turn_index}",
            )
            await _fire_frame_start(
                store,
                session_id,
                turn_id,
                frame_id,
                iteration=0,
            )
            await _fire_llm_complete(
                store,
                session_id,
                turn_id,
                frame_id,
                content=f"response {turn_index}",
                input_tokens=tokens,
                output_tokens=0,
            )
            await _fire_frame_complete(
                store,
                session_id,
                turn_id,
                frame_id,
                status="completed",
                duration_ms=100,
            )
            await _fire_turn_complete(
                store,
                session_id,
                turn_id,
                status="completed",
            )

        session = await store.get_session(session_id)
        assert session is not None
        assert session.total_turns == 2
        assert session.total_frames == 2
        assert session.total_input_tokens == sum(tokens_per_turn)

        turns = await store.list_turns_by_session(session_id)
        assert len(turns) == 2
        turn_total_tokens = sum(t.total_input_tokens for t in turns)
        turn_total_frames = sum(t.total_frames for t in turns)
        assert turn_total_tokens == session.total_input_tokens
        assert turn_total_frames == session.total_frames

        frames = await store.frame_repo.list_by_session(session_id)
        frame_total_tokens = sum(f.input_tokens for f in frames)
        assert frame_total_tokens == session.total_input_tokens

    async def test_session_status_transitions(
        self, execution_store: ExecutionStore
    ) -> None:
        """Session status: idle -> running (turn start) -> idle (turn complete)."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()

        await _fire_session_start(store, session_id, turn_id, user_input="hi")
        await store.flush()
        session = await store.get_session(session_id)
        assert session.status == SessionStatus.IDLE.value

        await _fire_turn_start(store, session_id, turn_id, turn_index=0)
        await store.flush()
        session = await store.get_session(session_id)
        assert session.status == SessionStatus.RUNNING.value

        await _fire_turn_complete(store, session_id, turn_id, status="completed")
        session = await store.get_session(session_id)
        assert session.status == SessionStatus.IDLE.value

    async def test_turn_and_frame_status_transitions(
        self, execution_store: ExecutionStore
    ) -> None:
        """Turn running -> completed; frame running_llm -> running_tools -> completed."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        frame_id = _new_id()

        await _fire_session_start(store, session_id, turn_id, user_input="hi")
        await _fire_turn_start(store, session_id, turn_id, turn_index=0)
        await store.flush()

        turn = await store.get_turn(turn_id)
        assert turn.status == "running"

        await _fire_frame_start(store, session_id, turn_id, frame_id, iteration=0)
        await store.flush()
        frame = await store.frame_repo.get(frame_id)
        assert frame.status == ExecutionStatus.RUNNING_LLM.value

        await _fire_llm_complete(
            store, session_id, turn_id, frame_id, input_tokens=10, output_tokens=0
        )
        frame = await store.frame_repo.get(frame_id)
        assert frame.status == ExecutionStatus.RUNNING_TOOLS.value

        await _fire_frame_complete(
            store, session_id, turn_id, frame_id, status="completed"
        )
        frame = await store.frame_repo.get(frame_id)
        assert frame.status == ExecutionStatus.COMPLETED.value

        await _fire_turn_complete(store, session_id, turn_id, status="completed")
        turn = await store.get_turn(turn_id)
        assert turn.status == "completed"

    async def test_context_snapshot_created_and_referenced(
        self, execution_store: ExecutionStore
    ) -> None:
        """Context snapshot created on TurnStartEvent, referenced by frames."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        frame_id = _new_id()
        snapshot_id = _new_id()

        context_snapshot_json = json.dumps(
            {
                "id": snapshot_id,
                "schema_version": 1,
                "tool_set_ref": {
                    "tool_ids": [1],
                    "tool_names": ["calculator"],
                    "version": "1.0.0",
                },
                "template_set_ref": {
                    "template_ids": [],
                    "template_paths": [],
                    "version": "1.0.0",
                },
                "content_hash": "snap_hash_xyz",
                "created_at_ms": 2000,
            }
        )

        await store.register_tool(name="calculator", schema_json="{}", risk_class="low")

        await _fire_session_start(store, session_id, turn_id, user_input="hi")
        await _fire_turn_start(
            store,
            session_id,
            turn_id,
            turn_index=0,
            context_snapshot_json=context_snapshot_json,
        )
        await _fire_frame_start(
            store,
            session_id,
            turn_id,
            frame_id,
            iteration=0,
            context_snapshot_id=snapshot_id,
        )
        await _fire_llm_complete(
            store, session_id, turn_id, frame_id, input_tokens=5, output_tokens=0
        )
        await _fire_frame_complete(
            store, session_id, turn_id, frame_id, status="completed"
        )
        await _fire_turn_complete(store, session_id, turn_id, status="completed")

        snapshot = await store.get_context_snapshot_by_turn(turn_id)
        assert snapshot is not None
        assert snapshot.id == snapshot_id
        assert snapshot.turn_id == turn_id
        assert snapshot.content_hash == "snap_hash_xyz"

        frame = await store.frame_repo.get(frame_id)
        assert frame.context_snapshot_id == snapshot_id

    async def test_message_projection_and_event_sequence(
        self, execution_store: ExecutionStore
    ) -> None:
        """Message projection: user (TurnStart), assistant (LLMComplete), tool (ToolResult)."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        frame_id = _new_id()

        await _fire_session_start(store, session_id, turn_id, user_input="hello")
        await _fire_turn_start(
            store, session_id, turn_id, turn_index=0, user_input="hello"
        )
        await _fire_frame_start(store, session_id, turn_id, frame_id, iteration=0)
        await _fire_llm_complete(
            store,
            session_id,
            turn_id,
            frame_id,
            content="assistant reply",
            input_tokens=10,
            output_tokens=0,
        )
        await _fire_tool_result(
            store,
            session_id,
            turn_id,
            frame_id,
            tool_call_id="call_1",
            tool_name="calculator",
            success=True,
            output="42",
        )
        await _fire_frame_complete(
            store, session_id, turn_id, frame_id, status="completed"
        )
        await _fire_turn_complete(store, session_id, turn_id, status="completed")

        messages = await store.get_all_messages(session_id)
        assert len(messages) == 3

        user_msgs = [m for m in messages if m["role"] == "user"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        tool_msgs = [m for m in messages if m["role"] == "tool"]

        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "hello"
        assert user_msgs[0]["turn_id"] == turn_id

        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "assistant reply"
        assert assistant_msgs[0]["frame_id"] == frame_id

        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_name"] == "calculator"
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert tool_msgs[0]["frame_id"] == frame_id

        turn_events = await store.event_repo.get_by_turn(turn_id)
        turn_event_types = {e["type"] for e in turn_events}
        assert EventType.SESSION_START.value in turn_event_types
        assert EventType.TURN_START.value in turn_event_types
        assert EventType.TURN_COMPLETE.value in turn_event_types

        frame_events = await store.get_frame_events(frame_id)
        frame_event_types = {e["type"] for e in frame_events}
        assert EventType.FRAME_START.value in frame_event_types
        assert EventType.LLM_COMPLETE.value in frame_event_types
        assert EventType.TOOL_RESULT.value in frame_event_types
        assert EventType.FRAME_COMPLETE.value in frame_event_types

        frame_seqs = sorted({e["seq"] for e in frame_events})
        # Setup consumes seqs 1 (session_start), 2 (turn_start), 3 (frame_start),
        # so the frame's events continue the per-session counter at 3..6.
        assert frame_seqs == [3, 4, 5, 6]

    async def test_failed_turn_and_frame(self, execution_store: ExecutionStore) -> None:
        """Turn and frame can fail: frame running_tools -> failed, turn running -> failed."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        frame_id = _new_id()

        await _fire_session_start(store, session_id, turn_id, user_input="hi")
        await _fire_turn_start(store, session_id, turn_id, turn_index=0)
        await _fire_frame_start(store, session_id, turn_id, frame_id, iteration=0)
        await _fire_llm_complete(
            store, session_id, turn_id, frame_id, input_tokens=5, output_tokens=0
        )

        await _fire_tool_result(
            store,
            session_id,
            turn_id,
            frame_id,
            tool_call_id="call_1",
            tool_name="calculator",
            success=False,
            output="",
        )

        frame = await store.frame_repo.get(frame_id)
        assert frame.status == ExecutionStatus.RUNNING_TOOLS.value

        await _fire_frame_complete(
            store,
            session_id,
            turn_id,
            frame_id,
            status="failed",
            finish_reason="error",
        )
        frame = await store.frame_repo.get(frame_id)
        assert frame.status == ExecutionStatus.FAILED.value

        await _fire_turn_complete(store, session_id, turn_id, status="failed")
        turn = await store.get_turn(turn_id)
        assert turn.status == "failed"
