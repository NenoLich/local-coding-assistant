"""Integration tests for resume & replay reconstruction round-trips.

Covers load_resume_batch, load_replay_batch, load_turn_replay,
build_frame frame reconstruction, and ContextSnapshot.to_runtime().
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from local_coding_assistant.runtime.engine.frame import ExecutionStatus, build_frame
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
    input_tokens: int = 6,
    output_tokens: int = 4,
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


class TestLoadResumeBatch:
    async def test_resume_batch_roundtrip_structure(
        self, execution_store: ExecutionStore
    ) -> None:
        """Full session -> close -> reopen -> load_resume_batch verifies structure."""
        store1 = execution_store
        db_path = store1.db_path

        session_id = _new_id()
        turn_id = _new_id()
        frame_id = _new_id()
        snapshot_id = _new_id()

        tool_id = await store1.register_tool(
            name="calculator", schema_json="{}", risk_class="low"
        )

        context_snapshot_json = json.dumps(
            {
                "id": snapshot_id,
                "schema_version": 1,
                "tool_set_ref": {
                    "tool_ids": [tool_id],
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
            input_tokens=15,
            output_tokens=10,
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
        await _fire_turn_complete(store1, session_id, turn_id, status="completed")

        await store1.close()

        store2 = ExecutionStore(db_path)
        await store2.init()
        try:
            batch = await store2.load_resume_batch(session_id)
            assert batch is not None
            assert batch.session.id == session_id
            assert batch.session.status == SessionStatus.IDLE.value
            assert batch.session.user_input == "hello world"
            assert len(batch.turns) == 1
            assert batch.turn.id == turn_id
            assert batch.turn.status == "completed"
            assert batch.snapshot is not None
            assert batch.snapshot.id == snapshot_id
            assert len(batch.frames) == 1
            assert batch.frames[0].id == frame_id
            assert batch.frames[0].status == ExecutionStatus.COMPLETED.value
            assert frame_id in batch.frame_events
            assert len(batch.frame_events[frame_id]) >= 1
            assert batch.missing_refs.is_empty is True
        finally:
            await store2.close()

    async def test_turn_selection_by_id_index_and_default(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_resume_batch selects by turn_id, turn_index, or defaults to last."""
        store = execution_store
        session_id = _new_id()

        turn_ids = []
        for index in range(3):
            turn_id = _new_id()
            turn_ids.append(turn_id)
            await _fire_session_start(
                store, session_id, turn_id, user_input=f"turn {index}"
            )
            await _fire_turn_start(
                store, session_id, turn_id, turn_index=index, user_input=f"turn {index}"
            )
            await _fire_frame_start(
                store, session_id, turn_id, f"{turn_id}_f", iteration=0
            )
            await _fire_frame_complete(
                store, session_id, turn_id, f"{turn_id}_f", status="completed"
            )
            await _fire_turn_complete(store, session_id, turn_id, status="completed")

        by_id = await store.load_resume_batch(session_id, turn_id=turn_ids[0])
        assert by_id is not None
        assert by_id.turn.id == turn_ids[0]

        by_index = await store.load_resume_batch(session_id, turn_index=1)
        assert by_index is not None
        assert by_index.turn.id == turn_ids[1]

        default = await store.load_resume_batch(session_id)
        assert default is not None
        assert default.turn.id == turn_ids[2]

        missing = await store.load_resume_batch(session_id, turn_id="turn_nope")
        assert missing is None

    async def test_frame_selection_explicit_and_default(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_resume_batch selects explicit frame_id or defaults to last frame."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        frame1 = _new_id()
        frame2 = _new_id()

        await _fire_session_start(store, session_id, turn_id, user_input="hi")
        await _fire_turn_start(
            store, session_id, turn_id, turn_index=0, user_input="hi"
        )
        await _fire_frame_start(store, session_id, turn_id, frame1, iteration=0)
        await _fire_llm_complete(
            store, session_id, turn_id, frame1, input_tokens=6, output_tokens=4
        )
        await _fire_frame_complete(
            store, session_id, turn_id, frame1, status="completed"
        )
        await _fire_frame_start(store, session_id, turn_id, frame2, iteration=1)
        await _fire_llm_complete(
            store, session_id, turn_id, frame2, input_tokens=12, output_tokens=8
        )
        await _fire_frame_complete(
            store, session_id, turn_id, frame2, status="completed"
        )
        await _fire_turn_complete(store, session_id, turn_id, status="completed")

        explicit = await store.load_resume_batch(
            session_id, turn_id=turn_id, frame_id=frame1
        )
        assert explicit is not None
        assert set(explicit.frame_events) == {frame1}
        assert len(explicit.frame_events[frame1]) >= 1

        default = await store.load_resume_batch(session_id, turn_id=turn_id)
        assert default is not None
        assert set(default.frame_events) == {frame2}
        assert len(default.frame_events[frame2]) >= 1

    async def test_missing_refs_reported(self, execution_store: ExecutionStore) -> None:
        """load_resume_batch reports missing tool_ids, template_ids, repo_hashes."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()

        existing_tool_id = await store.register_tool(
            name="calc", schema_json="{}", risk_class="low"
        )
        existing_template_id = await store.register_template(
            path="x.jinja2", content="X", content_hash="h1"
        )
        await store.register_repo_snapshot(
            target_project_root="/root",
            repo_map_json="{}",
            project_info_json=None,
            repo_snapshot_hash="abc123",
        )

        context_snapshot_json = json.dumps(
            {
                "id": f"snap_{turn_id}",
                "schema_version": 1,
                "tool_set_ref": {
                    "tool_ids": [existing_tool_id, 999],
                    "tool_names": ["calc", "missing"],
                    "version": "1.0.0",
                },
                "template_set_ref": {
                    "template_ids": [existing_template_id, 888],
                    "template_paths": ["x.jinja2", "missing.jinja2"],
                    "version": "1.0.0",
                },
                "repo_snapshot_ref": {
                    "target_project_root": "/root",
                    "repo_snapshot_hash": "deadbeef",
                },
                "content_hash": "hash",
                "created_at_ms": 1000,
            }
        )

        frame_id = _new_id()
        await _fire_session_start(store, session_id, turn_id, user_input="hi")
        await _fire_turn_start(
            store,
            session_id,
            turn_id,
            turn_index=0,
            user_input="hi",
            context_snapshot_json=context_snapshot_json,
        )
        await _fire_frame_start(store, session_id, turn_id, frame_id, iteration=0)
        await _fire_frame_complete(
            store, session_id, turn_id, frame_id, status="completed"
        )
        await _fire_turn_complete(store, session_id, turn_id, status="completed")

        batch = await store.load_resume_batch(session_id, turn_id=turn_id)
        assert batch is not None
        assert batch.snapshot is not None
        missing = batch.missing_refs
        assert missing.tool_ids == [999]
        assert missing.template_ids == [888]
        assert missing.repo_hashes == ["deadbeef"]
        assert missing.memory_ids == []
        assert not missing.is_empty

    async def test_missing_refs_empty_when_all_exist(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_resume_batch missing_refs is empty when all refs exist."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()

        tool_id = await store.register_tool(
            name="calc", schema_json="{}", risk_class="low"
        )
        template_id = await store.register_template(
            path="x.jinja2", content="X", content_hash="h1"
        )
        await store.register_repo_snapshot(
            target_project_root="/root",
            repo_map_json="{}",
            project_info_json=None,
            repo_snapshot_hash="abc123",
        )

        context_snapshot_json = json.dumps(
            {
                "id": f"snap_{turn_id}",
                "schema_version": 1,
                "tool_set_ref": {
                    "tool_ids": [tool_id],
                    "tool_names": ["calc"],
                    "version": "1.0.0",
                },
                "template_set_ref": {
                    "template_ids": [template_id],
                    "template_paths": ["x.jinja2"],
                    "version": "1.0.0",
                },
                "repo_snapshot_ref": {
                    "target_project_root": "/root",
                    "repo_snapshot_hash": "abc123",
                },
                "content_hash": "hash",
                "created_at_ms": 1000,
            }
        )

        frame_id = _new_id()
        await _fire_session_start(store, session_id, turn_id, user_input="hi")
        await _fire_turn_start(
            store,
            session_id,
            turn_id,
            turn_index=0,
            user_input="hi",
            context_snapshot_json=context_snapshot_json,
        )
        await _fire_frame_start(store, session_id, turn_id, frame_id, iteration=0)
        await _fire_frame_complete(
            store, session_id, turn_id, frame_id, status="completed"
        )
        await _fire_turn_complete(store, session_id, turn_id, status="completed")

        batch = await store.load_resume_batch(session_id, turn_id=turn_id)
        assert batch is not None
        assert batch.missing_refs.is_empty


class TestLoadReplayBatch:
    async def test_replay_batch_roundtrip_structure(
        self, execution_store: ExecutionStore
    ) -> None:
        """Full session -> close -> reopen -> load_replay_batch verifies SessionReplay."""
        store1 = execution_store
        db_path = store1.db_path

        session_id = _new_id()
        turn_ids = []
        frame_ids = []

        for turn_index in range(2):
            turn_id = _new_id()
            frame_id = _new_id()
            turn_ids.append(turn_id)
            frame_ids.append(frame_id)

            context_snapshot_json = json.dumps(
                {
                    "id": f"snap_{turn_id}",
                    "schema_version": 1,
                    "tool_set_ref": {
                        "tool_ids": [],
                        "tool_names": [],
                        "version": "1.0.0",
                    },
                    "template_set_ref": {
                        "template_ids": [],
                        "template_paths": [],
                        "version": "1.0.0",
                    },
                    "content_hash": f"snap_hash_{turn_index}",
                    "created_at_ms": 1000,
                }
            )

            await _fire_session_start(
                store1, session_id, turn_id, user_input=f"turn {turn_index}"
            )
            await _fire_turn_start(
                store1,
                session_id,
                turn_id,
                turn_index=turn_index,
                user_input=f"turn {turn_index}",
                context_snapshot_json=context_snapshot_json,
            )
            await _fire_frame_start(store1, session_id, turn_id, frame_id, iteration=0)
            await _fire_llm_complete(
                store1,
                session_id,
                turn_id,
                frame_id,
                content=f"response {turn_index}",
                input_tokens=15,
                output_tokens=10,
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
                total_input_tokens=0,
                total_output_tokens=0,
            )
            await _fire_turn_complete(store1, session_id, turn_id, status="completed")

        await store1.close()

        store2 = ExecutionStore(db_path)
        await store2.init()
        try:
            replay = await store2.load_replay_batch(session_id)
            assert replay is not None
            assert replay.session.id == session_id
            assert len(replay.turns) == 2

            for turn_replay in replay.turns:
                assert turn_replay.turn.id in turn_ids
                assert turn_replay.turn.session_id == session_id
                assert turn_replay.snapshot is not None
                assert len(turn_replay.frames) == 1
                assert turn_replay.frames[0].id in frame_ids
                assert turn_replay.frames[0].turn_id == turn_replay.turn.id
                assert turn_replay.frames[0].id in turn_replay.frame_events
                assert len(turn_replay.frame_events[turn_replay.frames[0].id]) >= 1
                assert len(turn_replay.messages) >= 1
                assert turn_replay.missing_refs.is_empty is True
        finally:
            await store2.close()


class TestLoadTurnReplay:
    async def test_turn_replay_single_turn_graph(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_turn_replay returns a single turn's full durable graph."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        frame_id = _new_id()
        snapshot_id = _new_id()

        context_snapshot_json = json.dumps(
            {
                "id": snapshot_id,
                "schema_version": 1,
                "tool_set_ref": {"tool_ids": [], "tool_names": [], "version": "1.0.0"},
                "template_set_ref": {
                    "template_ids": [],
                    "template_paths": [],
                    "version": "1.0.0",
                },
                "content_hash": "snap_hash",
                "created_at_ms": 1000,
            }
        )

        await _fire_session_start(store, session_id, turn_id, user_input="hi")
        await _fire_turn_start(
            store,
            session_id,
            turn_id,
            turn_index=0,
            user_input="hi",
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
            store, session_id, turn_id, frame_id, input_tokens=6, output_tokens=4
        )
        await _fire_frame_complete(
            store, session_id, turn_id, frame_id, status="completed"
        )
        await _fire_turn_complete(store, session_id, turn_id, status="completed")

        replay = await store.load_turn_replay(turn_id)
        assert replay is not None
        assert replay.turn.id == turn_id
        assert replay.turn.session_id == session_id
        assert replay.snapshot is not None
        assert replay.snapshot.id == snapshot_id
        assert len(replay.frames) == 1
        assert replay.frames[0].id == frame_id
        assert frame_id in replay.frame_events
        assert len(replay.frame_events[frame_id]) >= 1
        assert len(replay.messages) >= 1
        assert replay.missing_refs.is_empty is True

        nonexistent = await store.load_turn_replay("turn_nope")
        assert nonexistent is None


class TestFrameReconstruction:
    async def test_build_frame_restores_actions_tool_calls_result(
        self, execution_store: ExecutionStore
    ) -> None:
        """build_frame reconstructs actions, tool calls, and result from stored events."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        frame_id = _new_id()

        await _fire_session_start(store, session_id, turn_id, user_input="hi")
        await _fire_turn_start(
            store, session_id, turn_id, turn_index=0, user_input="hi"
        )
        await _fire_frame_start(store, session_id, turn_id, frame_id, iteration=0)
        await store.flush()
        # Clear adapter pending state so close() does not re-write already-flushed events
        store._event_adapter._pending.clear()

        ts_base = int(datetime.now(UTC).timestamp() * 1000)
        await store.append_event(
            trace_id=_new_id(),
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=1,
            event_type=EventType.LLM_COMPLETE,
            data_json=json.dumps(
                {
                    "content": "hello there",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "read_file",
                            "arguments": {"path": "a.txt"},
                        }
                    ],
                    "model": "gpt-4",
                    "input_tokens": 25,
                    "output_tokens": 0,
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "latency_ms": 100.0,
                }
            ),
            timestamp_ms=ts_base + 100,
        )
        await store.append_event(
            trace_id=_new_id(),
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=2,
            event_type=EventType.TOOL_START,
            data_json=json.dumps(
                {
                    "call_id": "call_1",
                    "tool_id": "read_file",
                    "arguments": {"path": "a.txt"},
                }
            ),
            timestamp_ms=ts_base + 200,
        )
        await store.append_event(
            trace_id=_new_id(),
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=3,
            event_type=EventType.TOOL_RESULT,
            data_json=json.dumps(
                {
                    "call_id": "call_1",
                    "tool_id": "read_file",
                    "output": "file content",
                    "success": True,
                }
            ),
            timestamp_ms=ts_base + 300,
        )
        await store.append_event(
            trace_id=_new_id(),
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=4,
            event_type=EventType.FRAME_COMPLETE,
            data_json=json.dumps(
                {
                    "final_answer": "done",
                    "finish_reason": "stop",
                    "duration_ms": 200,
                    "total_input_tokens": 25,
                    "total_output_tokens": 0,
                }
            ),
            timestamp_ms=ts_base + 400,
        )
        await store.flush()

        # Update the stored frame row to completed so build_frame reads the final status
        await store.update_frame_status(
            frame_id,
            ExecutionStatus.COMPLETED,
            finished_at=datetime.fromtimestamp((ts_base + 400) / 1000, UTC),
            duration_ms=200,
            input_tokens=15,
            output_tokens=10,
            finish_reason="stop",
        )

        stored = await store.frame_repo.get(frame_id)
        assert stored is not None
        assert stored.id == frame_id
        assert stored.status == ExecutionStatus.COMPLETED.value

        events = await store.event_repo.get_by_frame(frame_id)
        assert len(events) == 5
        assert [e["type"] for e in events] == [
            "frame_start",
            "llm_complete",
            "tool_start",
            "tool_result",
            "frame_complete",
        ]

        frame = build_frame(stored, events)

        assert frame.id == frame_id
        assert frame.session_id == session_id
        assert frame.turn_id == turn_id
        assert frame.iteration == 0
        assert frame.status == ExecutionStatus.COMPLETED
        assert frame.model_response_raw == "hello there"

        assert len(frame.actions) == 2
        assert frame.actions[0].kind.value == "llm_message"
        assert frame.actions[0].tool_calls[0].name == "read_file"
        assert frame.actions[1].kind.value == "tool_call"
        assert frame.actions[1].tool_execution_data["status"] == "completed"
        assert frame.actions[1].tool_execution_data["output"] == "file content"

        assert frame.result is not None
        assert frame.result.final_answer == "done"
        assert frame.result.finish_reason == "stop"


class TestContextSnapshotToRuntime:
    async def test_to_runtime_parses_json_refs(
        self, execution_store: ExecutionStore
    ) -> None:
        """StoredContextSnapshot.to_runtime parses JSON refs correctly."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()

        await _fire_session_start(store, session_id, turn_id, user_input="hi")
        await _fire_turn_start(
            store, session_id, turn_id, turn_index=0, user_input="hi"
        )
        await store.flush()

        snapshot_id = _new_id()
        tool_id = await store.register_tool(
            name="calc", schema_json="{}", risk_class="low"
        )
        template_id = await store.register_template(
            path="x.jinja2", content="X", content_hash="h1"
        )
        await store.register_repo_snapshot(
            target_project_root="/root",
            repo_map_json="{}",
            project_info_json=None,
            repo_snapshot_hash="abc123",
        )

        await store.create_context_snapshot(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            tool_set_ref=json.dumps(
                {"tool_ids": [tool_id], "tool_names": ["calc"], "version": "1.0.0"}
            ),
            template_set_ref=json.dumps(
                {
                    "template_ids": [template_id],
                    "template_paths": ["x.jinja2"],
                    "version": "1.0.0",
                }
            ),
            repo_snapshot_ref=json.dumps(
                {"target_project_root": "/root", "repo_snapshot_hash": "abc123"}
            ),
            content_hash="hash",
            created_at_ms=1000,
        )

        stored = await store.context_snapshot_repo.get_stored(snapshot_id)
        assert stored is not None

        runtime = stored.to_runtime()
        assert runtime.id == snapshot_id
        assert runtime.turn_id == turn_id
        assert runtime.content_hash == "hash"
        assert runtime.tool_set_ref.tool_ids == [tool_id]
        assert runtime.tool_set_ref.tool_names == ["calc"]
        assert runtime.template_set_ref.template_ids == [template_id]
        assert runtime.template_set_ref.template_paths == ["x.jinja2"]
        assert runtime.repo_snapshot_ref is not None
        assert runtime.repo_snapshot_ref.target_project_root == "/root"
        assert runtime.repo_snapshot_ref.repo_snapshot_hash == "abc123"

    async def test_to_runtime_handles_null_and_none_refs(
        self, execution_store: ExecutionStore
    ) -> None:
        """to_runtime normalizes "null"/NULL ref columns to empty refs."""
        store = execution_store
        turn_id = _new_id()
        session_id = _new_id()

        await _fire_session_start(store, session_id, turn_id, user_input="hi")
        await _fire_turn_start(
            store, session_id, turn_id, turn_index=0, user_input="hi"
        )
        await store.flush()

        # Literal "null" string stored as a ref column
        snapshot_null = _new_id()
        await store.create_context_snapshot(
            snapshot_id=snapshot_null,
            turn_id=turn_id,
            tool_set_ref="null",
            template_set_ref='""',
            repo_snapshot_ref=None,
            content_hash="hash",
            created_at_ms=1000,
        )

        stored_null = await store.context_snapshot_repo.get_stored(snapshot_null)
        assert stored_null is not None
        runtime_null = stored_null.to_runtime()
        assert runtime_null.tool_set_ref.tool_ids == []
        assert runtime_null.tool_set_ref.tool_names == []
        assert runtime_null.template_set_ref.template_ids == []
        assert runtime_null.repo_snapshot_ref is None

        # SQL NULL ref column
        snapshot_none = _new_id()
        await store.create_context_snapshot(
            snapshot_id=snapshot_none,
            turn_id=turn_id,
            tool_set_ref=None,
            template_set_ref=None,
            repo_snapshot_ref=None,
            content_hash="hash",
            created_at_ms=1000,
        )

        stored_none = await store.context_snapshot_repo.get_stored(snapshot_none)
        assert stored_none is not None
        runtime_none = stored_none.to_runtime()
        assert runtime_none.tool_set_ref.tool_ids == []
        assert runtime_none.template_set_ref.template_ids == []
        assert runtime_none.repo_snapshot_ref is None
