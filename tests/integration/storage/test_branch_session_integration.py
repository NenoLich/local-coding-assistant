"""Integration tests for ExecutionStore.branch_session.

Covers fork independence, foreign-key remapping, parent isolation,
file baseline remapping, branch provenance metadata, multiple forks
from the same parent, and close -> reopen round-trip persistence of a
fork (including load_resume_batch resumability).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from local_coding_assistant.runtime.engine.frame import ExecutionStatus
from local_coding_assistant.runtime.events.contracts import (
    EventType,
    FrameCompleteEvent,
    FrameStartEvent,
    LLMCompleteEvent,
    SessionStartEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.storage.execution_store import ExecutionStore, SessionStatus

WORKSPACE_ID = "ws_branch_test"


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


def _context_snapshot_json(snapshot_id: str, content_hash: str) -> str:
    return json.dumps(
        {
            "id": snapshot_id,
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
            "content_hash": content_hash,
            "created_at_ms": 1000,
        }
    )


async def _seed_parent_session(
    store: ExecutionStore,
) -> tuple[str, dict[str, str]]:
    """Parent session: turn 0 completed, turn 1 interrupted (running + paused)."""
    session_id = _new_id()
    await _fire_session_start(store, session_id, _new_id(), user_input="first")

    # Turn 0: completed
    turn_0_id = _new_id()
    snap_0_id = _new_id()
    frame_0_id = _new_id()
    await _fire_turn_start(
        store,
        session_id,
        turn_0_id,
        turn_index=0,
        user_input="first",
        context_snapshot_json=_context_snapshot_json(snap_0_id, "snap_hash_0"),
    )
    await _fire_frame_start(
        store,
        session_id,
        turn_0_id,
        frame_0_id,
        iteration=0,
        context_snapshot_id=snap_0_id,
    )
    await _fire_llm_complete(
        store, session_id, turn_0_id, frame_0_id, input_tokens=25, output_tokens=0
    )
    await _fire_frame_complete(
        store, session_id, turn_0_id, frame_0_id, status="completed"
    )
    await _fire_turn_complete(store, session_id, turn_0_id, status="completed")

    # Turn 1: interrupted (running, paused)
    turn_1_id = _new_id()
    snap_1_id = _new_id()
    frame_1_id = _new_id()
    await _fire_session_start(store, session_id, turn_1_id, user_input="second")
    await _fire_turn_start(
        store,
        session_id,
        turn_1_id,
        turn_index=1,
        user_input="second",
        context_snapshot_json=_context_snapshot_json(snap_1_id, "snap_hash_1"),
    )
    await _fire_frame_start(
        store,
        session_id,
        turn_1_id,
        frame_1_id,
        iteration=0,
        context_snapshot_id=snap_1_id,
    )
    await _fire_llm_complete(
        store, session_id, turn_1_id, frame_1_id, input_tokens=10, output_tokens=0
    )
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    await store.update_turn_status(
        turn_1_id, "running", paused_at_ms=now_ms, pause_reason="interrupt"
    )
    await store.upsert_file_baseline(
        workspace_id=WORKSPACE_ID,
        observation={
            "path": "src/a.py",
            "hash": "hash_a",
            "size": 10,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": None,
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        },
        turn_id=turn_1_id,
        frame_id=frame_1_id,
        session_id=session_id,
    )

    return session_id, {
        "turn_0_id": turn_0_id,
        "snap_0_id": snap_0_id,
        "frame_0_id": frame_0_id,
        "turn_1_id": turn_1_id,
        "snap_1_id": snap_1_id,
        "frame_1_id": frame_1_id,
    }


class TestBranchFromInterruptedTurn:
    async def test_fork_inherits_running_state_and_is_resumable(
        self, execution_store: ExecutionStore
    ) -> None:
        """Fork from an interrupted turn is running, resumable, with prefix turns."""
        store = execution_store
        session_id, ids = await _seed_parent_session(store)

        forked = await store.branch_session(
            new_session_id=_new_id(),
            parent_session_id=session_id,
            branch_from_turn_id=ids["turn_1_id"],
        )

        assert forked.id != session_id
        assert forked.status == SessionStatus.RUNNING.value
        assert forked.total_turns == 2
        assert forked.total_frames == 2

        metadata = json.loads(forked.metadata_json or "{}")
        assert metadata["branched_from_session"] == session_id
        assert metadata["branched_from_turn"] == ids["turn_1_id"]

        fork_turns = await store.list_turns_by_session(forked.id)
        assert [t.turn_index for t in fork_turns] == [0, 1]

        fork_turn_0 = next(t for t in fork_turns if t.turn_index == 0)
        fork_turn_1 = next(t for t in fork_turns if t.turn_index == 1)
        assert fork_turn_0.status == "completed"
        assert fork_turn_1.id != ids["turn_1_id"]
        assert fork_turn_1.status == "running"
        assert fork_turn_1.pause_reason == "interrupt"
        assert fork_turn_1.paused_at_ms is not None

        # Resumable: load_resume_batch works on the fork.
        batch = await store.load_resume_batch(forked.id)
        assert batch is not None
        assert batch.session.id == forked.id
        assert batch.session.status == SessionStatus.RUNNING.value
        assert batch.turn.id == fork_turn_1.id
        assert batch.snapshot is not None
        assert batch.snapshot.turn_id == fork_turn_1.id
        assert len(batch.frames) == 1
        assert batch.frames[0].turn_id == fork_turn_1.id
        assert batch.frames[0].id in batch.frame_events
        assert len(batch.frame_events[batch.frames[0].id]) >= 1
        assert batch.missing_refs.is_empty is True

    async def test_all_foreign_keys_are_remapped_in_fork(
        self, execution_store: ExecutionStore
    ) -> None:
        """Every FK in the fork points at forked rows, never at parent rows."""
        store = execution_store
        session_id, ids = await _seed_parent_session(store)

        forked = await store.branch_session(
            new_session_id=_new_id(),
            parent_session_id=session_id,
            branch_from_turn_id=ids["turn_1_id"],
        )

        fork_turns = {
            t.turn_index: t for t in await store.list_turns_by_session(forked.id)
        }
        assert set(fork_turns) == {0, 1}
        assert fork_turns[0].id != ids["turn_0_id"]
        assert fork_turns[1].id != ids["turn_1_id"]

        frames = await store.frame_repo.list_by_session(forked.id)
        assert len(frames) == 2
        assert {f.turn_id for f in frames} == {fork_turns[0].id, fork_turns[1].id}
        fork_frame_1 = next(f for f in frames if f.turn_id == fork_turns[1].id)
        assert fork_frame_1.id != ids["frame_1_id"]
        assert fork_frame_1.iteration == 0
        assert fork_frame_1.status == ExecutionStatus.RUNNING_TOOLS.value

        fork_snap_1 = await store.context_snapshot_repo.get_by_turn(fork_turns[1].id)
        assert fork_snap_1 is not None
        assert fork_snap_1.id != ids["snap_1_id"]
        assert fork_frame_1.context_snapshot_id == fork_snap_1.id
        assert fork_turns[1].context_snapshot_id == fork_snap_1.id

        events = await store.event_repo.get_by_frame(fork_frame_1.id)
        assert len(events) == 2
        assert {e["type"] for e in events} == {
            EventType.FRAME_START.value,
            EventType.LLM_COMPLETE.value,
        }
        assert all(e["session_id"] == forked.id for e in events)
        assert all(e["turn_id"] == fork_turns[1].id for e in events)
        assert all(e["frame_id"] == fork_frame_1.id for e in events)

        messages = await store.message_repo.get_by_session(forked.id)
        assert len(messages) == 4
        assert {m["turn_id"] for m in messages} == {fork_turns[0].id, fork_turns[1].id}
        assert all(m["session_id"] == forked.id for m in messages)
        frame_ids = {f.id for f in frames}
        for m in messages:
            if m["frame_id"] != "turn_start":
                assert m["frame_id"] in frame_ids

    async def test_parent_session_is_left_untouched(
        self, execution_store: ExecutionStore
    ) -> None:
        """Branching does not mutate the parent session or its rows."""
        store = execution_store
        session_id, ids = await _seed_parent_session(store)

        parent_turns_before = await store.list_turns_by_session(session_id)
        parent_frames_before = await store.frame_repo.list_by_session(session_id)
        parent_events_before = await store.event_repo.get_by_session(session_id)
        parent_messages_before = await store.message_repo.get_by_session(session_id)

        await store.branch_session(
            new_session_id=_new_id(),
            parent_session_id=session_id,
            branch_from_turn_id=ids["turn_1_id"],
        )

        parent = await store.get_session(session_id)
        assert parent is not None
        assert parent.status == SessionStatus.RUNNING.value
        assert await store.list_turns_by_session(session_id) == parent_turns_before
        assert (
            await store.frame_repo.list_by_session(session_id) == parent_frames_before
        )
        assert await store.event_repo.get_by_session(session_id) == parent_events_before
        assert (
            await store.message_repo.get_by_session(session_id)
            == parent_messages_before
        )
        parent_turns = await store.list_turns_by_session(session_id)
        assert {t.id for t in parent_turns} == {ids["turn_0_id"], ids["turn_1_id"]}

    async def test_fork_metadata_contains_branch_provenance(
        self, execution_store: ExecutionStore
    ) -> None:
        """Fork metadata records the parent session and branch turn."""
        store = execution_store
        session_id, ids = await _seed_parent_session(store)

        forked = await store.branch_session(
            new_session_id=_new_id(),
            parent_session_id=session_id,
            branch_from_turn_id=ids["turn_1_id"],
        )

        metadata = json.loads(forked.metadata_json or "{}")
        assert metadata["branched_from_session"] == session_id
        assert metadata["branched_from_turn"] == ids["turn_1_id"]


class TestBranchFromCompletedTurn:
    async def test_fork_stays_idle_with_prefix_turns_only(
        self, execution_store: ExecutionStore
    ) -> None:
        """Fork from a completed turn is idle and copies only turns up to it."""
        store = execution_store
        session_id, ids = await _seed_parent_session(store)

        forked = await store.branch_session(
            new_session_id=_new_id(),
            parent_session_id=session_id,
            branch_from_turn_id=ids["turn_0_id"],
        )

        assert forked.status == SessionStatus.IDLE.value
        assert forked.total_turns == 1
        assert forked.total_frames == 1

        fork_turns = await store.list_turns_by_session(forked.id)
        assert [t.turn_index for t in fork_turns] == [0]
        assert fork_turns[0].id != ids["turn_0_id"]

        frames = await store.frame_repo.list_by_session(forked.id)
        assert len(frames) == 1
        assert frames[0].turn_id == fork_turns[0].id
        assert frames[0].status == ExecutionStatus.COMPLETED.value


class TestFileBaselinesAndMultipleForks:
    async def test_file_baselines_remapped_to_fork_rows(
        self, execution_store: ExecutionStore
    ) -> None:
        """Baselines are re-attached to the fork's turn and frame IDs."""
        store = execution_store
        session_id, ids = await _seed_parent_session(store)

        forked = await store.branch_session(
            new_session_id=_new_id(),
            parent_session_id=session_id,
            branch_from_turn_id=ids["turn_1_id"],
        )

        fork_turns = {
            t.turn_index: t for t in await store.list_turns_by_session(forked.id)
        }
        frames = await store.frame_repo.list_by_session(forked.id)
        fork_frame_1 = next(f for f in frames if f.turn_id == fork_turns[1].id)

        cursor = await store._conn.execute(
            "SELECT turn_id, frame_id FROM workspace_file_baselines WHERE session_id = ?",
            (forked.id,),
        )
        rows = await cursor.fetchall()
        assert rows == [(fork_turns[1].id, fork_frame_1.id)]

        baselines = await store.load_file_baselines(WORKSPACE_ID, forked.id)
        assert len(baselines) == 1
        assert baselines[0]["path"] == "src/a.py"
        assert baselines[0]["hash"] == "hash_a"

    async def test_multiple_forks_from_same_parent_are_independent(
        self, execution_store: ExecutionStore
    ) -> None:
        """Two forks from the same parent diverge and never share IDs."""
        store = execution_store
        session_id, ids = await _seed_parent_session(store)

        fork_a = await store.branch_session(
            new_session_id=_new_id(),
            parent_session_id=session_id,
            branch_from_turn_id=ids["turn_0_id"],
        )
        fork_b = await store.branch_session(
            new_session_id=_new_id(),
            parent_session_id=session_id,
            branch_from_turn_id=ids["turn_1_id"],
        )

        assert fork_a.id != fork_b.id
        assert fork_a.status == SessionStatus.IDLE.value
        assert fork_b.status == SessionStatus.RUNNING.value

        turns_a = await store.list_turns_by_session(fork_a.id)
        turns_b = await store.list_turns_by_session(fork_b.id)
        assert [t.turn_index for t in turns_a] == [0]
        assert [t.turn_index for t in turns_b] == [0, 1]
        assert {t.id for t in turns_a}.isdisjoint({t.id for t in turns_b})

        frames_a = await store.frame_repo.list_by_session(fork_a.id)
        frames_b = await store.frame_repo.list_by_session(fork_b.id)
        assert {f.id for f in frames_a}.isdisjoint({f.id for f in frames_b})

        # Parent remains intact.
        parent = await store.get_session(session_id)
        assert parent.status == SessionStatus.RUNNING.value
        parent_turns = await store.list_turns_by_session(session_id)
        assert {t.id for t in parent_turns} == {ids["turn_0_id"], ids["turn_1_id"]}


class TestBranchRoundtripPersistence:
    async def test_fork_survives_close_reopen_and_resumes(
        self, execution_store: ExecutionStore
    ) -> None:
        """Fork persists across close -> reopen and load_resume_batch loads it."""
        store1 = execution_store
        db_path = store1.db_path

        session_id, ids = await _seed_parent_session(store1)
        forked = await store1.branch_session(
            new_session_id=_new_id(),
            parent_session_id=session_id,
            branch_from_turn_id=ids["turn_1_id"],
        )
        fork_session_id = forked.id
        await store1.close()

        store2 = ExecutionStore(db_path)
        await store2.init()
        try:
            forked_again = await store2.get_session(fork_session_id)
            assert forked_again is not None
            assert forked_again.status == SessionStatus.RUNNING.value
            metadata = json.loads(forked_again.metadata_json or "{}")
            assert metadata["branched_from_session"] == session_id
            assert metadata["branched_from_turn"] == ids["turn_1_id"]

            fork_turns = await store2.list_turns_by_session(fork_session_id)
            assert [t.turn_index for t in fork_turns] == [0, 1]

            batch = await store2.load_resume_batch(fork_session_id)
            assert batch is not None
            assert batch.session.id == fork_session_id
            assert batch.turn.id == next(t for t in fork_turns if t.turn_index == 1).id
            assert batch.snapshot is not None
            assert len(batch.frames) == 1
            assert batch.frames[0].id in batch.frame_events
            assert batch.missing_refs.is_empty is True
        finally:
            await store2.close()
