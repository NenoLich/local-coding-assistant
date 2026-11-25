"""Unit tests for ExecutionStore (new API).

Tests ExecutionStore delegation, transaction management, flush behavior, and
load methods using a real SQLite-backed ExecutionStore. Replaces the old
tests/unit/storage/test_execution_store.py which was written for the
previous background-writer API and asserted True in many cases.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import FrozenInstanceError, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from local_coding_assistant.runtime.engine.frame import ExecutionStatus
from local_coding_assistant.runtime.events.contracts import (
    ApprovalRequestForm,
    ApprovalRequiredEvent,
    EventType,
    FrameStartEvent,
    LLMCompleteEvent,
    SessionResumeEvent,
    SessionStartEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.storage.contracts import MissingRefs
from local_coding_assistant.storage.execution_store import ExecutionStore, SessionStatus
from local_coding_assistant.storage.store_event_adapter import StoreEventAdapter


class TestExecutionStoreBusyTimeout:
    """Tests for PRAGMA busy_timeout and retry logic in _write_batch_locked."""

    async def test_init_sets_busy_timeout_pragma(
        self, execution_store: ExecutionStore
    ) -> None:
        """init(): sets PRAGMA busy_timeout=5000 on the connection."""
        store = execution_store
        # Query the current busy_timeout value
        cursor = await store._conn.execute("PRAGMA busy_timeout")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 5000

    async def test_write_batch_locked_retries_on_database_locked(
        self, execution_store: ExecutionStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_write_batch_locked(): retries on 'database is locked' OperationalError."""

        store = execution_store
        call_count = {"count": 0}

        async def flaky_execute_batch_locked(batch: list[dict[str, Any]]) -> None:
            call_count["count"] += 1
            if call_count["count"] < 3:
                # Simulate the lock error - ensure connection is rolled back
                await store._conn.rollback()
                raise aiosqlite.OperationalError("database is locked")
            # On 3rd attempt, do nothing (success - no-op)
            await store._conn.execute("BEGIN IMMEDIATE")
            await store._conn.commit()

        monkeypatch.setattr(store, "_execute_batch_locked", flaky_execute_batch_locked)

        # Should succeed after 3 attempts (2 failures + 1 success)
        batch = [
            {
                "op": "event",
                "data": {
                    "trace_id": "t",
                    "session_id": "s",
                    "turn_id": "t",
                    "frame_id": "f",
                    "seq": 1,
                    "event_type": "FRAME_START",
                    "data_json": "{}",
                    "timestamp_ms": 1,
                },
            }
        ]
        await store._write_batch_locked(batch)
        assert call_count["count"] == 3

    async def test_write_batch_locked_raises_after_max_retries(
        self, execution_store: ExecutionStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_write_batch_locked(): raises after max retries exhausted."""

        store = execution_store
        call_count = {"count": 0}

        async def always_locked(batch: list[dict[str, Any]]) -> None:
            call_count["count"] += 1
            raise aiosqlite.OperationalError("database is locked")

        monkeypatch.setattr(store, "_execute_batch_locked", always_locked)

        batch = [
            {
                "op": "event",
                "data": {
                    "trace_id": "t",
                    "session_id": "s",
                    "turn_id": "t",
                    "frame_id": "f",
                    "seq": 1,
                    "event_type": "FRAME_START",
                    "data_json": "{}",
                    "timestamp_ms": 1,
                },
            }
        ]

        with pytest.raises(aiosqlite.OperationalError, match="database is locked"):
            await store._write_batch_locked(batch)

        # Should attempt max_retries (3) times
        assert call_count["count"] == 3

    async def test_write_batch_locked_non_locked_error_raises_immediately(
        self, execution_store: ExecutionStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_write_batch_locked(): non-lock errors raised immediately without retry."""

        store = execution_store
        call_count = {"count": 0}

        async def other_error(batch: list[dict[str, Any]]) -> None:
            call_count["count"] += 1
            raise aiosqlite.OperationalError("disk I/O error")

        monkeypatch.setattr(store, "_execute_batch_locked", other_error)

        batch = [
            {
                "op": "event",
                "data": {
                    "trace_id": "t",
                    "session_id": "s",
                    "turn_id": "t",
                    "frame_id": "f",
                    "seq": 1,
                    "event_type": "FRAME_START",
                    "data_json": "{}",
                    "timestamp_ms": 1,
                },
            }
        ]

        with pytest.raises(aiosqlite.OperationalError, match="disk I/O error"):
            await store._write_batch_locked(batch)

        # Should only attempt once, no retries
        assert call_count["count"] == 1

    async def test_write_batch_locked_rollback_on_error(
        self, execution_store: ExecutionStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_write_batch_locked(): rolls back on error before re-raising."""

        store = execution_store
        rollback_called = {"called": False}

        async def failing_batch(batch: list[dict[str, Any]]) -> None:
            # Simulate failure after BEGIN IMMEDIATE
            await store._conn.execute("BEGIN IMMEDIATE")
            raise aiosqlite.OperationalError("disk I/O error")

        original_rollback = store._conn.rollback

        async def tracking_rollback() -> None:
            rollback_called["called"] = True
            await original_rollback()

        monkeypatch.setattr(store, "_execute_batch_locked", failing_batch)
        monkeypatch.setattr(store._conn, "rollback", tracking_rollback)

        batch = [
            {
                "op": "event",
                "data": {
                    "trace_id": "t",
                    "session_id": "s",
                    "turn_id": "t",
                    "frame_id": "f",
                    "seq": 1,
                    "event_type": "FRAME_START",
                    "data_json": "{}",
                    "timestamp_ms": 1,
                },
            }
        ]

        with pytest.raises(aiosqlite.OperationalError, match="disk I/O error"):
            await store._write_batch_locked(batch)

        assert rollback_called["called"] is True


def _new_id() -> str:
    return str(uuid.uuid4())


async def _make_session(
    store: ExecutionStore,
    session_id: str | None = None,
    user_input: str = "query",
    agent_mode: str = "frame_agent",
    model_name: str = "gpt-4",
) -> str:
    """Create a session via create_session (always status='running')."""
    session_id = session_id or _new_id()
    await store.create_session(
        session_id=session_id,
        user_input=user_input,
        agent_mode=agent_mode,
        model_name=model_name,
    )
    session = await store.get_session(session_id)
    assert session is not None
    assert session.status == SessionStatus.RUNNING.value
    return session_id


async def _make_idle_session(
    store: ExecutionStore,
    session_id: str | None = None,
    user_input: str = "query",
) -> str:
    """Create an idle session directly via session_repo.upsert."""
    session_id = session_id or _new_id()
    await store.session_repo.upsert(
        session_id=session_id,
        user_input=user_input,
        status="idle",
        started_at_ms=1000,
        agent_mode="frame_agent",
        model_name="gpt-4",
    )
    return session_id


async def _make_turn(
    store: ExecutionStore,
    session_id: str,
    turn_id: str | None = None,
    turn_index: int = 0,
    user_input: str = "turn query",
    status: str = "running",
    config_snapshot_json: str = "{}",
    final_answer: str | None = None,
) -> str:
    turn_id = turn_id or _new_id()
    await store.create_turn(
        turn_id=turn_id,
        session_id=session_id,
        trace_id=_new_id(),
        turn_index=turn_index,
        user_input=user_input,
        status=status,
        config_snapshot_json=config_snapshot_json,
        final_answer=final_answer,
    )
    turn = await store.get_turn(turn_id)
    assert turn is not None
    assert turn.status == status
    return turn_id


async def _make_frame(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str | None = None,
    iteration: int = 0,
    status: str = "running_llm",
    model_name: str | None = "gpt-4",
    context_snapshot_id: str | None = None,
    history_through_seq: int = 0,
    prompt_hash: str | None = None,
) -> str:
    frame_id = frame_id or _new_id()
    await store.create_frame(
        frame_id=frame_id,
        session_id=session_id,
        turn_id=turn_id,
        trace_id=_new_id(),
        iteration=iteration,
        model_name=model_name,
        context_snapshot_id=context_snapshot_id,
        history_through_seq=history_through_seq,
        prompt_hash=prompt_hash,
    )
    frame = await store.frame_repo.get(frame_id)
    assert frame is not None
    assert frame.status == status
    return frame_id


class TestExecutionStoreLifecycle:
    async def test_init_creates_tables_and_initializes_all_repos(
        self, execution_store: ExecutionStore
    ) -> None:
        """init(): creates tables and initializes all 10 repositories."""
        store = execution_store
        assert store._conn is not None
        assert store.event_repo is not None
        assert store.frame_repo is not None
        assert store.session_repo is not None
        assert store.message_repo is not None
        assert store.tool_repo is not None
        assert store.template_repo is not None
        assert store.turn_repo is not None
        assert store.context_snapshot_repo is not None
        assert store.repo_snapshot_repo is not None
        assert store.workspace_file_baseline_repo is not None
        assert store.workspace_repo is not None

    async def test_close_flushes_and_closes(
        self, execution_store: ExecutionStore
    ) -> None:
        """close(): flushes adapter and closes the connection."""
        store = execution_store
        await store.process_event(
            SessionStartEvent(
                trace_id=_new_id(),
                turn_id=_new_id(),
                session_id=_new_id(),
                user_input="x",
                agent_mode="frame_agent",
                model_name="gpt-4",
            )
        )
        await store.close()


class TestExecutionStorePeriodicFlush:
    """Tests for the background flush loop started from ExecutionStore.init()."""

    async def test_periodic_flush_persists_non_terminal_event(
        self, tmp_path: Path
    ) -> None:
        """Periodic flush writes buffered non-terminal events within flush_interval_ms.

        ApprovalRequiredEvent is not in _should_flush, so process_event buffers
        it in the adapter without writing to SQLite immediately. The background
        flush loop (flush_interval_ms=50) should still persist it within ~1s,
        without waiting for a flush-triggering event or close().

        Events are routed through the adapter in production order
        (session -> turn -> frame -> approval) so the frame upsert has the
        iteration/started_at_ms from FrameStartEvent to merge against.
        """
        store = ExecutionStore(tmp_path / "flush.db", flush_interval_ms=50)
        await store.init()
        try:
            session_id = _new_id()
            turn_id = _new_id()
            frame_id = _new_id()
            trace_id = _new_id()

            await store.process_event(
                SessionStartEvent(
                    trace_id=trace_id,
                    turn_id=turn_id,
                    session_id=session_id,
                    user_input="hello agent",
                    agent_mode="frame_agent",
                    model_name="gpt-4",
                )
            )
            await store.process_event(
                TurnStartEvent(
                    trace_id=trace_id,
                    turn_id=turn_id,
                    session_id=session_id,
                    user_input="hello agent",
                    turn_index=0,
                )
            )
            await store.process_event(
                FrameStartEvent(
                    trace_id=trace_id,
                    turn_id=turn_id,
                    frame_id=frame_id,
                    session_id=session_id,
                    iteration=1,
                    context_snapshot_id="snap_1",
                    context_hash="sha256:abc",
                )
            )
            # ApprovalRequiredEvent is NOT in _should_flush -> buffered only.
            await store.process_event(
                ApprovalRequiredEvent(
                    trace_id=trace_id,
                    turn_id=turn_id,
                    frame_id=frame_id,
                    session_id=session_id,
                    approval_requests=[
                        ApprovalRequestForm(
                            approval_token="test-token",  # noqa: S106 - test fixture token, not a real secret
                            tool_call_id="call_1",
                            tool_name="write_file",
                            arguments={"path": "/tmp/foo.txt"},  # noqa: S108 - test fixture path, not a real temp dir
                        )
                    ],
                )
            )

            # Not yet durable before the periodic flush fires.
            events_before = await store.event_repo.get_by_turn(turn_id)
            assert events_before == []

            # Poll until the background flush loop persists the event.
            events_after = []
            for _ in range(50):  # up to ~1s
                events_after = await store.event_repo.get_by_turn(turn_id)
                if events_after:
                    break
                await asyncio.sleep(0.02)

            assert len(events_after) == 4
            approval_events = [
                e
                for e in events_after
                if e["type"] == EventType.APPROVAL_REQUIRED.value
            ]
            assert len(approval_events) == 1
            assert approval_events[0]["frame_id"] == frame_id
        finally:
            await store.close()

    async def test_flush_task_started_and_stopped(
        self, execution_store: ExecutionStore
    ) -> None:
        """init() starts the background flush task; close() cancels it."""
        store = execution_store
        assert store._flush_task is not None
        assert not store._flush_task.done()
        await store.close()
        assert store._flush_task.done()


class TestExecutionStoreRegistration:
    async def test_register_tool_delegates_and_commits(
        self, execution_store: ExecutionStore
    ) -> None:
        """register_tool(): delegates to tool_repo and commits."""
        store = execution_store
        tool_id = await store.register_tool(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="low",
            description="Read a file",
            version="1.0.0",
        )
        assert isinstance(tool_id, int)
        assert tool_id > 0

        tool = await store.tool_repo.get(tool_id)
        assert tool is not None
        assert tool["name"] == "read_file"

    async def test_register_tool_deduplicates_by_content_hash(
        self, execution_store: ExecutionStore
    ) -> None:
        """register_tool(): re-registering identical spec returns same id."""
        store = execution_store
        id_1 = await store.register_tool(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="low",
            version="1.0.0",
        )
        id_2 = await store.register_tool(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="low",
            version="1.0.0",
        )
        assert id_2 == id_1

    async def test_register_template_delegates_and_commits(
        self, execution_store: ExecutionStore
    ) -> None:
        """register_template(): delegates to template_repo and commits."""
        store = execution_store
        template_id = await store.register_template(
            path="tmpl.jinja2",
            content="Hi {{ name }}",
            content_hash="abc123",
            version="1.0.0",
        )
        assert isinstance(template_id, int)
        assert template_id > 0

        template = await store.template_repo.get(template_id)
        assert template is not None
        assert template["path"] == "tmpl.jinja2"

    async def test_register_template_deduplicates_by_content_hash(
        self, execution_store: ExecutionStore
    ) -> None:
        """register_template(): re-registering identical spec returns same id."""
        store = execution_store
        id_1 = await store.register_template(
            path="tmpl.jinja2",
            content="Hi {{ name }}",
            content_hash="abc123",
        )
        id_2 = await store.register_template(
            path="tmpl.jinja2",
            content="Hi {{ name }}",
            content_hash="abc123",
        )
        assert id_2 == id_1


class TestExecutionStoreSession:
    async def test_create_session_inserts_and_commits(
        self, execution_store: ExecutionStore
    ) -> None:
        """create_session(): inserts session row and commits immediately."""
        store = execution_store
        session_id = _new_id()
        await store.create_session(
            session_id=session_id,
            user_input="hello",
            agent_mode="no_agent",
            model_name="gpt-4",
            metadata_json='{"k": "v"}',
        )
        session = await store.get_session(session_id)
        assert session is not None
        assert session.id == session_id
        assert session.user_input == "hello"
        assert session.agent_mode == "no_agent"
        assert session.model_name == "gpt-4"
        assert session.metadata_json == '{"k": "v"}'
        assert session.status == SessionStatus.RUNNING.value
        assert session.total_input_tokens == 0
        assert session.total_output_tokens == 0
        assert session.total_turns == 0
        assert session.total_frames == 0

    async def test_get_session_returns_none_for_missing(
        self, execution_store: ExecutionStore
    ) -> None:
        """get_session(): returns None for unknown session id."""
        store = execution_store
        assert await store.get_session(_new_id()) is None

    async def test_associate_session_tools_insert_or_ignore_commits(
        self, execution_store: ExecutionStore
    ) -> None:
        """associate_session_tools(): INSERT OR IGNORE, commits."""
        store = execution_store
        session_id = await _make_session(store)
        tool_id = await store.register_tool(
            name="tool_a", schema_json="{}", risk_class="low"
        )
        await store.associate_session_tools(session_id, [tool_id])
        await store.associate_session_tools(session_id, [tool_id])  # idempotent
        assoc = await store.tool_repo.get_session_tools(session_id)
        assert len(assoc) == 1
        assert assoc[0]["id"] == tool_id

    async def test_associate_session_templates_insert_or_ignore_commits(
        self, execution_store: ExecutionStore
    ) -> None:
        """associate_session_templates(): INSERT OR IGNORE, commits."""
        store = execution_store
        session_id = await _make_session(store)
        template_id = await store.register_template(
            path="t.jinja2", content="c", content_hash="h"
        )
        await store.associate_session_templates(session_id, [template_id])
        assoc = await store.template_repo.get_session_templates(session_id)
        assert len(assoc) == 1
        assert assoc[0]["id"] == template_id

    async def test_list_sessions_filters_by_status(
        self, execution_store: ExecutionStore
    ) -> None:
        """list_sessions(): optional status filter applied in Python."""
        store = execution_store
        await _make_session(store, user_input="running one")
        await _make_idle_session(store, user_input="idle one")

        running = await store.list_sessions(status="running")
        assert len(running) == 1
        assert all(s.status == "running" for s in running)

        idle = await store.list_sessions(status="idle")
        assert len(idle) == 1
        assert all(s.status == "idle" for s in idle)


class TestExecutionStoreTurn:
    async def test_create_turn_inserts_and_commits(
        self, execution_store: ExecutionStore
    ) -> None:
        """create_turn(): inserts turn row and commits immediately."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id, turn_index=2)
        turn = await store.get_turn(turn_id)
        assert turn is not None
        assert turn.session_id == session_id
        assert turn.turn_index == 2
        assert turn.user_input == "turn query"
        assert turn.config_snapshot_json == "{}"
        assert turn.final_answer is None

    async def test_create_turn_with_final_answer(
        self, execution_store: ExecutionStore
    ) -> None:
        """create_turn(): persists final_answer on the turn row."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(
            store, session_id, turn_index=2, final_answer="The answer is 42"
        )
        turn = await store.get_turn(turn_id)
        assert turn is not None
        assert turn.final_answer == "The answer is 42"

    async def test_update_turn_status_delegates_and_commits(
        self, execution_store: ExecutionStore
    ) -> None:
        """update_turn_status(): delegates to turn_repo and commits."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id)
        await store.update_turn_status(
            turn_id=turn_id,
            status="completed",
            finished_at_ms=5000,
            total_input_tokens=60,
            total_output_tokens=40,
            total_frames=3,
        )
        turn = await store.get_turn(turn_id)
        assert turn.status == "completed"
        assert turn.finished_at_ms == 5000
        assert turn.total_input_tokens == 60
        assert turn.total_output_tokens == 40
        assert turn.total_frames == 3
        assert turn.final_answer is None

    async def test_update_turn_status_with_final_answer(
        self, execution_store: ExecutionStore
    ) -> None:
        """update_turn_status(): persists final_answer."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id)
        await store.update_turn_status(
            turn_id=turn_id,
            status="completed",
            finished_at_ms=5000,
            final_answer="The answer is 42",
        )
        turn = await store.get_turn(turn_id)
        assert turn.final_answer == "The answer is 42"

    async def test_get_last_turn_returns_highest_index(
        self, execution_store: ExecutionStore
    ) -> None:
        """get_last_turn(): returns the turn with the highest turn_index."""
        store = execution_store
        session_id = await _make_session(store)
        await _make_turn(store, session_id, turn_index=0)
        await _make_turn(store, session_id, turn_index=1)
        last = await store.get_last_turn(session_id)
        assert last is not None
        assert last.turn_index == 1

    async def test_list_turns_by_session_ordered_by_index(
        self, execution_store: ExecutionStore
    ) -> None:
        """list_turns_by_session(): ordered by turn_index."""
        store = execution_store
        session_id = await _make_session(store)
        await _make_turn(store, session_id, turn_index=2)
        await _make_turn(store, session_id, turn_index=0)
        await _make_turn(store, session_id, turn_index=1)
        turns = await store.list_turns_by_session(session_id)
        assert [t.turn_index for t in turns] == [0, 1, 2]

    async def test_get_frame_ids_by_turn_returns_ids(
        self, execution_store: ExecutionStore
    ) -> None:
        """get_frame_ids_by_turn(): delegates to frame_repo.list_ids_by_turn."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id)
        f1 = await _make_frame(store, session_id, turn_id, iteration=0)
        f2 = await _make_frame(store, session_id, turn_id, iteration=1)
        ids = await store.get_frame_ids_by_turn(turn_id)
        assert set(ids) == {f1, f2}


class TestExecutionStoreFrame:
    async def test_create_frame_inserts_and_commits(
        self, execution_store: ExecutionStore
    ) -> None:
        """create_frame(): inserts frame row and commits immediately."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id)
        frame_id = await _make_frame(
            store,
            session_id,
            turn_id,
            iteration=1,
            context_snapshot_id="snap_1",
            history_through_seq=5,
            prompt_hash="sha:abc",
        )
        frame = await store.frame_repo.get(frame_id)
        assert frame.context_snapshot_id == "snap_1"
        assert frame.history_through_seq == 5
        assert frame.prompt_hash == "sha:abc"

    async def test_update_frame_status_delegates_and_commits(
        self, execution_store: ExecutionStore
    ) -> None:
        """update_frame_status(): delegates to frame_repo and commits."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id)
        frame_id = await _make_frame(store, session_id, turn_id)
        finished = datetime.now(UTC)
        await store.update_frame_status(
            frame_id=frame_id,
            status=ExecutionStatus.COMPLETED,
            finished_at=finished,
            duration_ms=400,
            input_tokens=42,
            output_tokens=0,
            finish_reason="stop",
            error_message=None,
            pending_call_id=None,
        )
        frame = await store.frame_repo.get(frame_id)
        assert frame.status == ExecutionStatus.COMPLETED.value
        assert frame.finished_at_ms == int(finished.timestamp() * 1000)
        assert frame.duration_ms == 400
        assert frame.input_tokens == 42
        assert frame.output_tokens == 0
        assert frame.finish_reason == "stop"


class TestExecutionStoreContextSnapshot:
    async def test_create_context_snapshot_inserts_and_commits(
        self, execution_store: ExecutionStore
    ) -> None:
        """create_context_snapshot(): inserts snapshot and commits."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id)
        snapshot_id = _new_id()
        await store.create_context_snapshot(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            tool_set_ref='{"tool_ids": [1, 2]}',
            template_set_ref='{"template_ids": [3]}',
            content_hash="hash123",
            created_at_ms=1234,
        )
        snapshot = await store.get_context_snapshot(snapshot_id)
        assert snapshot is not None
        assert snapshot.id == snapshot_id
        assert snapshot.turn_id == turn_id
        assert snapshot.content_hash == "hash123"
        assert snapshot.created_at_ms == 1234

    async def test_get_context_snapshot_by_turn_delegates(
        self, execution_store: ExecutionStore
    ) -> None:
        """get_context_snapshot_by_turn(): delegates to repo."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id)
        snapshot_id = _new_id()
        await store.create_context_snapshot(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            content_hash="hash123",
        )
        snapshot = await store.get_context_snapshot_by_turn(turn_id)
        assert snapshot is not None
        assert snapshot.id == snapshot_id


class TestExecutionStoreMessageEvent:
    async def test_append_message_delegates_and_commits(
        self, execution_store: ExecutionStore
    ) -> None:
        """append_message(): delegates to message_repo and commits."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id)
        frame_id = await _make_frame(store, session_id, turn_id)
        await store.append_message(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=1,
            role="user",
            content="Hello",
            tool_call_id=None,
            tool_name=None,
            dedupe_key="user:" + turn_id,
        )
        messages = await store.message_repo.get_by_session(session_id)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    async def test_append_message_seq_none_computes_max_plus_one(
        self, execution_store: ExecutionStore
    ) -> None:
        """append_message(): seq=None computed as MAX(seq)+1 per session."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id)
        frame_id = await _make_frame(store, session_id, turn_id)
        await store.append_message(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="user",
            content="First",
            dedupe_key="a",
        )
        await store.append_message(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="user",
            content="Second",
            dedupe_key="b",
        )
        messages = await store.message_repo.get_by_session(session_id)
        assert len(messages) == 2
        assert messages[0]["seq"] == 1
        assert messages[1]["seq"] == 2

    async def test_append_message_dedupe_prevents_duplicates(
        self, execution_store: ExecutionStore
    ) -> None:
        """append_message(): dedupe_key prevents duplicate projection."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id)
        frame_id = await _make_frame(store, session_id, turn_id)
        await store.append_message(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="user",
            content="First",
            dedupe_key="user:" + turn_id,
        )
        await store.append_message(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="user",
            content="Duplicate",
            dedupe_key="user:" + turn_id,
        )
        messages = await store.message_repo.get_by_session(session_id)
        assert len(messages) == 1

    async def test_append_event_delegates_and_commits(
        self, execution_store: ExecutionStore
    ) -> None:
        """append_event(): delegates to event_repo and commits."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id)
        frame_id = await _make_frame(store, session_id, turn_id)
        await store.append_event(
            trace_id=_new_id(),
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=1,
            event_type=EventType.FRAME_START,
            data_json='{"iteration": 1}',
        )
        events = await store.get_frame_events(frame_id)
        assert len(events) == 1
        assert events[0]["type"] == EventType.FRAME_START.value
        assert events[0]["seq"] == 1


class TestExecutionStoreLoad:
    async def test_load_resume_batch_orchestrates_full_graph(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_resume_batch(): session + turns + snapshot + frames + events + missing_refs."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id, turn_index=0)
        tool_id = await store.register_tool(
            name="t", schema_json="{}", risk_class="low"
        )
        snapshot_id = _new_id()
        await store.create_context_snapshot(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            tool_set_ref=json.dumps({"tool_ids": [tool_id]}),
            content_hash="hash123",
        )
        frame_id = await _make_frame(store, session_id, turn_id, iteration=0)
        await store.append_event(
            trace_id=_new_id(),
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=1,
            event_type=EventType.FRAME_START,
            data_json="{}",
        )
        batch = await store.load_resume_batch(session_id)
        assert batch is not None
        assert batch.session.id == session_id
        assert len(batch.turns) == 1
        assert batch.turn.id == turn_id
        assert batch.snapshot is not None
        assert batch.snapshot.id == snapshot_id
        assert len(batch.frames) == 1
        assert batch.frames[0].id == frame_id
        assert frame_id in batch.frame_events
        assert len(batch.frame_events[frame_id]) == 1
        assert batch.missing_refs.is_empty is True

    async def test_load_resume_batch_returns_none_for_missing_session(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_resume_batch(): returns None when session does not exist."""
        store = execution_store
        assert await store.load_resume_batch(_new_id()) is None

    async def test_load_resume_batch_turn_selection_by_id_and_index(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_resume_batch(): selects turn by turn_id or turn_index."""
        store = execution_store
        session_id = await _make_session(store)
        turn_0 = await _make_turn(store, session_id, turn_index=0)
        turn_1 = await _make_turn(store, session_id, turn_index=1)

        by_id = await store.load_resume_batch(session_id, turn_id=turn_1)
        assert by_id is not None
        assert by_id.turn.id == turn_1

        by_index = await store.load_resume_batch(session_id, turn_index=0)
        assert by_index is not None
        assert by_index.turn.id == turn_0

    async def test_load_resume_batch_frame_selection_explicit_and_default(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_resume_batch(): explicit frame_id, else last frame of turn."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id, turn_index=0)
        frame_0 = await _make_frame(store, session_id, turn_id, iteration=0)
        frame_1 = await _make_frame(store, session_id, turn_id, iteration=1)

        default = await store.load_resume_batch(session_id, turn_id=turn_id)
        assert default is not None
        assert default.frames[-1].id == frame_1

        explicit = await store.load_resume_batch(
            session_id, turn_id=turn_id, frame_id=frame_0
        )
        assert explicit is not None
        # frames holds all frames of the selected turn; frame_events holds the
        # single frame being resumed
        assert frame_0 in {f.id for f in explicit.frames}
        assert frame_1 in {f.id for f in explicit.frames}
        assert explicit.frame_events[frame_0] == []

    async def test_load_turn_replay_builds_turn_replay(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_turn_replay(): builds TurnReplay for a single turn."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id, turn_index=0)
        snapshot_id = _new_id()
        await store.create_context_snapshot(
            snapshot_id=snapshot_id, turn_id=turn_id, content_hash="hash123"
        )
        frame_id = await _make_frame(store, session_id, turn_id, iteration=0)
        await store.append_event(
            trace_id=_new_id(),
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=1,
            event_type=EventType.LLM_COMPLETE,
            data_json='{"content": "hi"}',
        )
        await store.append_message(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="user",
            content="q",
            dedupe_key="u:" + turn_id,
        )
        replay = await store.load_turn_replay(turn_id)
        assert replay is not None
        assert replay.turn.id == turn_id
        assert replay.snapshot is not None
        assert replay.snapshot.id == snapshot_id
        assert len(replay.frames) == 1
        assert replay.frames[0].id == frame_id
        assert frame_id in replay.frame_events
        assert len(replay.frame_events[frame_id]) == 1
        assert len(replay.messages) == 1

    async def test_load_turn_replay_returns_none_for_missing_turn(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_turn_replay(): returns None when turn does not exist."""
        store = execution_store
        assert await store.load_turn_replay(_new_id()) is None

    async def test_load_replay_batch_builds_session_replay(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_replay_batch(): builds full SessionReplay for a session."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id, turn_index=0)
        frame_id = await _make_frame(store, session_id, turn_id, iteration=0)
        await store.append_event(
            trace_id=_new_id(),
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=1,
            event_type=EventType.FRAME_START,
            data_json="{}",
        )
        replay = await store.load_replay_batch(session_id)
        assert replay is not None
        assert replay.session.id == session_id
        assert len(replay.turns) == 1
        assert replay.turns[0].turn.id == turn_id
        assert len(replay.turns[0].frames) == 1
        assert frame_id in replay.turns[0].frame_events

    async def test_load_replay_batch_returns_none_for_missing_session(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_replay_batch(): returns None when session does not exist."""
        store = execution_store
        assert await store.load_replay_batch(_new_id()) is None


class TestExecutionStoreContextRefs:
    async def test_check_context_refs_per_kind_in_queries_memory_ids_empty(
        self, execution_store: ExecutionStore
    ) -> None:
        """check_context_refs(): one IN query per kind, memory_ids always empty."""
        store = execution_store
        tool_id = await store.register_tool(
            name="t", schema_json="{}", risk_class="low"
        )
        template_id = await store.register_template(
            path="p", content="c", content_hash="h"
        )
        repo_hash = await store.register_repo_snapshot(
            target_project_root="/root",
            repo_map_json="{}",
            project_info_json=None,
            repo_snapshot_hash="hash_xyz",
        )
        assert repo_hash.is_new is True

        missing = await store.check_context_refs(
            tool_ids=[tool_id, 99999],
            template_ids=[template_id, 88888],
            repo_hashes=["hash_xyz", "nope"],
            memory_ids=["mem_1"],
        )
        assert missing.tool_ids == [99999]
        assert missing.template_ids == [88888]
        assert missing.repo_hashes == ["nope"]
        assert missing.memory_ids == []
        assert missing.is_empty is False

    async def test_check_context_refs_empty_when_all_refs_exist(
        self, execution_store: ExecutionStore
    ) -> None:
        """check_context_refs(): empty MissingRefs when all refs exist."""
        store = execution_store
        tool_id = await store.register_tool(
            name="t", schema_json="{}", risk_class="low"
        )
        missing = await store.check_context_refs(tool_ids=[tool_id])
        assert missing.is_empty is True
        assert missing.memory_ids == []

    async def test_check_context_refs_no_args_returns_empty(
        self, execution_store: ExecutionStore
    ) -> None:
        """check_context_refs(): no args returns empty MissingRefs."""
        store = execution_store
        missing = await store.check_context_refs()
        assert missing.is_empty is True


class TestExecutionStoreToolsTemplates:
    async def test_get_tool_by_id_delegates_to_tool_repo(
        self, execution_store: ExecutionStore
    ) -> None:
        """get_tool_by_id(): delegates to tool_repo.get."""
        store = execution_store
        tool_id = await store.register_tool(
            name="t", schema_json="{}", risk_class="low"
        )
        tool = await store.get_tool_by_id(tool_id)
        assert tool is not None
        assert tool["id"] == tool_id
        assert await store.get_tool_by_id(99999) is None

    async def test_get_tools_by_ids_delegates_to_tool_repo(
        self, execution_store: ExecutionStore
    ) -> None:
        """get_tools_by_ids(): delegates to tool_repo.get_many."""
        store = execution_store
        id_1 = await store.register_tool(name="t1", schema_json="{}", risk_class="low")
        id_2 = await store.register_tool(name="t2", schema_json="{}", risk_class="low")
        tools = await store.get_tools_by_ids([id_1, id_2, 99999])
        assert len(tools) == 2
        ids = {t["id"] for t in tools}
        assert ids == {id_1, id_2}

    async def test_get_template_by_id_delegates_to_template_repo(
        self, execution_store: ExecutionStore
    ) -> None:
        """get_template_by_id(): delegates to template_repo.get."""
        store = execution_store
        template_id = await store.register_template(
            path="p", content="c", content_hash="h"
        )
        template = await store.get_template_by_id(template_id)
        assert template is not None
        assert template["id"] == template_id
        assert await store.get_template_by_id(99999) is None

    async def test_get_templates_by_ids_delegates_to_template_repo(
        self, execution_store: ExecutionStore
    ) -> None:
        """get_templates_by_ids(): delegates to template_repo.get_many."""
        store = execution_store
        id_1 = await store.register_template(path="p1", content="c1", content_hash="h1")
        id_2 = await store.register_template(path="p2", content="c2", content_hash="h2")
        templates = await store.get_templates_by_ids([id_1, id_2])
        assert len(templates) == 2
        ids = {t["id"] for t in templates}
        assert ids == {id_1, id_2}


class TestExecutionStoreRepoSnapshot:
    async def test_get_repo_snapshot_by_hash_delegates(
        self, execution_store: ExecutionStore
    ) -> None:
        """get_repo_snapshot_by_hash(): delegates to repo_snapshot_repo."""
        store = execution_store
        result = await store.register_repo_snapshot(
            target_project_root="/root",
            repo_map_json="{}",
            project_info_json=None,
            repo_snapshot_hash="hash_abc",
        )
        assert result.is_new is True
        data = await store.get_repo_snapshot_by_hash("hash_abc")
        assert data is not None
        assert data.repo_snapshot_hash == "hash_abc"
        assert data.target_project_root == "/root"
        assert await store.get_repo_snapshot_by_hash("nope") is None

    async def test_register_repo_snapshot_deduplicates_by_hash(
        self, execution_store: ExecutionStore
    ) -> None:
        """register_repo_snapshot(): insert-or-ignore by hash, returns is_new."""
        store = execution_store
        r1 = await store.register_repo_snapshot(
            target_project_root="/root",
            repo_map_json="{}",
            project_info_json=None,
            repo_snapshot_hash="hash_abc",
        )
        r2 = await store.register_repo_snapshot(
            target_project_root="/root",
            repo_map_json="{}",
            project_info_json=None,
            repo_snapshot_hash="hash_abc",
        )
        assert r1.is_new is True
        assert r2.is_new is False
        assert r1.id == r2.id


class TestExecutionStoreContextArtifacts:
    async def test_update_context_artifacts_registers_and_computes_versions(
        self, execution_store: ExecutionStore
    ) -> None:
        """update_context_artifacts(): transaction, registers tools/templates, computes versions."""
        store = execution_store
        result = await store.update_context_artifacts(
            tool_specs=[
                {
                    "name": "read_file",
                    "parameters": {"type": "object"},
                    "description": "Read a file",
                }
            ],
            template_specs=[{"tmpl.jinja2": "Hi {{ name }}"}],
            repo_map_data=None,
            project_info=None,
            target_project_root="/root",
            memories=[],
        )
        assert result["tools_with_ids"][0]["id"] == 1
        assert result["template_ids"] == [1]
        assert result["repo_snapshot_hash"] == "no_repo"
        assert result["memory_snapshot_version"] == "empty"
        assert len(result["tool_set_version"]) == 16
        assert len(result["template_set_version"]) == 16

    async def test_update_context_artifacts_deduplicates_tools(
        self, execution_store: ExecutionStore
    ) -> None:
        """update_context_artifacts(): dedupes tools by content_hash."""
        store = execution_store
        spec = {
            "name": "read_file",
            "parameters": {"type": "object"},
            "description": "Read a file",
        }
        r1 = await store.update_context_artifacts(
            tool_specs=[spec],
            template_specs=[],
            repo_map_data=None,
            project_info=None,
            target_project_root="/root",
            memories=[],
        )
        r2 = await store.update_context_artifacts(
            tool_specs=[spec],
            template_specs=[],
            repo_map_data=None,
            project_info=None,
            target_project_root="/root",
            memories=[],
        )
        assert r1["tools_with_ids"] == r2["tools_with_ids"]

    @dataclass
    class _FakeRepoMap:
        root: str
        files: list[str]

    async def test_update_context_artifacts_registers_repo_snapshot(
        self, execution_store: ExecutionStore
    ) -> None:
        """update_context_artifacts(): registers repo snapshot when data given."""
        store = execution_store
        result = await store.update_context_artifacts(
            tool_specs=[],
            template_specs=[],
            repo_map_data=self._FakeRepoMap(root="/root", files=["a.py"]),
            project_info=None,
            target_project_root="/root",
            memories=["mem_a", "mem_b"],
        )
        assert result["repo_snapshot_hash"] != "no_repo"
        assert len(result["repo_snapshot_hash"]) == 16
        assert result["memory_snapshot_version"] != "empty"
        assert len(result["memory_snapshot_version"]) == 16


class TestExecutionStoreBranch:
    async def test_branch_session_creates_independent_fork(
        self, execution_store: ExecutionStore
    ) -> None:
        """branch_session(): atomic cascade copy, FK remapped, parent isolated."""
        store = execution_store
        parent_id = await _make_session(store, user_input="parent query")
        turn_0 = await _make_turn(store, parent_id, turn_index=0)
        await _make_turn(store, parent_id, turn_index=1)
        snapshot_id = _new_id()
        await store.create_context_snapshot(
            snapshot_id=snapshot_id,
            turn_id=turn_0,
            content_hash="hash123",
        )
        # Link the turn to its snapshot (create_turn does not set it)
        await store.turn_repo.upsert(
            turn_id=turn_0,
            session_id=parent_id,
            trace_id=_new_id(),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            context_snapshot_id=snapshot_id,
            config_snapshot_json="{}",
        )
        frame_id = await _make_frame(store, parent_id, turn_0, iteration=0)
        await store.append_event(
            trace_id=_new_id(),
            session_id=parent_id,
            turn_id=turn_0,
            frame_id=frame_id,
            seq=1,
            event_type=EventType.FRAME_START,
            data_json="{}",
        )
        await store.append_message(
            session_id=parent_id,
            turn_id=turn_0,
            frame_id=frame_id,
            seq=None,
            role="user",
            content="q",
            dedupe_key="u:" + turn_0,
        )

        fork_id = _new_id()
        fork = await store.branch_session(
            new_session_id=fork_id,
            parent_session_id=parent_id,
            branch_from_turn_id=turn_0,
        )
        assert fork.id == fork_id
        metadata = json.loads(fork.metadata_json)
        assert metadata["branched_from_session"] == parent_id
        assert metadata["branched_from_turn"] == turn_0

        # Fork only contains turns up to branch point
        fork_turns = await store.list_turns_by_session(fork_id)
        assert len(fork_turns) == 1
        assert fork_turns[0].turn_index == 0

        # Parent unchanged: still has both turns
        parent_turns = await store.list_turns_by_session(parent_id)
        assert len(parent_turns) == 2

        # Forked turn references fork's snapshot (remapped), not parent's
        fork_turn = fork_turns[0]
        assert fork_turn.context_snapshot_id != snapshot_id
        fork_snapshot = await store.get_context_snapshot(fork_turn.context_snapshot_id)
        assert fork_snapshot is not None
        assert fork_snapshot.turn_id == fork_turn.id

        # Forked frame references fork's turn
        fork_frames = await store.frame_repo.list_by_session(fork_id)
        assert len(fork_frames) == 1
        assert fork_frames[0].turn_id == fork_turn.id
        fork_events = await store.get_frame_events(fork_frames[0].id)
        assert len(fork_events) == 1


class TestExecutionStoreRetention:
    async def test_cleanup_retention_no_error_on_empty_db(
        self, execution_store: ExecutionStore
    ) -> None:
        """cleanup_retention(): no error on empty database."""
        store = execution_store
        await store.cleanup_retention({"default": 30})

    async def test_cleanup_retention_deletes_expired_idle_sessions(
        self, execution_store: ExecutionStore
    ) -> None:
        """cleanup_retention(): deletes sessions older than TTL, preserves running."""
        store = execution_store
        old_id = await _make_idle_session(store, user_input="old")
        running_id = await _make_session(store, user_input="running")

        # Backdate the old session's updated_at beyond the TTL
        cutoff_ms = int((datetime.now(UTC).timestamp() - 60 * 86400) * 1000)
        await store.session_repo.upsert(
            session_id=old_id,
            user_input="old",
            status="idle",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
            updated_at_ms=cutoff_ms - 1,
        )

        await store.cleanup_retention({"default": 30})

        assert await store.get_session(old_id) is None
        assert await store.get_session(running_id) is not None


class TestExecutionStoreFileBaseline:
    async def test_upsert_file_baseline_delegates(
        self, execution_store: ExecutionStore
    ) -> None:
        """upsert_file_baseline(): delegates to baseline repo and commits."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id)
        frame_id = await _make_frame(store, session_id, turn_id)
        obs = {
            "path": "/tmp/foo.txt",  # noqa: S108 - test fixture path, not a real temp dir
            "hash": "abc123",
            "size": 10,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "1",
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }
        await store.upsert_file_baseline(
            workspace_id="ws_1",
            observation=obs,
            turn_id=turn_id,
            frame_id=frame_id,
            session_id=session_id,
        )
        baselines = await store.load_file_baselines("ws_1")
        assert len(baselines) == 1
        assert baselines[0]["path"] == "/tmp/foo.txt"  # noqa: S108
        assert baselines[0]["hash"] == "abc123"

    async def test_load_file_baselines_filters_by_session(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_file_baselines(): optional session_id filter."""
        store = execution_store
        session_1 = await _make_session(store, user_input="s1")
        session_2 = await _make_session(store, user_input="s2")
        turn_1 = await _make_turn(store, session_1)
        turn_2 = await _make_turn(store, session_2)
        frame_1 = await _make_frame(store, session_1, turn_1)
        frame_2 = await _make_frame(store, session_2, turn_2)
        obs = {
            "path": "/tmp/foo.txt",  # noqa: S108 - test fixture path, not a real temp dir
            "hash": "abc",
            "size": 1,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "1",
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }
        await store.upsert_file_baseline(
            workspace_id="ws_1",
            observation=obs,
            turn_id=turn_1,
            frame_id=frame_1,
            session_id=session_1,
        )
        await store.upsert_file_baseline(
            workspace_id="ws_1",
            observation=obs,
            turn_id=turn_2,
            frame_id=frame_2,
            session_id=session_2,
        )
        only_s1 = await store.load_file_baselines("ws_1", session_id=session_1)
        assert len(only_s1) == 1

    async def test_delete_file_baseline_delegates(
        self, execution_store: ExecutionStore
    ) -> None:
        """delete_file_baseline(): delegates to baseline repo and commits."""
        store = execution_store
        session_id = await _make_session(store)
        turn_id = await _make_turn(store, session_id)
        frame_id = await _make_frame(store, session_id, turn_id)
        obs = {
            "path": "/tmp/foo.txt",  # noqa: S108 - test fixture path, not a real temp dir
            "hash": "abc",
            "size": 1,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "1",
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }
        await store.upsert_file_baseline(
            workspace_id="ws_1",
            observation=obs,
            turn_id=turn_id,
            frame_id=frame_id,
            session_id=session_id,
        )
        await store.delete_file_baseline("ws_1", "/tmp/foo.txt")  # noqa: S108
        baselines = await store.load_file_baselines("ws_1")
        assert len(baselines) == 0


class TestExecutionStoreWorkspace:
    async def test_get_workspace_delegates_to_workspace_repo(
        self, execution_store: ExecutionStore
    ) -> None:
        """get_workspace(): delegates to workspace_repo.get_by_root."""
        store = execution_store
        assert await store.get_workspace("/tmp/ws") is None  # noqa: S108

    async def test_upsert_workspace_delegates_and_returns_record(
        self, execution_store: ExecutionStore
    ) -> None:
        """upsert_workspace(): delegates to workspace_repo.create."""
        store = execution_store
        record = await store.upsert_workspace(
            workspace_id="ws_1",
            workspace_root="/tmp/ws",  # noqa: S108
            config_hash="hash_abc",
        )
        assert record.id == "ws_1"
        assert record.workspace_root == "/tmp/ws"  # noqa: S108
        assert record.config_hash == "hash_abc"

    async def test_refresh_workspace_config_hash_delegates(
        self, execution_store: ExecutionStore
    ) -> None:
        """refresh_workspace_config_hash(): delegates to workspace_repo."""
        store = execution_store
        await store.upsert_workspace(
            workspace_id="ws_1",
            workspace_root="/tmp/ws",  # noqa: S108
            config_hash="old",
        )
        await store.refresh_workspace_config_hash("ws_1", "new")
        record = await store.get_workspace("/tmp/ws")  # noqa: S108
        assert record is not None
        assert record.config_hash == "new"


class TestExecutionStoreEventProcessing:
    async def test_resumed_turn_completion_preserves_snapshot_fields(
        self, execution_store: ExecutionStore
    ) -> None:
        """A fresh adapter must not erase a turn's persisted snapshots on completion."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        await store.process_event(
            SessionStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                user_input="hi",
                agent_mode="frame_agent",
                model_name="gpt-4",
            )
        )
        await store.process_event(
            TurnStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                user_input="hi",
                turn_index=0,
                config_snapshot_json='{"context_config": {}, "limits_config": {}}',
                context_snapshot_id="snapshot_1",
            )
        )
        await store.flush()

        # A restarted process has the database rows but no pending create data.
        store._event_adapter = StoreEventAdapter()
        await store.process_event(
            TurnCompleteEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                status="completed",
            )
        )

        turn = await store.get_turn(turn_id)
        assert turn is not None
        assert turn.config_snapshot_json == (
            '{"context_config": {}, "limits_config": {}}'
        )
        assert turn.context_snapshot_id == "snapshot_1"

    async def test_resumed_event_sequence_continues_from_persisted_events(
        self, execution_store: ExecutionStore
    ) -> None:
        """A fresh adapter must continue the session's persisted event sequence."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        await store.process_event(
            SessionStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                user_input="hi",
                agent_mode="frame_agent",
                model_name="gpt-4",
            )
        )
        await store.flush()
        prior_events = await store.event_repo.get_by_session(session_id)
        prior_max_seq = max(event["seq"] for event in prior_events)

        store._event_adapter = StoreEventAdapter()
        await store.process_event(
            SessionResumeEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                resumed_from_turn_id=turn_id,
            )
        )
        await store.flush()

        events = await store.event_repo.get_by_session(session_id)
        resumed_event = next(
            event for event in events if event["type"] == "session_resume"
        )
        assert resumed_event["seq"] > prior_max_seq

    async def test_process_event_session_start_creates_session(
        self, execution_store: ExecutionStore
    ) -> None:
        """process_event(): SessionStartEvent creates session via adapter (flushed)."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        await store.process_event(
            SessionStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                user_input="hi",
                agent_mode="frame_agent",
                model_name="gpt-4",
            )
        )
        await store.flush()
        session = await store.get_session(session_id)
        assert session is not None
        assert session.user_input == "hi"

    async def test_process_event_turn_start_makes_session_running(
        self, execution_store: ExecutionStore
    ) -> None:
        """process_event(): TurnStartEvent sets session status running (flushed)."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        await store.process_event(
            SessionStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                user_input="hi",
                agent_mode="frame_agent",
                model_name="gpt-4",
            )
        )
        await store.process_event(
            TurnStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                user_input="hi",
                turn_index=0,
                config_snapshot_json="{}",
            )
        )
        await store.flush()
        session = await store.get_session(session_id)
        assert session is not None
        assert session.status == SessionStatus.RUNNING.value

    async def test_process_event_turn_complete_sets_session_idle(
        self, execution_store: ExecutionStore
    ) -> None:
        """process_event(): TurnCompleteEvent (boundary) flushes and sets idle."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        await store.process_event(
            SessionStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                user_input="hi",
                agent_mode="frame_agent",
                model_name="gpt-4",
            )
        )
        await store.process_event(
            TurnStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                user_input="hi",
                turn_index=0,
                config_snapshot_json="{}",
            )
        )
        await store.process_event(
            TurnCompleteEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                status="completed",
                total_input_tokens=6,
                total_output_tokens=4,
                total_frames=1,
            )
        )
        session = await store.get_session(session_id)
        assert session is not None
        assert session.status == SessionStatus.IDLE.value
        turn = await store.get_turn(turn_id)
        assert turn.status == "completed"

    async def test_process_event_llm_complete_increments_frame_tokens(
        self, execution_store: ExecutionStore
    ) -> None:
        """process_event(): LLMCompleteEvent (boundary) increments frame tokens."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        frame_id = _new_id()
        # Drive the full lifecycle through the adapter so frame_create and
        # frame_update merge (iteration/started_at_ms come from FrameStart).
        await store.process_event(
            SessionStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                user_input="hi",
                agent_mode="frame_agent",
                model_name="gpt-4",
            )
        )
        await store.process_event(
            TurnStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                user_input="hi",
                turn_index=0,
                config_snapshot_json="{}",
            )
        )
        await store.process_event(
            FrameStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                frame_id=frame_id,
                iteration=0,
                context_snapshot_id="snap",
                context_hash="hash",
            )
        )
        await store.process_event(
            LLMCompleteEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                frame_id=frame_id,
                content="response",
                input_tokens=14,
                output_tokens=9,
                model="gpt-4",
            )
        )
        frame = await store.frame_repo.get(frame_id)
        assert frame is not None
        assert frame.input_tokens == 14
        assert frame.output_tokens == 9
        assert frame.status == "running_tools"

    async def test_should_flush_boundary_events(
        self, execution_store: ExecutionStore
    ) -> None:
        """_should_flush(): True for FRAME_COMPLETE/LLM_COMPLETE/ERROR/TOOL_RESULT/TURN_COMPLETE."""
        store = execution_store
        assert store._should_flush(EventType.FRAME_COMPLETE) is True
        assert store._should_flush(EventType.LLM_COMPLETE) is True
        assert store._should_flush(EventType.ERROR) is True
        assert store._should_flush(EventType.TOOL_RESULT) is True
        assert store._should_flush(EventType.TURN_COMPLETE) is True
        assert store._should_flush(EventType.SESSION_START) is False
        assert store._should_flush(EventType.TURN_START) is False
        assert store._should_flush(EventType.FRAME_START) is False

    async def test_flush_prepares_batch_and_writes(
        self, execution_store: ExecutionStore
    ) -> None:
        """flush(): prepares adapter batch and writes it."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()
        frame_id = _new_id()
        await store.process_event(
            SessionStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                user_input="hi",
                agent_mode="frame_agent",
                model_name="gpt-4",
            )
        )
        await store.process_event(
            TurnStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                user_input="hi",
                turn_index=0,
                config_snapshot_json="{}",
            )
        )
        await store.process_event(
            FrameStartEvent(
                trace_id=_new_id(),
                turn_id=turn_id,
                session_id=session_id,
                frame_id=frame_id,
                iteration=0,
                context_snapshot_id="snap",
                context_hash="hash",
            )
        )
        # Flush non-boundary events that are still pending in the adapter
        await store.flush()
        session = await store.get_session(session_id)
        assert session is not None
        frame = await store.frame_repo.get(frame_id)
        assert frame is not None


class TestMissingRefsModel:
    async def test_missing_refs_is_empty_and_defaults(self) -> None:
        """MissingRefs: is_empty property and field defaults."""
        assert MissingRefs().is_empty is True
        assert MissingRefs(tool_ids=[1]).is_empty is False
        assert MissingRefs(memory_ids=["x"]).is_empty is False

    async def test_missing_refs_is_frozen(self) -> None:
        """MissingRefs: frozen dataclass immutability."""
        refs = MissingRefs(tool_ids=[1])
        with pytest.raises(FrozenInstanceError):
            refs.tool_ids = []  # frozen
