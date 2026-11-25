"""Integration tests: resume_session through the branch (fork) path.

Seeds a real SQLite-backed store with an interrupted turn that is NOT the
last turn (a later turn superseded it), then drives the real
``RuntimeManager.resume_session`` branch path with the frame agent mocked
out. Covers selector remapping onto the fork and the fork's resumability.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.config.path_manager import PathManager
from local_coding_assistant.config.schemas import AppConfig, RuntimeConfig
from local_coding_assistant.runtime.context.profile import (
    ContextProfile,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.engine.frame import ExecutionStatus
from local_coding_assistant.runtime.events.contracts import EventType
from local_coding_assistant.runtime.manager import RuntimeManager
from local_coding_assistant.runtime.resume import ResumeValidationError
from local_coding_assistant.storage.execution_store import ExecutionStore


def _profile_json() -> str:
    return ContextProfile(
        runtime_context_config=RuntimeContextConfig(),
        tool_set_ref=ToolSetRef(tool_ids=[], tool_names=[]),
        template_set_ref=TemplateSetRef(template_ids=[], template_paths=[]),
    ).model_dump_json()


def _config_snapshot_json() -> str:
    return json.dumps(
        {
            "context_config": RuntimeContextConfig().model_dump(),
            "limits_config": RuntimeLimitsConfig().model_dump(),
        }
    )


async def _seed_turn(
    store: ExecutionStore,
    *,
    session_id: str,
    turn_index: int,
    user_input: str,
    turn_status: str,
    frame_status: ExecutionStatus,
    paused: bool = False,
) -> dict[str, str]:
    """Seed one turn: snapshot + single running/interrupted frame + event."""
    turn_id = f"{session_id}_turn_{turn_index}"
    snapshot_id = f"snap_{turn_id}"
    frame_id = f"{turn_id}_frame_1"
    trace_id = f"trace_{turn_id}"

    await store.create_turn(
        turn_id=turn_id,
        session_id=session_id,
        trace_id=trace_id,
        turn_index=turn_index,
        user_input=user_input,
        status=turn_status,
        context_snapshot_id=snapshot_id,
        config_snapshot_json=_config_snapshot_json(),
    )
    await store.create_context_snapshot(
        snapshot_id=snapshot_id,
        turn_id=turn_id,
        tool_set_ref=json.dumps({"tool_ids": [], "tool_names": []}),
        template_set_ref=json.dumps({"template_ids": [], "template_paths": []}),
        memory_snapshot_ref=json.dumps({"memory_ids": [], "version": "v"}),
        content_hash="snap_hash",
    )
    await store.create_frame(
        frame_id=frame_id,
        session_id=session_id,
        turn_id=turn_id,
        trace_id=trace_id,
        iteration=1,
        context_snapshot_id=snapshot_id,
    )
    await store.update_frame_status(frame_id, status=frame_status)
    await store.append_event(
        trace_id=trace_id,
        session_id=session_id,
        turn_id=turn_id,
        frame_id=frame_id,
        seq=1,
        event_type=EventType.LLM_COMPLETE,
        data_json=json.dumps(
            {
                "content": "reading the file",
                "tool_calls": [
                    {"id": "call_1", "name": "read_file", "arguments": {"path": "a"}}
                ],
                "model": "gpt-4",
                "total_tokens": 42,
            }
        ),
    )
    if paused:
        await store.update_turn_status(
            turn_id,
            status="running",
            paused_at_ms=int(datetime.now(UTC).timestamp() * 1000),
            pause_reason="interrupt",
        )
    await store.flush()
    return {"turn_id": turn_id, "snapshot_id": snapshot_id, "frame_id": frame_id}


async def _seed_parent(
    store: ExecutionStore, *, turn_0_status: str = "running"
) -> tuple[str, dict[int, dict[str, str]]]:
    """Seed a session whose turn 0 is not the last turn.

    turn 0: interrupted (running + paused frame, mid tool call)
    turn 1: a later running turn (as if execution continued past turn 0)
    """
    session_id = str(uuid.uuid4())
    await store.session_repo.upsert(
        session_id=session_id,
        user_input="first",
        status="running",
        started_at_ms=int(datetime.now(UTC).timestamp() * 1000),
        agent_mode="frame_agent",
        model_name="gpt-4",
        context_profile_json=_profile_json(),
    )
    await store._conn.commit()

    turn_0 = await _seed_turn(
        store,
        session_id=session_id,
        turn_index=0,
        user_input="first",
        turn_status=turn_0_status,
        frame_status=ExecutionStatus.RUNNING_TOOLS,
        paused=turn_0_status == "running",
    )
    turn_1 = await _seed_turn(
        store,
        session_id=session_id,
        turn_index=1,
        user_input="second",
        turn_status="running",
        frame_status=ExecutionStatus.RUNNING_TOOLS,
    )
    return session_id, {0: turn_0, 1: turn_1}


class _StubConfigManager:
    """Minimal config manager with a real AppConfig (static resolution).

    A real ``global_config`` is required: the ContextManager builds a real
    ``RuntimeContextConfig`` from it.
    """

    def __init__(self) -> None:
        self.global_config = AppConfig(
            runtime=RuntimeConfig(session_context_resolution="static")
        )
        self.path_manager = PathManager(is_testing=True)
        self._overrides: dict[str, Any] = {}

    @property
    def session_overrides(self) -> dict[str, Any]:
        return dict(self._overrides)

    def load_global_config(self) -> AppConfig:
        return self.global_config

    def set_session_overrides(self, overrides: dict[str, Any]) -> None:
        self._overrides.update(overrides)

    def get_tools(self) -> dict[str, Any]:
        return {}

    def get_provider_by_model(self, model_id: str) -> str:
        return "test-provider"


def _make_runtime(store: ExecutionStore) -> RuntimeManager:
    """RuntimeManager over the real store with the frame agent mocked."""
    pipeline = MagicMock()
    pipeline.emit = AsyncMock()

    runtime = RuntimeManager(
        config_manager=_StubConfigManager(),
        store=store,
        event_pipeline=pipeline,
    )
    runtime._frame_agent = MagicMock()
    runtime._frame_agent.resume_turn = AsyncMock()
    return runtime


def _resume_args(runtime: RuntimeManager) -> tuple[object, object, object]:
    """Capture (turn, session, frame) handed to the frame agent."""
    runtime._frame_agent.resume_turn.assert_awaited_once()
    return runtime._frame_agent.resume_turn.await_args.args


@pytest.fixture
async def store(tmp_path: Path) -> ExecutionStore:
    execution_store = ExecutionStore(tmp_path / "branch_resume.db")
    await execution_store.init()
    yield execution_store
    await execution_store.close()


class TestBranchResume:
    async def test_fork_resumes_branch_point_turn(self, store) -> None:
        session_id, turns = await _seed_parent(store)
        runtime = _make_runtime(store)

        session = await runtime.resume_session(session_id, turn_id=turns[0]["turn_id"])

        # A new forked session was created and is itself resumable.
        assert session.id != session_id
        forked = await store.get_session(session.id)
        assert forked is not None
        assert forked.status == "running"

        turn, resumed_session, frame = _resume_args(runtime)
        assert resumed_session.id == session.id
        # The parent's turn ID no longer exists in the fork; the turn is
        # addressed by its preserved turn_index instead.
        assert turn.turn_index == 0  # type: ignore[attr-defined]
        assert turn.turn_id != turns[0]["turn_id"]  # type: ignore[attr-defined]
        assert turn.trace_id != turn.turn_id  # type: ignore[attr-defined]

        # The paused frame is restored from the copied events: its recorded
        # tool call survives the fork.
        assert frame is not None
        assert frame.id != turns[0]["frame_id"]  # type: ignore[attr-defined]
        assert frame.status == ExecutionStatus.RUNNING_TOOLS  # type: ignore[attr-defined]
        tool_names = [
            call.name
            for action in frame.actions  # type: ignore[attr-defined]
            for call in action.tool_calls
        ]
        assert tool_names == ["read_file"]

        # The parent is untouched (still interrupted, still running).
        parent = await store.get_session(session_id)
        assert parent is not None
        assert parent.status == "running"
        parent_turns = {t.id: t for t in await store.list_turns_by_session(session_id)}
        assert set(parent_turns) == {turns[0]["turn_id"], turns[1]["turn_id"]}
        assert parent_turns[turns[0]["turn_id"]].status == "running"

    async def test_fork_resumes_with_turn_index_selector(self, store) -> None:
        session_id, turns = await _seed_parent(store)
        runtime = _make_runtime(store)

        session = await runtime.resume_session(session_id, turn_index=0)

        assert session.id != session_id
        turn, _session, frame = _resume_args(runtime)
        assert turn.turn_index == 0  # type: ignore[attr-defined]
        assert turn.turn_id != turns[0]["turn_id"]  # type: ignore[attr-defined]
        assert frame is not None

    async def test_fork_remaps_explicit_frame_selector(self, store) -> None:
        session_id, turns = await _seed_parent(store)
        runtime = _make_runtime(store)

        session = await runtime.resume_session(
            session_id,
            turn_id=turns[0]["turn_id"],
            frame_id=turns[0]["frame_id"],
        )

        assert session.id != session_id
        turn, _session, frame = _resume_args(runtime)
        # The parent frame ID was remapped onto the fork's frame (matched
        # by turn index + frame iteration).
        assert frame is not None
        assert frame.id != turns[0]["frame_id"]  # type: ignore[attr-defined]
        forked_turn = next(
            t
            for t in await store.list_turns_by_session(session.id)
            if t.turn_index == turn.turn_index  # type: ignore[attr-defined]
        )
        forked_frames = await store.frame_repo.list_by_turn(forked_turn.id)
        assert {f.id for f in forked_frames} == {frame.id}  # type: ignore[attr-defined]

    async def test_resume_last_turn_does_not_branch(self, store) -> None:
        session_id, turns = await _seed_parent(store)
        runtime = _make_runtime(store)

        # No selector: the last (turn 1) turn is resumed in place.
        session = await runtime.resume_session(session_id)

        assert session.id == session_id
        turn, _session, frame = _resume_args(runtime)
        assert turn.turn_index == 1  # type: ignore[attr-defined]
        assert turn.turn_id == turns[1]["turn_id"]  # type: ignore[attr-defined]
        assert frame is not None
        assert frame.id == turns[1]["frame_id"]  # type: ignore[attr-defined]

    async def test_fork_from_completed_turn_is_rejected_with_guidance(
        self, store
    ) -> None:
        """No-message resume requires an interrupted branch point.

        The fork is created 'idle' (nothing interrupted at that turn), and
        the planner must refuse with actionable guidance instead of
        offering a completed turn.
        """
        session_id, turns = await _seed_parent(store, turn_0_status="completed")
        runtime = _make_runtime(store)

        with pytest.raises(ResumeValidationError) as exc:
            await runtime.resume_session(session_id, turn_id=turns[0]["turn_id"])

        message = str(exc.value)
        assert "idle" in message
        assert "resume_session_with_message" in message
        # The fork still exists (branching happened before validation) and
        # records where it was forked from.
        cursor = await store._conn.execute(
            "SELECT metadata_json FROM sessions WHERE id != ?", (session_id,)
        )
        forked_metadata = await cursor.fetchall()
        assert any(row[0] and "branched_from_turn" in row[0] for row in forked_metadata)
