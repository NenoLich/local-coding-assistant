"""Runtime-layer replay & reconstruction integration tests.

Seeds a real SQLite-backed store (session + turn + snapshot with refs +
running frame + events), then exercises the runtime reconstruction path:
``build_frame`` from stored events, ``ContextSnapshot.to_runtime()``,
``RuntimeTurn.from_batch``, ``select_resumable_frame``, snapshot schema
validation, and ContextManager snapshot resolution.

Only the LLM/gateway execution layers are absent (no frame execution).
Content-hash re-rendering is deliberately not asserted here: per the resume
validator, that belongs to the future replay flow, not the resume path.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.runtime.context.manager import ContextManager
from local_coding_assistant.runtime.context.profile import (
    ContextSnapshot,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.engine.frame import (
    ExecutionStatus,
    build_frame,
)
from local_coding_assistant.runtime.events.contracts import EventType
from local_coding_assistant.runtime.resume import (
    ResumeValidationError,
    select_resumable_frame,
    validate_snapshot_schema,
)
from local_coding_assistant.runtime.session.turn import RuntimeTurn
from local_coding_assistant.storage.execution_store import ExecutionStore


def _new_id() -> str:
    return f"t_{uuid.uuid4().hex[:12]}"


async def _seed_session(
    store: ExecutionStore,
    *,
    frame_status: str = "running_tools",
    schema_version: int = 1,
    tool_id: int | None = None,
    template_id: int | None = None,
) -> tuple[str, str, str, str]:
    """Create a resumable session: turn + snapshot (with refs) + frame + events.

    Returns (session_id, turn_id, snapshot_id, frame_id).
    """
    session_id = _new_id()
    turn_id = f"{session_id}_turn_0"
    snapshot_id = f"snap_{turn_id}"
    frame_id = f"{turn_id}_frame_1"
    trace_id = _new_id()

    await store.create_session(
        session_id=session_id,
        user_input="replay me",
        agent_mode="frame_agent",
        model_name="gpt-4",
    )

    await store.create_turn(
        turn_id=turn_id,
        session_id=session_id,
        trace_id=trace_id,
        turn_index=0,
        user_input="replay me",
        config_snapshot_json=json.dumps(
            {
                "context_config": RuntimeContextConfig().model_dump(),
                "limits_config": RuntimeLimitsConfig().model_dump(),
            }
        ),
        context_snapshot_id=snapshot_id,
    )

    if tool_id is None:
        tool_id = await store.register_tool(
            name="read_file",
            schema_json=json.dumps(
                {"type": "object", "properties": {"path": {"type": "string"}}}
            ),
            risk_class="low",
        )
    if template_id is None:
        template_id = await store.register_template(
            path="base/system_core.jinja2", content="System", content_hash="hash_tpl"
        )
    await store.register_repo_snapshot(
        target_project_root="project_root",
        repo_map_json="{}",
        project_info_json=None,
        repo_snapshot_hash="repo_hash",
    )

    await store.create_context_snapshot(
        snapshot_id=snapshot_id,
        turn_id=turn_id,
        schema_version=schema_version,
        tool_set_ref=json.dumps(
            {"tool_ids": [tool_id], "tool_names": ["read_file"], "version": "1.0.0"}
        ),
        template_set_ref=json.dumps(
            {
                "template_ids": [template_id],
                "template_paths": ["base/system_core.jinja2"],
                "version": "1.0.0",
            }
        ),
        repo_snapshot_ref=json.dumps(
            {"target_project_root": "project_root", "repo_snapshot_hash": "repo_hash"}
        ),
        memory_snapshot_ref=json.dumps({"memory_ids": [], "version": "v"}),
        content_hash="snap_hash",
    )

    await store.create_frame(
        frame_id=frame_id,
        session_id=session_id,
        trace_id=trace_id,
        turn_id=turn_id,
        iteration=1,
    )
    if frame_status != "running_llm":
        await store.update_frame_status(frame_id, status=ExecutionStatus(frame_status))

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
    await store.append_event(
        trace_id=trace_id,
        session_id=session_id,
        turn_id=turn_id,
        frame_id=frame_id,
        seq=2,
        event_type=EventType.TOOL_RESULT,
        data_json=json.dumps(
            {
                "call_id": "call_1",
                "tool_id": "read_file",
                "output": "file content",
                "success": True,
            }
        ),
    )
    await store.append_event(
        trace_id=trace_id,
        session_id=session_id,
        turn_id=turn_id,
        frame_id=frame_id,
        seq=3,
        event_type=EventType.FRAME_COMPLETE,
        data_json=json.dumps(
            {
                "final_answer": "done",
                "finish_reason": "stop",
                "duration_ms": 200,
                "total_input_tokens": 10,
                "total_output_tokens": 5,
            }
        ),
    )
    await store.flush()
    return session_id, turn_id, snapshot_id, frame_id


def _make_context_manager(store: ExecutionStore) -> ContextManager:
    config_manager = MagicMock()
    config_manager.global_config.runtime.session_context_resolution = "static"
    return ContextManager(
        tool_gateway=None,
        repo_service=None,
        memory_provider=MagicMock(),
        store=store,
        config_manager=config_manager,
    )


class TestFrameReconstruction:
    async def test_build_frame_from_real_stored_events(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "frame.db")
        await store.init()
        try:
            _session_id, _turn_id, _snapshot_id, frame_id = await _seed_session(store)

            stored = await store.frame_repo.get(frame_id)
            assert stored is not None
            assert stored.id == frame_id
            assert stored.status == ExecutionStatus.RUNNING_TOOLS.value

            events = await store.event_repo.get_by_frame(frame_id)
            assert len(events) == 3
            assert [e["type"] for e in events] == [
                "llm_complete",
                "tool_result",
                "frame_complete",
            ]

            frame = build_frame(stored, events)

            assert frame.id == frame_id
            assert frame.status == ExecutionStatus.RUNNING_TOOLS
            assert frame.model_response_raw == "reading the file"

            assert len(frame.actions) == 2
            assert frame.actions[0].kind.value == "llm_message"
            assert frame.actions[0].tool_calls[0].name == "read_file"
            assert frame.actions[0].tool_calls[0].arguments == {"path": "a"}
            assert frame.actions[1].kind.value == "tool_call"
            assert frame.actions[1].tool_execution_data["status"] == "completed"
            assert frame.actions[1].tool_execution_data["output"] == "file content"

            assert frame.result is not None
            assert frame.result.final_answer == "done"
            assert frame.result.finish_reason == "stop"
        finally:
            await store.close()


class TestContextSnapshotToRuntime:
    async def test_to_runtime_from_real_stored_snapshot(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "snapshot.db")
        await store.init()
        try:
            _session_id, _turn_id, snapshot_id, _frame_id = await _seed_session(store)

            stored = await store.context_snapshot_repo.get_stored(snapshot_id)
            assert stored is not None

            runtime = stored.to_runtime()
            assert runtime.id == snapshot_id
            assert runtime.turn_id == _turn_id
            assert runtime.content_hash == "snap_hash"
            assert runtime.tool_set_ref.tool_ids == [1]
            assert runtime.tool_set_ref.tool_names == ["read_file"]
            assert runtime.template_set_ref.template_ids == [1]
            assert runtime.template_set_ref.template_paths == [
                "base/system_core.jinja2"
            ]
            assert runtime.repo_snapshot_ref is not None
            assert runtime.repo_snapshot_ref.target_project_root == "project_root"
            assert runtime.repo_snapshot_ref.repo_snapshot_hash == "repo_hash"
        finally:
            await store.close()

    async def test_to_runtime_handles_null_refs(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "snapshot.db")
        await store.init()
        try:
            session_id = _new_id()
            turn_id = f"{session_id}_turn_0"

            await store.create_session(
                session_id=session_id,
                user_input="x",
                agent_mode="frame_agent",
                model_name="gpt-4",
            )
            await store.create_turn(
                turn_id=turn_id,
                session_id=session_id,
                trace_id=_new_id(),
                turn_index=0,
                user_input="x",
                config_snapshot_json=json.dumps(
                    {
                        "context_config": RuntimeContextConfig().model_dump(),
                        "limits_config": RuntimeLimitsConfig().model_dump(),
                    }
                ),
            )

            snapshot_id = _new_id()
            await store.create_context_snapshot(
                snapshot_id=snapshot_id,
                turn_id=turn_id,
                tool_set_ref="null",
                template_set_ref='""',
                repo_snapshot_ref=None,
                content_hash="hash",
                created_at_ms=1000,
            )

            stored = await store.context_snapshot_repo.get_stored(snapshot_id)
            assert stored is not None
            runtime = stored.to_runtime()
            assert runtime.tool_set_ref.tool_ids == []
            assert runtime.tool_set_ref.tool_names == []
            assert runtime.template_set_ref.template_ids == []
            assert runtime.repo_snapshot_ref is None
        finally:
            await store.close()


class TestRuntimeTurnFromBatch:
    async def test_from_batch_rebuilds_turn_with_snapshot(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "turn.db")
        await store.init()
        try:
            session_id, turn_id, snapshot_id, frame_id = await _seed_session(store)

            batch = await store.load_resume_batch(session_id, turn_id=turn_id)
            assert batch is not None
            assert batch.snapshot is not None
            assert batch.snapshot.id == snapshot_id
            assert len(batch.frames) == 1
            assert batch.frames[0].id == frame_id

            turn = RuntimeTurn.from_batch(batch)

            assert turn.turn_id == turn_id
            assert turn.session_id == session_id
            assert turn.is_resumed is True
            assert turn.context_snapshot is not None
            assert turn.context_snapshot.id == snapshot_id
            assert turn.context_config is not None
            assert turn.limits_config is not None
            assert frame_id in turn.frame_ids
        finally:
            await store.close()


class TestSelectResumableFrame:
    async def test_selects_running_frame_from_real_batch(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "select.db")
        await store.init()
        try:
            _session_id, _turn_id, _snapshot_id, frame_id = await _seed_session(store)

            batch = await store.load_resume_batch(_session_id, turn_id=_turn_id)
            frame, warnings = select_resumable_frame(batch, None)
            assert frame is not None
            assert frame.id == frame_id
            assert warnings == []
        finally:
            await store.close()

    async def test_explicit_terminal_frame_raises(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "select.db")
        await store.init()
        try:
            _session_id, _turn_id, _snapshot_id, frame_id = await _seed_session(
                store, frame_status="completed"
            )

            batch = await store.load_resume_batch(_session_id, turn_id=_turn_id)
            with pytest.raises(ResumeValidationError) as exc:
                select_resumable_frame(batch, frame_id)
            assert "cannot be resumed" in str(exc.value)
        finally:
            await store.close()

    async def test_explicit_missing_frame_raises(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "select.db")
        await store.init()
        try:
            _session_id, _turn_id, _snapshot_id, _frame_id = await _seed_session(store)

            batch = await store.load_resume_batch(_session_id, turn_id=_turn_id)
            with pytest.raises(ResumeValidationError):
                select_resumable_frame(batch, "frame_nope")
        finally:
            await store.close()

    async def test_terminal_last_frame_starts_fresh(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "select.db")
        await store.init()
        try:
            _session_id, _turn_id, _snapshot_id, _frame_id = await _seed_session(
                store, frame_status="completed"
            )

            batch = await store.load_resume_batch(_session_id, turn_id=_turn_id)
            frame, warnings = select_resumable_frame(batch, None)
            assert frame is None
            assert len(warnings) == 1
            assert "terminal status completed" in warnings[0]
        finally:
            await store.close()


class TestSnapshotSchemaValidation:
    async def test_supported_schema_passes(self) -> None:
        snapshot = ContextSnapshot(
            id="snap_1",
            turn_id="turn_1",
            schema_version=1,
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
            content_hash="hash",
            created_at_ms=0,
        )
        assert validate_snapshot_schema(snapshot) == []

    async def test_unsupported_schema_raises(self) -> None:
        snapshot = ContextSnapshot(
            id="snap_1",
            turn_id="turn_1",
            schema_version=99,
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
            content_hash="hash",
            created_at_ms=0,
        )
        with pytest.raises(ResumeValidationError) as exc:
            validate_snapshot_schema(snapshot)
        assert exc.value.details["schema_version"] == 99

    async def test_missing_snapshot_raises(self) -> None:
        with pytest.raises(ResumeValidationError):
            validate_snapshot_schema(None)


class TestSnapshotResolution:
    async def test_context_manager_resolves_snapshot_refs(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "resolve.db")
        await store.init()
        try:
            _session_id, _turn_id, snapshot_id, _frame_id = await _seed_session(store)

            context_manager = _make_context_manager(store)
            stored_snapshot = await store.context_snapshot_repo.get_stored(snapshot_id)
            snapshot = stored_snapshot.to_runtime()
            bundle = await context_manager.update_bundle_cache_from_snapshot(
                session_id=_session_id,
                snapshot=snapshot,
                config_snapshot=RuntimeContextConfig(),
            )

            assert bundle is not None
            assert bundle.profile.tool_set_ref.tool_ids == [1]
            assert bundle.refs.tools
            assert any(t.get("name") == "read_file" for t in bundle.refs.tools)
        finally:
            await store.close()


class TestMissingRefsReporting:
    async def test_batch_reports_missing_tool_template_repo(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "missing.db")
        await store.init()
        try:
            session_id = _new_id()
            turn_id = f"{session_id}_turn_0"
            snapshot_id = f"snap_{turn_id}"

            await store.create_session(
                session_id=session_id,
                user_input="x",
                agent_mode="frame_agent",
                model_name="gpt-4",
            )
            await store.create_turn(
                turn_id=turn_id,
                session_id=session_id,
                trace_id=_new_id(),
                turn_index=0,
                user_input="x",
                config_snapshot_json=json.dumps(
                    {
                        "context_config": RuntimeContextConfig().model_dump(),
                        "limits_config": RuntimeLimitsConfig().model_dump(),
                    }
                ),
                context_snapshot_id=snapshot_id,
            )

            # Snapshot pins ids/hashes that do not exist in the store.
            await store.create_context_snapshot(
                snapshot_id=snapshot_id,
                turn_id=turn_id,
                tool_set_ref=json.dumps(
                    {
                        "tool_ids": [1, 999],
                        "tool_names": ["calc", "ghost"],
                        "version": "1.0.0",
                    }
                ),
                template_set_ref=json.dumps(
                    {
                        "template_ids": [2, 888],
                        "template_paths": ["a.j2", "b.j2"],
                        "version": "1.0.0",
                    }
                ),
                repo_snapshot_ref=json.dumps(
                    {"target_project_root": "/root", "repo_snapshot_hash": "deadbeef"}
                ),
                content_hash="hash",
                created_at_ms=1000,
            )
            await store.create_frame(
                frame_id=f"{turn_id}_f",
                session_id=session_id,
                trace_id=_new_id(),
                turn_id=turn_id,
                iteration=1,
            )
            await store.flush()

            batch = await store.load_resume_batch(session_id, turn_id=turn_id)
            assert batch is not None
            missing = batch.missing_refs
            assert missing.tool_ids == [1, 999]
            assert missing.template_ids == [2, 888]
            assert missing.repo_hashes == ["deadbeef"]
            assert missing.memory_ids == []
            assert not missing.is_empty
        finally:
            await store.close()
