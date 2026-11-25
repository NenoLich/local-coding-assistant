"""Integration tests: resume validation through the real store and planners.

Seeds a real SQLite-backed store (session + turn + snapshot with refs +
running frame + events), then runs the real ``build_resume_plan`` /
``build_approval_resume_plan`` against the real ContextManager and
SessionManager. Only the LLM/gateway layers are absent (no execution).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.runtime.context.manager import ContextManager
from local_coding_assistant.runtime.context.profile import (
    ContextProfile,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.engine.frame import ExecutionStatus
from local_coding_assistant.runtime.events.contracts import EventType
from local_coding_assistant.runtime.resume import (
    MissingContextReferencesError,
    ResumeValidationError,
    build_approval_resume_plan,
    build_resume_plan,
    validate_session_profile,
)
from local_coding_assistant.runtime.session import SessionManager
from local_coding_assistant.runtime.session.turn import RuntimeTurn
from local_coding_assistant.storage.execution_store import ExecutionStore


async def _seed_session(
    store: ExecutionStore,
    *,
    referenced_tool_id: int | None = None,
    referenced_template_id: int | None = None,
    frame_status: str = "running_tools",
) -> tuple[str, str]:
    """Create a resumable session: turn + snapshot (with refs) + running frame."""
    session_id = str(uuid.uuid4())
    turn_id = f"{session_id}_turn_0"
    snapshot_id = f"snap_{turn_id}"
    frame_id = f"{turn_id}_frame_1"
    trace_id = str(uuid.uuid4())

    # The live flow persists the session row via the store adapter; seed it
    # directly with a context profile (single raw write, then commit so the
    # batched write path can begin its own transaction).
    await store.session_repo.upsert(
        session_id=session_id,
        user_input="resume me",
        status="running",
        started_at_ms=int(datetime.now(UTC).timestamp() * 1000),
        agent_mode="frame_agent",
        model_name="gpt-4",
        context_profile_json=ContextProfile(
            runtime_context_config=RuntimeContextConfig(),
            tool_set_ref=ToolSetRef(tool_ids=[], tool_names=[]),
            template_set_ref=TemplateSetRef(template_ids=[], template_paths=[]),
        ).model_dump_json(),
    )
    await store._conn.commit()

    await store.create_turn(
        turn_id=turn_id,
        session_id=session_id,
        trace_id=trace_id,
        turn_index=0,
        user_input="resume me",
        config_snapshot_json=json.dumps(
            {
                "context_config": RuntimeContextConfig().model_dump(),
                "limits_config": RuntimeLimitsConfig().model_dump(),
            }
        ),
        context_snapshot_id=snapshot_id,
    )

    tool_id = referenced_tool_id
    if tool_id is None:
        tool_id = await store.register_tool(
            name="read_file",
            schema_json=json.dumps(
                {"type": "object", "properties": {"path": {"type": "string"}}}
            ),
            risk_class="low",
        )
    template_id = referenced_template_id
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
        tool_set_ref=json.dumps({"tool_ids": [tool_id], "tool_names": ["read_file"]}),
        template_set_ref=json.dumps(
            {
                "template_ids": [template_id],
                "template_paths": ["base/system_core.jinja2"],
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

    # Recorded LLM turn with a tool call: the frame resumes the tools phase.
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
    await store.flush()
    return session_id, turn_id


def _make_context_manager(
    store: ExecutionStore, resolution: str = "static"
) -> ContextManager:
    config_manager = MagicMock()
    config_manager.global_config.runtime.session_context_resolution = resolution
    return ContextManager(
        tool_gateway=None,
        repo_service=None,
        memory_provider=MagicMock(),
        store=store,
        config_manager=config_manager,
    )


class TestSessionProfileValidation:
    async def test_static_profile_with_missing_tool_fails(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "profile.db")
        await store.init()
        try:
            session_id, _turn_id = await _seed_session(store)
            session_manager = SessionManager(store)
            session = await session_manager.resume_session(session_id, load_turns=True)
            # Stored profile pins a tool id that does not exist
            profile = session.context_profile.model_copy(
                update={
                    "tool_set_ref": ToolSetRef(
                        tool_ids=[31337], tool_names=["ghost"], version="v_profile"
                    )
                }
            )

            with pytest.raises(MissingContextReferencesError) as exc:
                await validate_session_profile(
                    context_manager=_make_context_manager(store),
                    store=store,
                    profile=profile,
                )
            assert exc.value.missing_refs == ["tool:31337"]
        finally:
            await store.close()

    async def test_dynamic_profile_skips_checks(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "profile.db")
        await store.init()
        try:
            session_id, _turn_id = await _seed_session(store)
            session_manager = SessionManager(store)
            session = await session_manager.resume_session(session_id, load_turns=True)
            profile = session.context_profile.model_copy(
                update={
                    "tool_set_ref": ToolSetRef(
                        tool_ids=[31337], tool_names=["ghost"], version="v_profile"
                    )
                }
            )

            result = await validate_session_profile(
                context_manager=_make_context_manager(store, resolution="dynamic"),
                store=store,
                profile=profile,
            )
            assert result.warnings == []
        finally:
            await store.close()

    async def test_static_profile_tool_drift_warns(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "profile.db")
        await store.init()
        try:
            session_id, turn_id = await _seed_session(store)
            session_manager = SessionManager(store)
            session = await session_manager.resume_session(session_id, load_turns=True)

            snapshot = await store.get_context_snapshot_by_turn(turn_id)
            assert snapshot is not None
            tool_id = snapshot.tool_set_ref.tool_ids[0]
            tool = await store.get_tool_by_id(tool_id)
            assert tool is not None

            # Pin a version computed from a different schema than the stored
            # row: recomputation over the row must diverge -> drift warning.
            context_manager = _make_context_manager(store)
            drifted_version = context_manager.compute_tool_set_version(
                [
                    {
                        "id": tool_id,
                        "name": tool["name"],
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "integer"}},
                        },
                    }
                ]
            )
            profile = session.context_profile.model_copy(
                update={
                    "tool_set_ref": ToolSetRef(
                        tool_ids=[tool_id],
                        tool_names=[tool["name"]],
                        version=drifted_version,
                    )
                }
            )

            result = await validate_session_profile(
                context_manager=context_manager,
                store=store,
                profile=profile,
            )
            assert len(result.warnings) == 1
            assert "drifted" in result.warnings[0]
        finally:
            await store.close()


class TestResumeValidationIntegration:
    async def test_resume_plan_from_real_store(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "resume.db")
        await store.init()
        try:
            session_id, turn_id = await _seed_session(store)
            session_manager = SessionManager(store)
            session = await session_manager.resume_session(session_id, load_turns=True)
            assert session is not None

            plan = await build_resume_plan(
                store=store,
                context_manager=_make_context_manager(store),
                session_manager=session_manager,
                session_id=session_id,
            )

            assert plan.turn.turn_id == turn_id
            assert plan.turn.context_snapshot is not None
            assert plan.frame is not None
            assert plan.frame.id == f"{turn_id}_frame_1"
            assert plan.warnings == []
            # The pinned frame's tool call is restorable
            tool_call = plan.frame.actions[0].tool_calls[0]
            assert tool_call.name == "read_file"
        finally:
            await store.close()

    async def test_missing_tool_ref_fails_resume(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "resume.db")
        await store.init()
        try:
            # Snapshot references a tool that does not exist in the store
            session_id, _turn_id = await _seed_session(store, referenced_tool_id=424242)
            session_manager = SessionManager(store)
            await session_manager.resume_session(session_id, load_turns=True)

            with pytest.raises(ResumeValidationError) as exc:
                await build_resume_plan(
                    store=store,
                    context_manager=_make_context_manager(store),
                    session_manager=session_manager,
                    session_id=session_id,
                )
            assert exc.value.details["missing_refs"]["tool_ids"] == [424242]
        finally:
            await store.close()

    async def test_missing_template_ref_fails_resume(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "resume.db")
        await store.init()
        try:
            # Snapshot references a template that does not exist in the store
            session_id, _turn_id = await _seed_session(
                store, referenced_template_id=987654
            )
            session_manager = SessionManager(store)
            await session_manager.resume_session(session_id, load_turns=True)

            with pytest.raises(ResumeValidationError) as exc:
                await build_resume_plan(
                    store=store,
                    context_manager=_make_context_manager(store),
                    session_manager=session_manager,
                    session_id=session_id,
                )
            assert exc.value.details["missing_refs"]["template_ids"] == [987654]
        finally:
            await store.close()

    async def test_approval_plan_keeps_cached_turn(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "resume.db")
        await store.init()
        try:
            session_id, turn_id = await _seed_session(store)
            session_manager = SessionManager(store)
            await session_manager.resume_session(session_id, load_turns=True)

            plan = await build_resume_plan(
                store=store,
                context_manager=_make_context_manager(store),
                session_manager=session_manager,
                session_id=session_id,
            )
            cached_turn: RuntimeTurn = plan.turn

            approval_plan = await build_approval_resume_plan(
                store=store,
                session=plan.session,
                turn=cached_turn,
                frame_id=f"{turn_id}_frame_1",
            )

            assert approval_plan.turn is cached_turn
            assert approval_plan.frame is not None
            assert approval_plan.frame.id == f"{turn_id}_frame_1"
            assert approval_plan.warnings == []
        finally:
            await store.close()
