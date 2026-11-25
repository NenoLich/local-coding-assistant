"""Unit tests for resume plan construction (planner + frame selection)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.runtime.context.profile import (
    ContextProfile,
    ContextSnapshot,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.resume import (
    ResumeValidationError,
    build_approval_resume_plan,
    build_resume_plan,
    select_resumable_frame,
)
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.storage.contracts import (
    MissingRefs,
    ResumeBatch,
    StoredFrame,
    StoredSession,
    StoredTurn,
)


def make_snapshot(schema_version: int = 1) -> ContextSnapshot:
    return ContextSnapshot(
        id="snap_1",
        turn_id="turn_1",
        schema_version=schema_version,
        tool_set_ref=ToolSetRef(tool_ids=[1], tool_names=["read_file"]),
        template_set_ref=TemplateSetRef(template_ids=[2], template_paths=["base.j2"]),
        content_hash="hash",
        created_at_ms=0,
    )


def make_stored_turn(turn_id: str = "turn_1", status: str = "running") -> StoredTurn:
    return StoredTurn(
        id=turn_id,
        session_id="s1",
        trace_id="t1",
        turn_index=0,
        user_input="hi",
        status=status,
        started_at_ms=1000,
        config_snapshot_json=json.dumps(
            {
                "context_config": RuntimeContextConfig().model_dump(),
                "limits_config": RuntimeLimitsConfig().model_dump(),
            }
        ),
        context_snapshot_id="snap_1",
    )


def make_batch(
    frames: list[tuple[str, str]] | None = None,
    session_status: str = "running",
    turn_status: str = "running",
    snapshot: ContextSnapshot | None = None,
    missing_refs: MissingRefs | None = None,
    frame_events: dict[str, list[dict]] | None = None,
) -> ResumeBatch:
    if frames is None:
        frames = [("frame_1", "running_tools")]
    return ResumeBatch(
        session=StoredSession(
            id="s1",
            user_input="hi",
            agent_mode="frame_agent",
            status=session_status,
            started_at_ms=1000,
        ),
        turn=make_stored_turn(status=turn_status),
        snapshot=snapshot if snapshot is not None else make_snapshot(),
        turns=[make_stored_turn(status=turn_status)],
        frames=[
            StoredFrame(
                id=frame_id,
                session_id="s1",
                turn_id="turn_1",
                trace_id="t1",
                iteration=i + 1,
                status=status,
                started_at_ms=1000,
                context_snapshot_id="snap_1",
            )
            for i, (frame_id, status) in enumerate(frames)
        ],
        frame_events=frame_events or {},
        missing_refs=missing_refs or MissingRefs(),
    )


def make_session() -> SessionState:
    return SessionState(
        id="s1",
        context_profile=ContextProfile(
            runtime_context_config=RuntimeContextConfig(),
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
        ),
    )


def make_fakes(batch: ResumeBatch | None):
    store = MagicMock()
    store.load_resume_batch = AsyncMock(return_value=batch)
    context_manager = MagicMock()
    context_manager.update_bundle_cache_from_snapshot = AsyncMock(
        return_value=MagicMock(refs=MagicMock(tools=[]))
    )
    context_manager.compute_tool_set_version = MagicMock(return_value="v_resolved")
    session_manager = MagicMock()
    session_manager.get_cached = MagicMock(return_value=make_session())
    return store, context_manager, session_manager


class TestSelectResumableFrame:
    def test_last_frame_resumable(self) -> None:
        batch = make_batch()
        frame, warnings = select_resumable_frame(batch, None)
        assert frame is not None
        assert frame.id == "frame_1"
        assert warnings == []

    def test_last_frame_terminal_starts_fresh(self) -> None:
        batch = make_batch(frames=[("frame_1", "completed")])
        frame, warnings = select_resumable_frame(batch, None)
        assert frame is None
        assert len(warnings) == 1
        assert "terminal status completed" in warnings[0]

    def test_explicit_frame(self) -> None:
        batch = make_batch(
            frames=[("frame_1", "completed"), ("frame_2", "running_tools")]
        )
        frame, _ = select_resumable_frame(batch, "frame_2")
        assert frame is not None
        assert frame.id == "frame_2"

    def test_explicit_missing_frame_fatal(self) -> None:
        batch = make_batch()
        with pytest.raises(ResumeValidationError):
            select_resumable_frame(batch, "frame_nope")

    def test_explicit_terminal_frame_fatal(self) -> None:
        batch = make_batch(frames=[("frame_1", "failed")])
        with pytest.raises(ResumeValidationError) as exc:
            select_resumable_frame(batch, "frame_1")
        assert "terminal status failed" in str(exc.value)

    def test_snapshot_mismatch_starts_fresh(self) -> None:
        batch = make_batch(frames=[("frame_1", "running_tools")])
        batch.frames[0].context_snapshot_id = "snap_other"
        frame, warnings = select_resumable_frame(batch, None)
        assert frame is None
        assert len(warnings) == 1
        assert "snap_other" in warnings[0]

    def test_frame_events_restored(self) -> None:
        batch = make_batch(
            frame_events={
                "frame_1": [
                    {
                        "type": "llm_complete",
                        "data_json": json.dumps({"content": "hi"}),
                        "timestamp": 1000,
                        "seq": 1,
                    }
                ]
            }
        )
        frame, _ = select_resumable_frame(batch, None)
        assert frame is not None
        assert frame.model_response_raw == "hi"
        assert frame.actions


class TestBuildResumePlan:
    async def test_happy_path(self) -> None:
        batch = make_batch(
            frame_events={
                "frame_1": [
                    {
                        "type": "llm_complete",
                        "data_json": json.dumps({"content": "hi"}),
                        "timestamp": 1000,
                        "seq": 1,
                    }
                ]
            }
        )
        store, ctx_mgr, session_mgr = make_fakes(batch)

        plan = await build_resume_plan(
            store=store,
            context_manager=ctx_mgr,
            session_manager=session_mgr,
            session_id="s1",
        )

        assert plan.turn.turn_id == "turn_1"
        assert plan.frame is not None
        assert plan.frame.id == "frame_1"
        assert plan.warnings == []
        ctx_mgr.update_bundle_cache_from_snapshot.assert_awaited_once()
        store.load_resume_batch.assert_awaited_once_with(
            "s1", turn_id=None, turn_index=None, frame_id=None
        )

    async def test_session_not_cached_fatal(self) -> None:
        store, ctx_mgr, session_mgr = make_fakes(make_batch())
        session_mgr.get_cached = MagicMock(return_value=None)
        with pytest.raises(ResumeValidationError) as exc:
            await build_resume_plan(
                store=store,
                context_manager=ctx_mgr,
                session_manager=session_mgr,
                session_id="s1",
            )
        assert "not loaded in cache" in str(exc.value)

    async def test_session_idle_fatal(self) -> None:
        store, ctx_mgr, session_mgr = make_fakes(make_batch(session_status="idle"))
        with pytest.raises(ResumeValidationError) as exc:
            await build_resume_plan(
                store=store,
                context_manager=ctx_mgr,
                session_manager=session_mgr,
                session_id="s1",
            )
        assert "idle" in str(exc.value)

    async def test_missing_batch_fatal(self) -> None:
        store, ctx_mgr, session_mgr = make_fakes(None)
        with pytest.raises(ResumeValidationError) as exc:
            await build_resume_plan(
                store=store,
                context_manager=ctx_mgr,
                session_manager=session_mgr,
                session_id="s1",
                turn_id="turn_x",
            )
        assert "not found" in str(exc.value)

    async def test_missing_tool_refs_fatal(self) -> None:
        batch = make_batch(missing_refs=MissingRefs(tool_ids=[9]))
        store, ctx_mgr, session_mgr = make_fakes(batch)
        with pytest.raises(ResumeValidationError) as exc:
            await build_resume_plan(
                store=store,
                context_manager=ctx_mgr,
                session_manager=session_mgr,
                session_id="s1",
            )
        assert exc.value.details["missing_refs"]["tool_ids"] == [9]

    async def test_unsupported_snapshot_schema_fatal(self) -> None:
        batch = make_batch(snapshot=make_snapshot(schema_version=99))
        store, ctx_mgr, session_mgr = make_fakes(batch)
        with pytest.raises(ResumeValidationError):
            await build_resume_plan(
                store=store,
                context_manager=ctx_mgr,
                session_manager=session_mgr,
                session_id="s1",
            )

    async def test_explicit_terminal_frame_fatal(self) -> None:
        batch = make_batch(frames=[("frame_1", "completed")])
        store, ctx_mgr, session_mgr = make_fakes(batch)
        with pytest.raises(ResumeValidationError) as exc:
            await build_resume_plan(
                store=store,
                context_manager=ctx_mgr,
                session_manager=session_mgr,
                session_id="s1",
                frame_id="frame_1",
            )
        assert "cannot be resumed" in str(exc.value)

    async def test_tool_set_version_drift_warns(self) -> None:
        snapshot = make_snapshot()
        snapshot.tool_set_ref.version = "v_pinned"
        batch = make_batch(
            snapshot=snapshot,
            frame_events={
                "frame_1": [
                    {
                        "type": "llm_complete",
                        "data_json": json.dumps({"content": "hi"}),
                        "timestamp": 1000,
                        "seq": 1,
                    }
                ]
            },
        )
        store, ctx_mgr, session_mgr = make_fakes(batch)
        # Stored tool rows re-resolve to a different schema hash than pinned
        ctx_mgr.compute_tool_set_version = MagicMock(return_value="v_drifted")

        plan = await build_resume_plan(
            store=store,
            context_manager=ctx_mgr,
            session_manager=session_mgr,
            session_id="s1",
        )
        assert any("v_pinned -> v_drifted" in w for w in plan.warnings)
        ctx_mgr.compute_tool_set_version.assert_called_once()

    async def test_warnings_collected(self) -> None:
        batch = make_batch(missing_refs=MissingRefs(memory_ids=["m1"]))
        store, ctx_mgr, session_mgr = make_fakes(batch)
        session = make_session()
        session.history_sequence = 0
        session_mgr.get_cached = MagicMock(return_value=session)
        # frame with an event boundary but shorter live history
        batch.frames[0].history_through_seq = 7

        plan = await build_resume_plan(
            store=store,
            context_manager=ctx_mgr,
            session_manager=session_mgr,
            session_id="s1",
        )
        assert any("memory" in w for w in plan.warnings)
        assert any("no restored events" in w for w in plan.warnings)
        assert any("seq 0" in w for w in plan.warnings)


class TestBuildApprovalResumePlan:
    async def test_keeps_cached_turn_and_rebuilds_frame(self) -> None:
        from local_coding_assistant.runtime.session.turn import RuntimeTurn

        session = make_session()
        turn = RuntimeTurn(
            turn_id="turn_1",
            session_id="s1",
            trace_id="t1",
            turn_index=0,
            user_input="hi",
            context_snapshot=make_snapshot(),
            context_snapshot_id="snap_1",
            context_config=RuntimeContextConfig(),
            limits_config=RuntimeLimitsConfig(),
        )
        batch = make_batch(
            frame_events={
                "frame_1": [
                    {
                        "type": "llm_complete",
                        "data_json": json.dumps({"content": "hi"}),
                        "timestamp": 1000,
                        "seq": 1,
                    }
                ]
            }
        )
        store, _ctx, _sm = make_fakes(batch)

        plan = await build_approval_resume_plan(
            store=store, session=session, turn=turn, frame_id="frame_1"
        )

        assert plan.turn is turn  # cached turn object preserved (holds approvals)
        assert plan.frame is not None
        assert plan.frame.id == "frame_1"

    async def test_missing_paused_frame_fatal(self) -> None:
        from local_coding_assistant.runtime.session.turn import RuntimeTurn

        turn = RuntimeTurn(
            turn_id="turn_1",
            session_id="s1",
            trace_id="t1",
            turn_index=0,
            user_input="hi",
            context_snapshot=make_snapshot(),
            context_snapshot_id="snap_1",
            context_config=RuntimeContextConfig(),
            limits_config=RuntimeLimitsConfig(),
        )
        store, _ctx, _sm = make_fakes(make_batch(frames=[("frame_1", "completed")]))
        with pytest.raises(ResumeValidationError) as exc:
            await build_approval_resume_plan(
                store=store, session=make_session(), turn=turn, frame_id="frame_1"
            )
        assert "cannot be resumed" in str(exc.value)
