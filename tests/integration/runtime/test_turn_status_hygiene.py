"""Verification: turn/session status hygiene.

Reproduces the "superseded turn" scenario:
1. Turn A is interrupted (status=running, paused)
2. resume_session_with_message creates Turn B (status=running)
3. Check whether Turn A is still "running" in the store
4. Complete Turn B (session -> idle)
5. Check whether resume_session offers a stale turn
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.runtime.context import (
    ContextManager,
    ContextProfile,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.manager import RuntimeManager
from local_coding_assistant.runtime.resume import (
    ResumeValidationError,
    build_resume_plan,
)
from local_coding_assistant.runtime.session import SessionManager
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


async def _set_session_status(
    store: ExecutionStore, session_id: str, status: str
) -> None:
    """Set session status while preserving the stored context profile.

    ``ExecutionStore.update_session`` does not carry the profile, and the
    upsert overwrites ``context_profile_json`` with NULL, so the profile
    is re-supplied here (the live flow keeps it in the cached
    SessionState; the stored column is only written on branch).
    """
    stored = await store.get_session(session_id)
    await store.session_repo.upsert(
        session_id=session_id,
        user_input=stored.user_input if stored else "",
        status=status,
        started_at_ms=stored.started_at_ms if stored else 0,
        agent_mode=stored.agent_mode if stored else "frame_agent",
        model_name=stored.model_name if stored else None,
        context_profile_json=(
            stored.context_profile_json if stored else _profile_json()
        ),
    )
    assert store._conn is not None
    await store._conn.commit()


async def _seed_interrupted_turn(store: ExecutionStore) -> tuple[str, str]:
    """Create a session whose only turn is interrupted (running + paused)."""
    session_id = str(uuid.uuid4())
    turn_id = f"{session_id}_turn_0"
    snapshot_id = f"snap_{turn_id}"
    trace_id = str(uuid.uuid4())

    await store.session_repo.upsert(
        session_id=session_id,
        user_input="first message",
        status="running",
        started_at_ms=int(datetime.now(UTC).timestamp() * 1000),
        agent_mode="frame_agent",
        model_name="gpt-4",
        context_profile_json=_profile_json(),
    )
    assert store._conn is not None
    await store._conn.commit()

    await store.create_turn(
        turn_id=turn_id,
        session_id=session_id,
        trace_id=trace_id,
        turn_index=0,
        user_input="first message",
        status="running",
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
    # Mark the turn paused (interrupted) - it stays "running" in the store.
    await store.update_turn_status(
        turn_id,
        status="running",
        paused_at_ms=int(datetime.now(UTC).timestamp() * 1000),
        pause_reason="interrupt",
    )
    await store.flush()
    return session_id, turn_id


async def _create_superseding_turn(
    store: ExecutionStore, session_id: str, turn_index: int
) -> str:
    """Simulate resume_session_with_message creating a new running turn.

    This mirrors what TurnStartEvent persists: a new turn row with
    status=running and the session flipped to running. The prior turn
    is NOT touched.
    """
    turn_id = f"{session_id}_turn_{turn_index}"
    snapshot_id = f"snap_{turn_id}"
    trace_id = str(uuid.uuid4())

    await store.create_turn(
        turn_id=turn_id,
        session_id=session_id,
        trace_id=trace_id,
        turn_index=turn_index,
        user_input="second message",
        status="running",
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
    # TurnStartEvent flips the session to running.
    await _set_session_status(store, session_id, "running")
    await store.flush()
    return turn_id


class TestSupersededTurnHygiene:
    def _runtime_with_mock_agents(self, store: ExecutionStore) -> RuntimeManager:
        """Build a RuntimeManager over the real store with the turn
        and frame agents mocked out (no LLM/snapshot resolution)."""
        config_manager = MagicMock()
        config_manager.global_config.runtime.session_context_resolution = "static"
        config_manager.global_config.llm.model_name = "gpt-4"
        config_manager.global_config.llm.temperature = 0.0
        config_manager.global_config.llm.max_tokens = 4096
        config_manager.global_config.runtime.tool_call_mode = "native"
        config_manager.global_config.sandbox.enabled = False
        config_manager.global_config.prompt.agents_md.enabled = True
        config_manager.global_config.prompt.agents_md.walk_up = True
        config_manager.global_config.prompt.agents_md.filenames = []
        config_manager.global_config.prompt.agents_md.global_paths = []
        config_manager.global_config.prompt.config_instructions = []
        config_manager.global_config.prompt.templates = {}
        config_manager.global_config.prompt.trim_blocks = True
        config_manager.global_config.prompt.lstrip_blocks = True
        config_manager.get_provider_by_model = MagicMock(return_value="openai")
        runtime = RuntimeManager(
            config_manager=config_manager,
            store=store,
            event_pipeline=MagicMock(),
        )
        runtime._turn_manager = MagicMock()
        runtime._turn_manager.create_turn = AsyncMock(return_value=MagicMock())
        runtime._frame_agent = MagicMock()
        runtime._frame_agent.run_turn = AsyncMock()
        return runtime

    async def test_direct_turn_creation_leaves_prior_running(self, tmp_path) -> None:
        """Store layer alone: a manually created turn does not
        finalize the prior running turn (the runtime manager
        closes this gap - see the test below)."""
        store = ExecutionStore(tmp_path / "hygiene.db")
        await store.init()
        try:
            session_id, turn_a = await _seed_interrupted_turn(store)

            # A turn created directly (bypassing resume_session_with_message)
            # while turn A is still running.
            turn_b = await _create_superseding_turn(store, session_id, 1)

            turns = await store.list_turns_by_session(session_id)
            by_id = {t.id: t for t in turns}
            # Turn A was never finalized - it is stuck in "running".
            assert by_id[turn_a].status == "running"
            assert by_id[turn_a].finished_at_ms is None
            # Turn B is also running.
            assert by_id[turn_b].status == "running"
        finally:
            await store.close()

    async def test_resume_session_with_message_supersedes_prior_turn(
        self, tmp_path
    ) -> None:
        """resume_session_with_message finalizes the prior running turn."""
        store = ExecutionStore(tmp_path / "hygiene.db")
        await store.init()
        try:
            session_id, turn_a = await _seed_interrupted_turn(store)

            runtime = self._runtime_with_mock_agents(store)

            await runtime.resume_session_with_message(session_id, "second message")

            # The prior interrupted turn is now finalized as superseded.
            stored_a = await store.get_turn(turn_a)
            assert stored_a is not None
            assert stored_a.status == "superseded"
            assert stored_a.finished_at_ms is not None
        finally:
            await store.close()

    async def test_resume_session_with_message_skips_terminal_turn(
        self, tmp_path
    ) -> None:
        """A completed prior turn is left untouched."""
        store = ExecutionStore(tmp_path / "hygiene.db")
        await store.init()
        try:
            session_id, turn_a = await _seed_interrupted_turn(store)
            # Complete the prior turn so it is terminal.
            await store.update_turn_status(
                turn_a,
                status="completed",
                finished_at_ms=int(datetime.now(UTC).timestamp() * 1000),
            )
            await _set_session_status(store, session_id, "idle")
            await store.flush()

            runtime = self._runtime_with_mock_agents(store)

            await runtime.resume_session_with_message(session_id, "second message")

            # The completed turn keeps its status.
            stored_a = await store.get_turn(turn_a)
            assert stored_a is not None
            assert stored_a.status == "completed"
        finally:
            await store.close()

    async def test_resume_after_superseding_turn_completes(self, tmp_path) -> None:
        """Once the superseding turn completes, resume is gated off.

        The stale turn A is still 'running', but the session is idle, so
        build_resume_plan must refuse rather than offer the stale turn.
        """
        store = ExecutionStore(tmp_path / "hygiene.db")
        await store.init()
        try:
            session_id, turn_a = await _seed_interrupted_turn(store)
            turn_b = await _create_superseding_turn(store, session_id, 1)

            # Turn B completes: turn -> completed, session -> idle.
            await store.update_turn_status(
                turn_b,
                status="completed",
                finished_at_ms=int(datetime.now(UTC).timestamp() * 1000),
            )
            await _set_session_status(store, session_id, "idle")
            await store.flush()

            # Sanity: turn A is still stuck running.
            turns = await store.list_turns_by_session(session_id)
            by_id = {t.id: t for t in turns}
            assert by_id[turn_a].status == "running"

            session_manager = SessionManager(store)
            await session_manager.resume_session(session_id, load_turns=True)
            context_manager = ContextManager(
                tool_gateway=None,
                repo_service=None,
                memory_provider=MagicMock(),
                store=store,
                config_manager=MagicMock(),
            )

            # Default resume (no selector) must NOT offer the stale turn A.
            with pytest.raises(ResumeValidationError) as exc:
                await build_resume_plan(
                    store=store,
                    context_manager=context_manager,
                    session_manager=session_manager,
                    session_id=session_id,
                )
            assert "idle" in str(exc.value)

            # Even an explicit request for the stale turn A is rejected:
            # the session is idle, so there is no interrupted turn to
            # resume.
            with pytest.raises(ResumeValidationError) as exc:
                await build_resume_plan(
                    store=store,
                    context_manager=context_manager,
                    session_manager=session_manager,
                    session_id=session_id,
                    turn_id=turn_a,
                )
            assert "idle" in str(exc.value)
        finally:
            await store.close()

    async def test_resume_picks_latest_running_turn(self, tmp_path) -> None:
        """With two running turns, resume selects the latest (max index)."""
        store = ExecutionStore(tmp_path / "hygiene.db")
        await store.init()
        try:
            session_id, _turn_a = await _seed_interrupted_turn(store)
            turn_b = await _create_superseding_turn(store, session_id, 1)

            session_manager = SessionManager(store)
            await session_manager.resume_session(session_id, load_turns=True)
            context_manager = ContextManager(
                tool_gateway=None,
                repo_service=None,
                memory_provider=MagicMock(),
                store=store,
                config_manager=MagicMock(),
            )

            # Both turns are running; the default resume must pick turn B
            # (the latest), not the stale turn A.
            plan = await build_resume_plan(
                store=store,
                context_manager=context_manager,
                session_manager=session_manager,
                session_id=session_id,
            )
            assert plan.turn.turn_id == turn_b
        finally:
            await store.close()
