"""
Integration tests for RuntimeManager entrypoints.

Covers the public entrypoint surface of ``RuntimeManager``:
- ``start_session``: new session, first turn, turn run, event emission
- ``resume_session``: in-place resume, branching, validation errors
- ``resume_session_with_message``: new turn creation, superseding, branching
- ``respond_approval``: approval response application and resume
- ``interrupt_session``: interrupt a running turn
- ``apply_run_options``: session override translation
- ``start``/``stop`` lifecycle

The real ``SessionManager``, ``TurnManager`` and ``ContextManager`` are used
against a real SQLite-backed ``ExecutionStore``. The ``FrameAgent`` is mocked
out so LLM/tool execution (covered by ``test_frame_agent_loop`` and
``test_runtime_engine_e2e``) is not exercised here; the focus is on the
entrypoints' orchestration of session/turn creation, event emission, and
store interaction.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
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
from local_coding_assistant.runtime.engine.approval import (
    ApprovalRequest,
    ApprovalStatus,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.engine.frame import ExecutionStatus
from local_coding_assistant.runtime.events.contracts import (
    ApprovalResponseForm,
    EventType,
    SessionInterruptEvent,
    SessionResumeEvent,
    SessionStartEvent,
)
from local_coding_assistant.runtime.manager import RuntimeManager
from local_coding_assistant.runtime.resume import ResumeValidationError
from local_coding_assistant.runtime.session.turn import TurnStatus
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


class _StubConfigManager:
    """Minimal config manager with a real AppConfig (static resolution)."""

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
        return "mock-provider"


class MockEventPipeline:
    """Mock event pipeline that captures emitted events."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.subscriptions: list[Any] = []

    async def emit(self, event: Any) -> None:
        self.events.append(event)

    async def close(self) -> None:
        pass

    def subscribe(self, subscriber: Any) -> None:
        self.subscriptions.append(subscriber)

    @property
    def subscribers(self) -> list[Any]:
        return []


def _find_event(events: list[Any], event_type: type) -> Any | None:
    return next((e for e in events if isinstance(e, event_type)), None)


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
    """Seed one turn: snapshot + single frame + event."""
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


async def _seed_session(
    store: ExecutionStore,
    *,
    session_id: str | None = None,
    user_input: str = "first",
    status: str = "idle",
) -> str:
    """Seed a session row with a stored context profile."""
    session_id = session_id or str(uuid.uuid4())
    await store._write_batch(
        [
            {
                "op": "upsert_session",
                "data": {
                    "session_id": session_id,
                    "user_input": user_input,
                    "status": status,
                    "started_at_ms": int(datetime.now(UTC).timestamp() * 1000),
                    "agent_mode": "frame_agent",
                    "model_name": "gpt-4",
                    "context_profile_json": _profile_json(),
                },
            }
        ]
    )
    await store._conn.commit()
    return session_id


def _make_runtime(
    store: ExecutionStore,
) -> tuple[RuntimeManager, MockEventPipeline]:
    """RuntimeManager over the real store with the frame agent mocked out."""
    pipeline = MockEventPipeline()
    runtime = RuntimeManager(
        config_manager=_StubConfigManager(),
        store=store,
        event_pipeline=pipeline,
    )
    runtime._frame_agent = MagicMock()
    runtime._frame_agent.run_turn = AsyncMock()
    runtime._frame_agent.resume_turn = AsyncMock()
    return runtime, pipeline


@pytest.fixture
async def store(tmp_path: Any) -> ExecutionStore:
    execution_store = ExecutionStore(tmp_path / "runtime_manager.db")
    await execution_store.init()
    yield execution_store
    await execution_store.close()


class TestRuntimeManagerStartSession:
    """Tests for the ``start_session`` entrypoint."""

    async def test_start_session_creates_session_and_turn(
        self, store: ExecutionStore
    ) -> None:
        runtime, pipeline = _make_runtime(store)

        session = await runtime.start_session("Hello, world!")

        # Session was created and cached.
        assert session.id is not None
        cached = runtime._session_manager.get_cached(session.id)
        assert cached is session
        assert len(session.turns) == 1

        turn = session.get_last_turn()
        assert turn is not None
        assert turn.turn_index == 0
        assert turn.user_input == "Hello, world!"
        assert turn.status == TurnStatus.RUNNING

        # Frame agent was asked to run the turn.
        runtime._frame_agent.run_turn.assert_awaited_once_with(turn, session)

        # SessionStartEvent was emitted.
        start_event = _find_event(pipeline.events, SessionStartEvent)
        assert start_event is not None
        assert start_event.session_id == session.id
        assert start_event.turn_id == turn.turn_id
        assert start_event.user_input == "Hello, world!"
        assert start_event.agent_mode == "frame_agent"

        # A turn was recorded for retention tracking.
        # (record_turn is internal to SessionManager; verified via cache.)

    async def test_start_session_applies_session_metadata(
        self, store: ExecutionStore
    ) -> None:
        runtime, _pipeline = _make_runtime(store)
        metadata = {"key": "value", "source": "test"}

        session = await runtime.start_session("Hello", session_metadata=metadata)

        assert session.metadata == metadata

        # The SessionStartEvent carried the serialized metadata.
        start_event = _find_event(
            runtime.event_pipeline.events if runtime.event_pipeline else [],
            SessionStartEvent,
        )
        if start_event is not None:
            assert start_event.metadata_json == json.dumps(metadata)

    async def test_start_session_accepts_trace_id(self, store: ExecutionStore) -> None:
        runtime, pipeline = _make_runtime(store)
        trace_id = "trace_custom_123"

        await runtime.start_session("Hello", trace_id=trace_id)

        start_event = _find_event(pipeline.events, SessionStartEvent)
        assert start_event is not None
        assert start_event.trace_id == trace_id

    async def test_start_session_raises_when_not_initialized(self) -> None:
        """Constructing without a store leaves entrypoints uninitialized."""
        runtime = RuntimeManager(config_manager=_StubConfigManager())
        assert runtime._session_manager is None
        assert runtime._turn_manager is None
        assert runtime._frame_agent is None

        with pytest.raises(RuntimeError, match="not fully initialized"):
            await runtime.start_session("Hello")


class TestRuntimeManagerApplyRunOptions:
    """Tests for the ``apply_run_options`` entrypoint."""

    def test_apply_run_options_sets_session_overrides(
        self, store: ExecutionStore
    ) -> None:
        runtime, _pipeline = _make_runtime(store)

        class FakeOptions:
            def to_session_overrides(self) -> dict[str, Any]:
                return {"llm.temperature": 0.9, "llm.max_tokens": 2048}

        runtime.apply_run_options(FakeOptions())

        assert runtime.config_manager.session_overrides == {
            "llm.temperature": 0.9,
            "llm.max_tokens": 2048,
        }


class TestRuntimeManagerResumeSession:
    """Tests for the ``resume_session`` entrypoint."""

    async def test_resume_last_turn_in_place(self, store: ExecutionStore) -> None:
        """Resuming with no selector picks the last turn without branching."""
        session_id = await _seed_session(store, status="running")
        turn = await _seed_turn(
            store,
            session_id=session_id,
            turn_index=0,
            user_input="first",
            turn_status="running",
            frame_status=ExecutionStatus.RUNNING_TOOLS,
            paused=True,
        )

        runtime, pipeline = _make_runtime(store)

        session = await runtime.resume_session(session_id)

        assert session.id == session_id

        turn_arg, resumed_session, frame = (
            runtime._frame_agent.resume_turn.await_args.args
        )
        assert resumed_session.id == session_id
        assert turn_arg.turn_id == turn["turn_id"]
        assert turn_arg.turn_index == 0
        assert frame is not None
        assert frame.status == ExecutionStatus.RUNNING_TOOLS

        # A SessionResumeEvent was emitted.
        resume_event = _find_event(pipeline.events, SessionResumeEvent)
        assert resume_event is not None
        assert resume_event.session_id == session_id
        assert resume_event.frame_id == frame.id

        # No branch event was emitted.
        branch_events = [
            e
            for e in pipeline.events
            if isinstance(
                e,
                __import__(
                    "local_coding_assistant.runtime.events.contracts",
                    fromlist=["SessionBranchEvent"],
                ).SessionBranchEvent,
            )
        ]
        assert len(branch_events) == 0

    async def test_resume_non_last_turn_branches(self, store: ExecutionStore) -> None:
        """Resuming a non-last turn auto-branches to a new session."""
        session_id = await _seed_session(store, status="running")
        t0 = await _seed_turn(
            store,
            session_id=session_id,
            turn_index=0,
            user_input="first",
            turn_status="running",
            frame_status=ExecutionStatus.RUNNING_TOOLS,
            paused=True,
        )
        await _seed_turn(
            store,
            session_id=session_id,
            turn_index=1,
            user_input="second",
            turn_status="running",
            frame_status=ExecutionStatus.RUNNING_TOOLS,
        )

        runtime, pipeline = _make_runtime(store)

        session = await runtime.resume_session(session_id, turn_id=t0["turn_id"])

        # A forked session was created.
        assert session.id != session_id

        turn_arg, resumed_session, frame = (
            runtime._frame_agent.resume_turn.await_args.args
        )
        assert resumed_session.id == session.id
        # The turn is addressed by its preserved turn_index (IDs regenerated).
        assert turn_arg.turn_index == 0
        assert turn_arg.turn_id != t0["turn_id"]
        assert frame is not None

        # A SessionBranchEvent was emitted.
        from local_coding_assistant.runtime.events.contracts import SessionBranchEvent

        branch_event = _find_event(pipeline.events, SessionBranchEvent)
        assert branch_event is not None
        assert branch_event.new_session_id == session.id
        assert branch_event.parent_session_id == session_id

    async def test_resume_completed_turn_raises(self, store: ExecutionStore) -> None:
        """Resuming a completed (non-interrupted) turn is rejected."""
        session_id = await _seed_session(store, status="idle")
        await _seed_turn(
            store,
            session_id=session_id,
            turn_index=0,
            user_input="first",
            turn_status="completed",
            frame_status=ExecutionStatus.COMPLETED,
        )

        runtime, _pipeline = _make_runtime(store)

        with pytest.raises(ResumeValidationError) as exc:
            await runtime.resume_session(session_id, turn_index=0)

        assert "idle" in str(exc.value).lower()


class TestRuntimeManagerResumeSessionWithMessage:
    """Tests for the ``resume_session_with_message`` entrypoint."""

    async def test_resume_with_message_creates_new_turn(
        self, store: ExecutionStore
    ) -> None:
        """Continuing an idle session creates a new turn in the same session."""
        session_id = await _seed_session(store, status="idle")
        await _seed_turn(
            store,
            session_id=session_id,
            turn_index=0,
            user_input="first",
            turn_status="completed",
            frame_status=ExecutionStatus.COMPLETED,
        )

        runtime, _pipeline = _make_runtime(store)

        session = await runtime.resume_session_with_message(
            session_id, "second message"
        )

        assert session.id == session_id
        # A new turn was created and run.
        runtime._frame_agent.run_turn.assert_awaited_once()
        turn_arg, session_arg = runtime._frame_agent.run_turn.await_args.args
        assert session_arg.id == session_id
        assert turn_arg.user_input == "second message"
        assert turn_arg.turn_index == 1

        # Turn was recorded for retention.
        assert len(session.turns) >= 1

    async def test_resume_with_message_supersedes_prior_running_turn(
        self, store: ExecutionStore
    ) -> None:
        """A prior running turn is finalized as superseded."""
        session_id = await _seed_session(store, status="running")
        turn = await _seed_turn(
            store,
            session_id=session_id,
            turn_index=0,
            user_input="first",
            turn_status="running",
            frame_status=ExecutionStatus.RUNNING_TOOLS,
            paused=True,
        )

        runtime, _pipeline = _make_runtime(store)

        await runtime.resume_session_with_message(session_id, "second message")

        # The prior interrupted turn is finalized as superseded.
        stored = await store.get_turn(turn["turn_id"])
        assert stored is not None
        assert stored.status == "superseded"
        assert stored.finished_at_ms is not None

    async def test_resume_with_message_non_last_turn_branches(
        self, store: ExecutionStore
    ) -> None:
        """Branching from a non-last turn creates a forked session."""
        session_id = await _seed_session(store, status="running")
        t0 = await _seed_turn(
            store,
            session_id=session_id,
            turn_index=0,
            user_input="first",
            turn_status="running",
            frame_status=ExecutionStatus.RUNNING_TOOLS,
            paused=True,
        )
        await _seed_turn(
            store,
            session_id=session_id,
            turn_index=1,
            user_input="second",
            turn_status="running",
            frame_status=ExecutionStatus.RUNNING_TOOLS,
        )

        runtime, pipeline = _make_runtime(store)

        session = await runtime.resume_session_with_message(
            session_id, "new message", turn_id=t0["turn_id"]
        )

        assert session.id != session_id

        from local_coding_assistant.runtime.events.contracts import SessionBranchEvent

        branch_event = _find_event(pipeline.events, SessionBranchEvent)
        assert branch_event is not None
        assert branch_event.new_session_id == session.id

    async def test_resume_with_message_skips_terminal_prior_turn(
        self, store: ExecutionStore
    ) -> None:
        """A completed prior turn is left untouched (no supersede)."""
        session_id = await _seed_session(store, status="idle")
        turn = await _seed_turn(
            store,
            session_id=session_id,
            turn_index=0,
            user_input="first",
            turn_status="completed",
            frame_status=ExecutionStatus.COMPLETED,
        )

        runtime, _pipeline = _make_runtime(store)

        await runtime.resume_session_with_message(session_id, "second message")

        # Completed turn keeps its status.
        stored = await store.get_turn(turn["turn_id"])
        assert stored is not None
        assert stored.status == "completed"


class TestRuntimeManagerRespondApproval:
    """Tests for the ``respond_approval`` entrypoint."""

    async def test_respond_approval_resumes_turn(self, store: ExecutionStore) -> None:
        """Applying approval responses resumes the paused turn."""
        session_id = await _seed_session(store, status="running")
        turn = await _seed_turn(
            store,
            session_id=session_id,
            turn_index=0,
            user_input="first",
            turn_status="running",
            frame_status=ExecutionStatus.RUNNING_TOOLS,
            paused=True,
        )

        # Load and cache the session (respond_approval is cache-only).
        session = await runtime_load_session(store, session_id)
        runtime, pipeline = _make_runtime(store)
        runtime._session_manager.cache_session(session)

        token = "approval_token_123"  # noqa: S105 - test fixture token, not a real secret
        tool_call_id = "call_1"

        # Seed a pending approval on the turn (as run_turn would have done).
        turn_obj = session.get_turn(turn["turn_id"])
        assert turn_obj is not None
        turn_obj.pending_approvals[tool_call_id] = ApprovalRequest(
            token=token,
            tool_call_id=tool_call_id,
            tool_name="read_file",
            arguments={"path": "a"},
            risk_class="high",
            session_id=session_id,
            frame_id=turn["frame_id"],
            turn_id=turn["turn_id"],
            status=ApprovalStatus.PENDING,
            response=None,
            timeout_seconds=300,
        )

        await runtime.respond_approval(
            session_id,
            turn["turn_id"],
            [
                ApprovalResponseForm(
                    approval_token=token,
                    tool_call_id=tool_call_id,
                    response="yes_once",
                )
            ],
            frame_id=turn["frame_id"],
        )

        # Frame agent was asked to resume the turn.
        runtime._frame_agent.resume_turn.assert_awaited_once()
        _resume_turn_arg, resume_session_arg, resume_frame_arg = (
            runtime._frame_agent.resume_turn.await_args.args
        )
        assert resume_session_arg.id == session_id
        assert resume_frame_arg is not None

        # A SessionResumeEvent was emitted.
        resume_event = _find_event(pipeline.events, SessionResumeEvent)
        assert resume_event is not None
        assert resume_event.session_id == session_id
        assert resume_event.frame_id == resume_frame_arg.id

    async def test_respond_approval_session_not_cached_raises(
        self, store: ExecutionStore
    ) -> None:
        """respond_approval requires the session to be in the cache."""
        session_id = await _seed_session(store, status="running")
        runtime, _pipeline = _make_runtime(store)

        with pytest.raises(ValueError, match="not available for approval response"):
            await runtime.respond_approval(
                session_id,
                "nonexistent_turn",
                [
                    ApprovalResponseForm(
                        approval_token="t",  # noqa: S106 - test fixture token, not a real secret
                        tool_call_id="c",
                        response="yes_once",
                    ),
                ],
            )

    async def test_respond_approval_turn_not_found_raises(
        self, store: ExecutionStore
    ) -> None:
        """respond_approval raises when the turn is not in the session."""
        session_id = await _seed_session(store, status="running")
        await _seed_turn(
            store,
            session_id=session_id,
            turn_index=0,
            user_input="first",
            turn_status="running",
            frame_status=ExecutionStatus.RUNNING_TOOLS,
            paused=True,
        )

        session = await runtime_load_session(store, session_id)
        runtime, _pipeline = _make_runtime(store)
        runtime._session_manager.cache_session(session)

        with pytest.raises(ValueError, match="not found in session"):
            await runtime.respond_approval(
                session_id,
                "nonexistent_turn",
                [
                    ApprovalResponseForm(
                        approval_token="t",  # noqa: S106 - test fixture token, not a real secret
                        tool_call_id="c",
                        response="yes_once",
                    ),
                ],
            )


class TestRuntimeManagerInterruptSession:
    """Tests for the ``interrupt_session`` entrypoint."""

    async def test_interrupt_session_pauses_running_turn(
        self, store: ExecutionStore
    ) -> None:
        """Interrupting a running turn pauses it and emits an interrupt event."""
        session_id = await _seed_session(store, status="running")
        turn = await _seed_turn(
            store,
            session_id=session_id,
            turn_index=0,
            user_input="first",
            turn_status="running",
            frame_status=ExecutionStatus.RUNNING_TOOLS,
            paused=False,
        )

        # Cache the session with a RUNNING turn.
        session = await runtime_load_session(store, session_id)
        # Force the turn status to RUNNING (interrupt checks status).
        last = session.get_last_turn()
        assert last is not None
        last.status = TurnStatus.RUNNING

        runtime, pipeline = _make_runtime(store)
        runtime._session_manager.cache_session(session)

        result = await runtime.interrupt_session(session_id)

        assert result.id == session_id

        # The turn was interrupted in memory.
        assert last.pause_reason == "interrupt"
        assert last.paused_at is not None
        assert last.interrupt_event.is_set()

        # A SessionInterruptEvent was emitted.
        interrupt_event = _find_event(pipeline.events, SessionInterruptEvent)
        assert interrupt_event is not None
        assert interrupt_event.session_id == session_id
        assert interrupt_event.turn_id == turn["turn_id"]
        assert interrupt_event.reason == "interrupt"

    async def test_interrupt_session_no_running_turn_raises(
        self, store: ExecutionStore
    ) -> None:
        """Interrupting a session with no running turn raises."""
        session_id = await _seed_session(store, status="idle")
        await _seed_turn(
            store,
            session_id=session_id,
            turn_index=0,
            user_input="first",
            turn_status="completed",
            frame_status=ExecutionStatus.COMPLETED,
        )

        session = await runtime_load_session(store, session_id)
        runtime, _pipeline = _make_runtime(store)
        runtime._session_manager.cache_session(session)

        with pytest.raises(ValueError, match="No running turn found"):
            await runtime.interrupt_session(session_id)


class TestRuntimeManagerLifecycle:
    """Tests for ``start``/``stop`` lifecycle methods."""

    def test_start_and_stop(self, store: ExecutionStore) -> None:
        runtime, _pipeline = _make_runtime(store)

        runtime.start()  # should not raise
        runtime.stop()  # should not raise


async def runtime_load_session(store: ExecutionStore, session_id: str) -> Any:
    """Load a session with turns from the store for caching."""
    from local_coding_assistant.runtime.session.session_manager import SessionManager

    session_manager = SessionManager(store)
    return await session_manager.resume_session(session_id, load_turns=True)
