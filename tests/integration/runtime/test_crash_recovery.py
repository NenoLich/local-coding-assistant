"""Crash recovery integration tests for the runtime module.

Covers:
* Kill mid-LLM: frame left in RUNNING_LLM; resume re-enters the LLM phase and
  drives the frame to COMPLETED / turn to COMPLETED.
* Kill mid-tools: frame left in RUNNING_TOOLS with a recorded terminal tool
  outcome; resume skips the completed tool (idempotent) and executes only the
  pending call, then completes the turn.
* Kill after frame COMPLETED but before TurnCompleteEvent: frame is terminal
  while the turn is still RUNNING; resume starts a fresh frame and completes
  the turn.
* Resume with a running session/turn: a running session yields a loadable,
  resumable batch and turn.
* Frame status preservation: the stored frame status survives the simulated
  crash (RUNNING_LLM / RUNNING_TOOLS) and is resumed from that state.
* Idempotent event replay: build_frame reconstructs actions from stored events
  and the resume path does not re-execute already-terminal work.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from local_coding_assistant.agent.llm import LLMOptions, LLMTask, LLMToolCall
from local_coding_assistant.runtime.context.profile import RuntimeContextConfig
from local_coding_assistant.runtime.engine import (
    ActionKind,
    ExecutionStatus,
    FrameAgent,
    RuntimeEngine,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.engine.frame import build_frame
from local_coding_assistant.runtime.events.contracts import (
    EventType,
    RuntimeEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.runtime.resume import select_resumable_frame
from local_coding_assistant.runtime.session.turn import RuntimeTurn, TurnStatus
from local_coding_assistant.runtime.types import (
    ExecutionMode,
    PromptContext,
    RenderedPrompt,
)
from local_coding_assistant.storage.execution_store import ExecutionStore
from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ToolCallRequest,
    ToolExecutionResult,
    ToolExecutionStatus,
)


class MockLLMChunk:
    """Mock LLM chunk with attributes read by RuntimeEngine._update_chunk_metadata."""

    def __init__(
        self,
        content: str = "",
        reasoning: str = "",
        tool_calls: list[LLMToolCall] | None = None,
        model: str = "mock-model",
        provider: str = "mock-provider",
        total_tokens: int | None = 100,
        prompt_tokens: int | None = 50,
        completion_tokens: int | None = 50,
        reasoning_tokens: int | None = 0,
        latency_ms: float | None = 10.0,
        finish_reason: str = "stop",
    ):
        self.content = content
        self.reasoning = reasoning
        self.tool_calls = tool_calls or []
        self.model = model
        self.provider = provider
        self.total_tokens = total_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.latency_ms = latency_ms
        self.finish_reason = finish_reason
        self.usage = None
        self.metadata = None


class MockLLMService:
    """Mock LLM service that returns scripted responses."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.call_count = 0

    async def stream(self, task: LLMTask, *, options: LLMOptions | None = None):
        if self.responses:
            response_data = self.responses[
                min(self.call_count, len(self.responses) - 1)
            ]
        else:
            response_data = {"content": "Default response", "tool_calls": []}
        self.call_count += 1

        content = response_data.get("content", "")
        tool_calls_data = response_data.get("tool_calls", [])
        reasoning = response_data.get("reasoning", "")
        finish_reason = response_data.get("finish_reason", "stop")
        total_tokens = response_data.get("total_tokens", 100 if tool_calls_data else 50)
        prompt_tokens = response_data.get(
            "prompt_tokens", 50 if tool_calls_data else 25
        )
        completion_tokens = response_data.get(
            "completion_tokens", 50 if tool_calls_data else 25
        )
        reasoning_tokens = response_data.get("reasoning_tokens", 0)
        latency_ms = response_data.get("latency_ms", 10.0)

        if reasoning:
            yield MockLLMChunk(
                reasoning=reasoning,
                model="mock-model",
                provider="mock-provider",
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
            )

        if tool_calls_data:
            tc_objects = [
                LLMToolCall(
                    id=tc.get("id", str(uuid.uuid4())),
                    name=tc.get("name", "unknown"),
                    arguments=tc.get("arguments", {}),
                )
                for tc in tool_calls_data
            ]
            yield MockLLMChunk(
                content=content,
                tool_calls=tc_objects,
                model="mock-model",
                provider="mock-provider",
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
            )
        else:
            yield MockLLMChunk(
                content=content,
                model="mock-model",
                provider="mock-provider",
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
            )


class MockToolGateway:
    """Mock tool gateway that completes every call and records the requests."""

    def __init__(self) -> None:
        self._event_callback = None
        self.executed_calls: list[ToolCallRequest] = []

    def register_event_callback(self, callback) -> None:
        self._event_callback = callback

    async def execute_many(
        self,
        requests: list[ToolCallRequest],
        interrupt_event: Any = None,
    ) -> list[ToolExecutionResult]:
        self.executed_calls.extend(requests)
        results = []
        for request in requests:
            results.append(
                ToolExecutionResult(
                    status=ToolExecutionStatus.COMPLETED.value,
                    output=f"Mock output for {request.tool_id}",
                    backend=BackendType.HOST,
                    timing={"elapsed_seconds": 0.1},
                )
            )
        return results


class CapturingContextManager:
    """Mock ContextManager returning a fixed rendered prompt."""

    def __init__(self, rendered_prompt: RenderedPrompt | None = None):
        self._rendered = rendered_prompt or RenderedPrompt(
            system_messages=["You are a helpful assistant."],
            user_messages=["Test user input"],
            tool_schemas=[],
            history=[],
            metadata={"model": "mock-model", "temperature": 0.7},
        )
        self.resolve_call_count = 0
        self.captured_continuations: list[Any] = []

    async def resolve_context(
        self,
        session: Any,
        snapshot: Any,
        continuation: Any,
        config_snapshot: Any,
        history: list | None = None,
        history_through_seq: int | None = None,
    ) -> tuple[PromptContext, RenderedPrompt]:
        self.resolve_call_count += 1
        self.captured_continuations.append(continuation)
        prompt_ctx = PromptContext(
            session_id=getattr(session, "id", "session_123"),
            execution_mode=ExecutionMode.CLASSIC_TOOLS,
            tool_call_mode="classic",
        )
        return prompt_ctx, self._rendered

    async def clear_turn_cache(self, turn_id: str) -> None:
        return None


class MockEventPipeline:
    """Mock event pipeline that captures emitted events."""

    def __init__(self):
        self.events: list[RuntimeEvent] = []

    async def emit(self, event: RuntimeEvent) -> None:
        self.events.append(event)

    async def close(self) -> None:
        return None

    @property
    def subscribers(self) -> list:
        return []


def _new_id() -> str:
    return f"t_{uuid.uuid4().hex[:12]}"


def event_types(events: list[RuntimeEvent]) -> list[str]:
    return [e.type for e in events]


def find_event(events: list[RuntimeEvent], event_type) -> Any | None:
    return next((e for e in events if isinstance(e, event_type)), None)


async def _seed_crashed_turn(
    store: ExecutionStore,
    *,
    frame_status: str = "running_llm",
    turn_status: str = "running",
    llm_tool_calls: list[dict[str, Any]] | None = None,
    completed_tool: dict[str, Any] | None = None,
    llm_content: str = "reading the file",
) -> tuple[str, str, str, str]:
    """Seed a session/turn/frame as if a crash interrupted execution.

    The session and turn are left in "running" status and the frame is left in
    ``frame_status`` (RUNNING_LLM / RUNNING_TOOLS / COMPLETED), simulating a
    kill before TurnCompleteEvent was persisted.

    Returns (session_id, turn_id, _snapshot_id, frame_id).
    """
    session_id = _new_id()
    turn_id = f"{session_id}_turn_0"
    snapshot_id = f"snap_{turn_id}"
    frame_id = f"{turn_id}_frame_1"
    trace_id = _new_id()
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    await store.create_session(
        session_id=session_id,
        user_input="crash recovery test",
        agent_mode="frame_agent",
        model_name="mock-model",
    )

    await store.create_turn(
        turn_id=turn_id,
        session_id=session_id,
        trace_id=trace_id,
        turn_index=0,
        user_input="crash recovery test",
        status=turn_status,
        config_snapshot_json=json.dumps(
            {
                "context_config": RuntimeContextConfig().model_dump(),
                "limits_config": RuntimeLimitsConfig().model_dump(),
            }
        ),
        context_snapshot_id=snapshot_id,
    )

    tool_id = await store.register_tool(
        name="read_file",
        schema_json=json.dumps(
            {"type": "object", "properties": {"path": {"type": "string"}}}
        ),
        risk_class="low",
    )
    template_id = await store.register_template(
        path="base/sys.jinja2", content="System", content_hash="hash_tpl"
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
        tool_set_ref=json.dumps(
            {"tool_ids": [tool_id], "tool_names": ["read_file"], "version": "1.0.0"}
        ),
        template_set_ref=json.dumps(
            {
                "template_ids": [template_id],
                "template_paths": ["base/sys.jinja2"],
                "version": "1.0.0",
            }
        ),
        repo_snapshot_ref=json.dumps(
            {"target_project_root": "project_root", "repo_snapshot_hash": "repo_hash"}
        ),
        memory_snapshot_ref=json.dumps({"memory_ids": [], "version": "v"}),
        content_hash="snap_hash",
        created_at_ms=now_ms,
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

    llm_calls = llm_tool_calls or []
    await store.append_event(
        trace_id=trace_id,
        session_id=session_id,
        turn_id=turn_id,
        frame_id=frame_id,
        seq=1,
        event_type=EventType.LLM_COMPLETE,
        data_json=json.dumps(
            {
                "content": llm_content,
                "tool_calls": llm_calls,
                "model": "mock-model",
                "total_tokens": 42,
                "prompt_tokens": 20,
                "completion_tokens": 20,
                "finish_reason": "stop",
                "latency_ms": 10.0,
            }
        ),
        timestamp_ms=now_ms,
    )

    seq = 2
    if completed_tool is not None:
        await store.append_event(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=seq,
            event_type=EventType.TOOL_RESULT,
            data_json=json.dumps(
                {
                    "call_id": completed_tool["call_id"],
                    "tool_id": completed_tool["tool_id"],
                    "output": completed_tool.get("output", "ok"),
                    "success": True,
                    "timing": {},
                }
            ),
            timestamp_ms=now_ms,
        )
        seq += 1

    if frame_status == "completed":
        await store.append_event(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=seq,
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
            timestamp_ms=now_ms,
        )

    await store.flush()
    return session_id, turn_id, snapshot_id, frame_id


async def _make_frame_agent(
    llm_service: Any,
    tool_gateway: Any,
    event_pipeline: MockEventPipeline,
    context_manager: CapturingContextManager,
) -> FrameAgent:
    engine = RuntimeEngine(
        llm_service=llm_service,
        tool_gateway=tool_gateway,
        event_pipeline=event_pipeline,
    )
    return FrameAgent(
        engine=engine,
        context_manager=context_manager,
        event_pipeline=event_pipeline,
    )


class TestCrashRecoveryMidLLM:
    async def test_resume_from_running_llm_completes_turn(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "crash_mid_llm.db")
        await store.init()
        try:
            session_id, turn_id, _snapshot_id, frame_id = await _seed_crashed_turn(
                store, frame_status="running_llm"
            )

            # Frame status is preserved across the simulated crash.
            stored_frame = await store.frame_repo.get(frame_id)
            assert stored_frame is not None
            assert stored_frame.status == ExecutionStatus.RUNNING_LLM.value

            batch = await store.load_resume_batch(session_id, turn_id=turn_id)
            assert batch is not None
            assert batch.session.status == "running"
            assert batch.turn.status == "running"

            turn = RuntimeTurn.from_batch(batch)
            assert turn.status == TurnStatus.RUNNING
            assert turn.is_resumed is True
            assert frame_id in turn.frame_ids

            frame, warnings = select_resumable_frame(batch, None)
            assert frame is not None
            assert frame.id == frame_id
            assert frame.status == ExecutionStatus.RUNNING_LLM
            assert warnings == []

            llm_service = MockLLMService(
                responses=[{"content": "Recovered after crash", "tool_calls": []}]
            )
            tool_gateway = MockToolGateway()
            event_pipeline = MockEventPipeline()
            context_manager = CapturingContextManager()
            frame_agent = await _make_frame_agent(
                llm_service, tool_gateway, event_pipeline, context_manager
            )
            session = SimpleNamespace(
                id=session_id, history=[], history_sequence=0, context_profile=None
            )

            await frame_agent.resume_turn(turn, session, frame_to_resume=frame)

            assert turn.status == TurnStatus.COMPLETED
            assert turn.final_answer == "Recovered after crash"
            assert len(turn.frame_ids) == 1  # same frame resumed, no new frame
            assert len(tool_gateway.executed_calls) == 0
            assert llm_service.call_count == 1

            turn_complete = find_event(event_pipeline.events, TurnCompleteEvent)
            assert turn_complete is not None
            assert turn_complete.status == "completed"
            assert turn_complete.total_frames == 1
            assert turn_complete.finish_reason == "final_answer"
            # resume_turn does not re-emit TurnStartEvent.
            assert not any(isinstance(e, TurnStartEvent) for e in event_pipeline.events)
        finally:
            await store.close()


class TestCrashRecoveryMidTools:
    async def test_resume_from_running_tools_skips_completed_tool(
        self, tmp_path
    ) -> None:
        store = ExecutionStore(tmp_path / "crash_mid_tools.db")
        await store.init()
        try:
            completed_tool = {
                "call_id": "call_done",
                "tool_id": "read_file",
                "output": "already done",
            }
            session_id, turn_id, _snapshot_id, frame_id = await _seed_crashed_turn(
                store,
                frame_status="running_tools",
                llm_tool_calls=[
                    {
                        "id": "call_done",
                        "name": "read_file",
                        "arguments": {"path": "a.txt"},
                    },
                    {
                        "id": "call_pending",
                        "name": "read_file",
                        "arguments": {"path": "b.txt"},
                    },
                ],
                completed_tool=completed_tool,
            )

            # Reconstruct the frame from stored events: call_done is terminal,
            # call_pending was never executed.
            stored = await store.frame_repo.get(frame_id)
            events = await store.event_repo.get_by_frame(frame_id)
            frame = build_frame(stored, events)
            assert frame.status == ExecutionStatus.RUNNING_TOOLS
            tool_actions = [a for a in frame.actions if a.kind == ActionKind.TOOL_CALL]
            assert len(tool_actions) == 1
            assert tool_actions[0].tool_execution_data["status"] == "completed"
            assert tool_actions[0].tool_execution_data["call_id"] == "call_done"

            batch = await store.load_resume_batch(session_id, turn_id=turn_id)
            turn = RuntimeTurn.from_batch(batch)
            assert turn.status == TurnStatus.RUNNING

            frame, warnings = select_resumable_frame(batch, None)
            assert frame is not None
            assert frame.id == frame_id
            assert frame.status == ExecutionStatus.RUNNING_TOOLS
            assert warnings == []

            llm_service = MockLLMService(
                responses=[
                    {"content": "Final answer after tool result", "tool_calls": []}
                ]
            )
            tool_gateway = MockToolGateway()
            event_pipeline = MockEventPipeline()
            context_manager = CapturingContextManager()
            frame_agent = await _make_frame_agent(
                llm_service, tool_gateway, event_pipeline, context_manager
            )
            session = SimpleNamespace(
                id=session_id, history=[], history_sequence=0, context_profile=None
            )

            await frame_agent.resume_turn(turn, session, frame_to_resume=frame)

            assert turn.status == TurnStatus.COMPLETED
            assert turn.final_answer == "Final answer after tool result"
            # Idempotency: only the pending call was executed; the completed call
            # was skipped because its terminal outcome was already recorded.
            assert len(tool_gateway.executed_calls) == 1
            assert tool_gateway.executed_calls[0].call_id == "call_pending"
            assert tool_gateway.executed_calls[0].tool_id == "read_file"
            assert len(turn.frame_ids) == 2  # resumed frame + continuation frame

            turn_complete = find_event(event_pipeline.events, TurnCompleteEvent)
            assert turn_complete is not None
            assert turn_complete.status == "completed"
            assert turn_complete.total_frames == 2
            assert not any(isinstance(e, TurnStartEvent) for e in event_pipeline.events)
        finally:
            await store.close()


class TestCrashRecoveryAfterFrameComplete:
    async def test_resume_after_frame_complete_starts_new_frame(self, tmp_path) -> None:
        store = ExecutionStore(tmp_path / "crash_after_complete.db")
        await store.init()
        try:
            # Frame is COMPLETED (terminal) but the turn never got TurnCompleteEvent.
            session_id, turn_id, _snapshot_id, frame_id = await _seed_crashed_turn(
                store,
                frame_status="completed",
                turn_status="running",
                llm_tool_calls=[],
                llm_content="done",
            )

            stored = await store.frame_repo.get(frame_id)
            assert stored.status == ExecutionStatus.COMPLETED.value

            batch = await store.load_resume_batch(session_id, turn_id=turn_id)
            assert batch is not None
            assert batch.session.status == "running"
            assert batch.turn.status == "running"

            turn = RuntimeTurn.from_batch(batch)
            assert turn.status == TurnStatus.RUNNING
            assert frame_id in turn.frame_ids

            # Terminal last frame -> select_resumable_frame returns None and a new
            # frame will be started on resume.
            frame, warnings = select_resumable_frame(batch, None)
            assert frame is None
            assert len(warnings) == 1
            assert "terminal status completed" in warnings[0]

            llm_service = MockLLMService(
                responses=[{"content": "Fresh answer after resume", "tool_calls": []}]
            )
            tool_gateway = MockToolGateway()
            event_pipeline = MockEventPipeline()
            context_manager = CapturingContextManager()
            frame_agent = await _make_frame_agent(
                llm_service, tool_gateway, event_pipeline, context_manager
            )
            session = SimpleNamespace(
                id=session_id, history=[], history_sequence=0, context_profile=None
            )

            await frame_agent.resume_turn(turn, session, frame_to_resume=None)

            assert turn.status == TurnStatus.COMPLETED
            assert turn.final_answer == "Fresh answer after resume"
            assert len(turn.frame_ids) == 2  # original completed frame + new frame
            assert frame_id in turn.frame_ids

            turn_complete = find_event(event_pipeline.events, TurnCompleteEvent)
            assert turn_complete is not None
            assert turn_complete.status == "completed"
            assert turn_complete.total_frames == 2
        finally:
            await store.close()


class TestResumeRunningStatus:
    async def test_running_session_is_resumable(self, tmp_path) -> None:
        """A session left in 'running' status yields a loadable, resumable turn."""
        store = ExecutionStore(tmp_path / "resume_running.db")
        await store.init()
        try:
            session_id, turn_id, _snapshot_id, frame_id = await _seed_crashed_turn(
                store, frame_status="running_llm", turn_status="running"
            )

            batch = await store.load_resume_batch(session_id, turn_id=turn_id)
            assert batch is not None
            assert batch.session.status == "running"
            assert batch.turn.status == "running"

            turn = RuntimeTurn.from_batch(batch)
            assert turn.status == TurnStatus.RUNNING
            assert turn.is_resumed is True

            frame, warnings = select_resumable_frame(batch, None)
            assert frame is not None
            assert frame.id == frame_id
            assert frame.status == ExecutionStatus.RUNNING_LLM
            assert warnings == []
        finally:
            await store.close()


class TestFrameStatusPreservation:
    @pytest.mark.parametrize("frame_status", ["running_llm", "running_tools"])
    async def test_frame_status_survives_crash(
        self, tmp_path, frame_status: str
    ) -> None:
        store = ExecutionStore(tmp_path / f"preserve_{frame_status}.db")
        await store.init()
        try:
            _session_id, _turn_id, _snapshot_id, frame_id = await _seed_crashed_turn(
                store, frame_status=frame_status
            )

            stored = await store.frame_repo.get(frame_id)
            assert stored is not None
            assert stored.status == frame_status
        finally:
            await store.close()


class TestIdempotentEventReplay:
    async def test_build_frame_reconstructs_recorded_outcomes(self, tmp_path) -> None:
        """build_frame reproduces actions from stored events without re-execution."""
        store = ExecutionStore(tmp_path / "idempotent.db")
        await store.init()
        try:
            completed_tool = {
                "call_id": "call_done",
                "tool_id": "read_file",
                "output": "already done",
            }
            _session_id, _turn_id, _snapshot_id, frame_id = await _seed_crashed_turn(
                store,
                frame_status="running_tools",
                llm_tool_calls=[
                    {
                        "id": "call_done",
                        "name": "read_file",
                        "arguments": {"path": "a.txt"},
                    },
                    {
                        "id": "call_pending",
                        "name": "read_file",
                        "arguments": {"path": "b.txt"},
                    },
                ],
                completed_tool=completed_tool,
            )

            stored = await store.frame_repo.get(frame_id)
            events = await store.event_repo.get_by_frame(frame_id)
            frame = build_frame(stored, events)

            assert frame.status == ExecutionStatus.RUNNING_TOOLS
            assert frame.model_response_raw == "reading the file"

            # The LLM action carries both tool calls (from the pre-crash LLM_COMPLETE).
            llm_action = next(
                a for a in frame.actions if a.kind == ActionKind.LLM_MESSAGE
            )
            assert [tc.id for tc in llm_action.tool_calls] == [
                "call_done",
                "call_pending",
            ]

            # Only call_done has a recorded terminal outcome; call_pending is absent.
            tool_actions = [a for a in frame.actions if a.kind == ActionKind.TOOL_CALL]
            assert len(tool_actions) == 1
            assert tool_actions[0].tool_execution_data["call_id"] == "call_done"
            assert tool_actions[0].tool_execution_data["status"] == "completed"

            # Resume must not re-execute call_done (proven by the mid-tools test);
            # here we assert the stored event log itself is the source of truth and
            # contains exactly one TOOL_RESULT for call_done and none for call_pending.
            result_events = [
                e for e in events if e["type"] == EventType.TOOL_RESULT.value
            ]
            assert len(result_events) == 1
            assert json.loads(result_events[0]["data_json"])["call_id"] == "call_done"
        finally:
            await store.close()
