"""Integration tests for the FrameAgent full turn loop.

Covers:
* run_turn with a mocked IExecutionEngine: simple completion, multi-frame continue
* resume_turn: approval pause/resume with frame_to_resume
* Frame count limit enforcement (FRAME_LIMIT_REACHED)
* Turn limits enforcement: token budget, deadline
* Continuation decisions applied: CONTINUE, RETRY_GENERATION, COMPACT_AND_CONTINUE, STOP
* FrameContinuation directives captured via context manager: tool observations,
  compaction, retry params, approval
* TurnCompleteEvent with correct metrics (total_input_tokens, total_output_tokens,
  total_frames, finish_reason)
* Event ordering: turn_start -> frame* -> turn_complete
* Exception handling: transient error resumability through the turn loop
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from local_coding_assistant.agent.llm import LLMOptions, LLMTask, LLMToolCall
from local_coding_assistant.config.schemas import (
    TokenBudgetConfig,
)
from local_coding_assistant.runtime.context.profile import (
    ContextSnapshot,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.engine import (
    ActionKind,
    ExecutionStatus,
    FrameAgent,
    IExecutionEngine,
    RuntimeEngine,
    RuntimeFrame,
    RuntimeLimits,
    RuntimeLimitsConfig,
)
from local_coding_assistant.runtime.engine.approval import (
    ApprovalStatus,
)
from local_coding_assistant.runtime.engine.continuation import (
    FrameContinuation,
)
from local_coding_assistant.runtime.engine.limits import (
    LoopDetectionConfig,
    LoopGuard,
    TokenBudget,
)
from local_coding_assistant.runtime.events.contracts import (
    ApprovalRequiredEvent,
    ApprovalResponseForm,
    ContinuationDecisionEvent,
    RuntimeEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.runtime.session.turn import RuntimeTurn, TurnStatus
from local_coding_assistant.runtime.types import (
    ExecutionMode,
    PromptContext,
    RenderedPrompt,
)
from local_coding_assistant.tools_platform.contracts import (
    ApprovalResponse as ToolApprovalResponse,
)
from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ToolCallRequest,
    ToolDecision,
    ToolDecisionOutcome,
    ToolEvent,
    ToolEventType,
    ToolExecutionResult,
    ToolExecutionStatus,
)

# ============================================================================
# Mock LLM service / chunks
# ============================================================================


class MockLLMChunk:
    """Mock LLM chunk with attributes read by RuntimeEngine._update_chunk_metadata."""

    def __init__(
        self,
        content: str = "",
        reasoning: str = "",
        tool_calls: list[LLMToolCall] | None = None,
        model: str = "mock-model",
        provider: str = "mock-provider",
        total_tokens: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        latency_ms: float | None = None,
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


class MockLLMService:
    """Mock LLM service that returns scripted responses."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.call_count = 0

    async def stream(
        self, task: LLMTask, *, options: LLMOptions | None = None
    ) -> AsyncIterator[MockLLMChunk]:
        if self.call_count >= len(self.responses):
            response_data = {"content": "Default response", "tool_calls": []}
        else:
            response_data = self.responses[self.call_count]
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

        if response_data.get("error"):
            raise Exception(response_data["error"])

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


class TransientThenSuccessLLMService:
    """LLM service that fails transiently once, then succeeds."""

    def __init__(self, error_message: str = "Temporarily unavailable"):
        self.error_message = error_message
        self.call_count = 0

    async def stream(
        self, task: LLMTask, *, options: LLMOptions | None = None
    ) -> AsyncIterator[MockLLMChunk]:
        self.call_count += 1
        if self.call_count == 1:
            raise Exception(self.error_message)
        yield MockLLMChunk(
            content="Recovered after transient error",
            model="mock-model",
            provider="mock-provider",
            total_tokens=50,
            prompt_tokens=25,
            completion_tokens=25,
            finish_reason="stop",
        )


# ============================================================================
# Mock tool gateway
# ============================================================================


class MockToolGateway:
    """Mock tool gateway that completes all tool calls."""

    def __init__(self):
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
            result = ToolExecutionResult(
                status=ToolExecutionStatus.COMPLETED,
                output=f"Mock output for {request.tool_id}",
                backend=BackendType.HOST,
                timing={"elapsed_seconds": 0.1},
            )
            if self._event_callback:
                await self._event_callback(
                    ToolEvent(
                        type=ToolEventType.TOOL_CALL_START,
                        call_id=request.call_id,
                        session_id=request.session_id,
                        frame_id=request.frame_id,
                        trace_id=str(uuid.uuid4()),
                        data={
                            "tool_id": request.tool_id,
                            "arguments": request.raw_arguments,
                            "backend": "classic",
                        },
                    )
                )
                await self._event_callback(
                    ToolEvent(
                        type=ToolEventType.TOOL_CALL_COMPLETE,
                        call_id=request.call_id,
                        session_id=request.session_id,
                        frame_id=request.frame_id,
                        trace_id=str(uuid.uuid4()),
                        data={
                            "tool_id": request.tool_id,
                            "status": "completed",
                            "output": result.output,
                            "timing": result.timing,
                            "backend": "classic",
                        },
                    )
                )
            results.append(result)
        return results


class MockToolGatewayWithApproval:
    """Mock tool gateway that requires approval on first execution."""

    def __init__(self, require_approval: bool = True):
        self._event_callback = None
        self.executed_calls: list[ToolCallRequest] = []
        self.require_approval = require_approval

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
            if self.require_approval and not request.confirmation_token:
                result = ToolExecutionResult(
                    status=ToolExecutionStatus.PENDING_APPROVAL,
                    backend=BackendType.HOST,
                    decision=ToolDecision(
                        outcome=ToolDecisionOutcome.ASK,
                        reason="Tool requires approval",
                        policy_rule="test_approval",
                        confirmation_token=f"token_{request.call_id}",
                    ),
                )
                if self._event_callback:
                    await self._event_callback(
                        ToolEvent(
                            type=ToolEventType.TOOL_CALL_START,
                            call_id=request.call_id,
                            session_id=request.session_id,
                            frame_id=request.frame_id,
                            trace_id=str(uuid.uuid4()),
                            data={
                                "tool_id": request.tool_id,
                                "arguments": request.raw_arguments,
                                "backend": "classic",
                            },
                        )
                    )
            else:
                result = ToolExecutionResult(
                    status=ToolExecutionStatus.COMPLETED,
                    output=f"Approved output for {request.tool_id}",
                    backend=BackendType.HOST,
                    timing={"elapsed_seconds": 0.1},
                )
                if self._event_callback:
                    await self._event_callback(
                        ToolEvent(
                            type=ToolEventType.TOOL_CALL_START,
                            call_id=request.call_id,
                            session_id=request.session_id,
                            frame_id=request.frame_id,
                            trace_id=str(uuid.uuid4()),
                            data={
                                "tool_id": request.tool_id,
                                "arguments": request.raw_arguments,
                                "backend": "classic",
                            },
                        )
                    )
                    await self._event_callback(
                        ToolEvent(
                            type=ToolEventType.TOOL_CALL_COMPLETE,
                            call_id=request.call_id,
                            session_id=request.session_id,
                            frame_id=request.frame_id,
                            trace_id=str(uuid.uuid4()),
                            data={
                                "tool_id": request.tool_id,
                                "status": "completed",
                                "output": result.output,
                                "timing": result.timing,
                                "backend": "classic",
                            },
                        )
                    )
            results.append(result)
        return results


# ============================================================================
# Mock context manager that captures continuations
# ============================================================================


class CapturingContextManager:
    """Mock ContextManager returning a fixed rendered prompt and capturing
    the FrameContinuation passed to each resolve_context call."""

    def __init__(self, rendered_prompt: RenderedPrompt | None = None):
        self._rendered = rendered_prompt or RenderedPrompt(
            system_messages=["You are a helpful assistant."],
            user_messages=["Test user input"],
            tool_schemas=[],
            history=[],
            metadata={"model": "mock-model", "temperature": 0.7},
        )
        self.cleared_turns: list[str] = []
        self.resolve_call_count = 0
        self.captured_continuations: list[FrameContinuation] = []

    async def resolve_context(
        self,
        session: Any,
        snapshot: Any,
        continuation: FrameContinuation,
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
        self.cleared_turns.append(turn_id)


# ============================================================================
# Mock event pipeline
# ============================================================================


class MockEventPipeline:
    """Mock event pipeline that captures emitted events."""

    def __init__(self):
        self.events: list[RuntimeEvent] = []

    async def emit(self, event: RuntimeEvent) -> None:
        self.events.append(event)

    async def close(self) -> None:
        pass

    @property
    def subscribers(self) -> list:
        return []


# ============================================================================
# Scripted engine for continuation-directive tests
# ============================================================================


class CapturingEngine(IExecutionEngine):
    """Wraps an IExecutionEngine to capture the last frame passed to execute_frame."""

    def __init__(self, engine: IExecutionEngine):
        self._engine = engine
        self.last_frame: RuntimeFrame | None = None

    async def execute_frame(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        resolved: Any,
        rendered: Any,
    ) -> RuntimeFrame:
        self.last_frame = frame
        return await self._engine.execute_frame(frame, turn, resolved, rendered)


class ScriptedEngine(IExecutionEngine):
    """Engine returning pre-built frames in sequence."""

    def __init__(self, frames: list[RuntimeFrame]):
        self._frames = list(frames)
        self.call_count = 0
        self.received_frames: list[RuntimeFrame] = []

    async def execute_frame(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        resolved: Any,
        rendered: Any,
    ) -> RuntimeFrame:
        self.call_count += 1
        self.received_frames.append(frame)
        if self.call_count <= len(self._frames):
            return self._frames[self.call_count - 1]
        return RuntimeFrame(
            id=frame.id,
            session_id=frame.session_id,
            turn_id=frame.turn_id,
            trace_id=frame.trace_id,
            iteration=frame.iteration,
            status=ExecutionStatus.COMPLETED,
        )


def scripted_frame(
    status: ExecutionStatus,
    finish_reason: str | None = None,
    final_answer: str | None = None,
    error_message: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    iteration: int = 1,
) -> RuntimeFrame:
    """Build a RuntimeFrame with an optional result for outcome derivation."""
    frame = RuntimeFrame(
        id=f"frame_{uuid.uuid4()}",
        session_id="session_1",
        turn_id="turn_1",
        trace_id="trace_1",
        iteration=iteration,
        status=status,
    )
    if tool_calls:
        llm_action = frame.add_action(ActionKind.LLM_MESSAGE, name="llm_generation")
        frame.complete_action(
            llm_action.id,
            final_answer or "",
            metadata={
                "tool_calls": tool_calls,
                "model": "mock-model",
                "provider": "mock-provider",
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50,
                "reasoning_tokens": 0,
                "latency_ms": 10.0,
            },
        )
    if (
        final_answer is not None
        or error_message is not None
        or finish_reason is not None
    ):
        frame.complete(
            final_answer=final_answer,
            error_message=error_message,
            finish_reason=finish_reason,
        )
    return frame


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def mock_event_pipeline() -> MockEventPipeline:
    return MockEventPipeline()


@pytest.fixture
def mock_context_manager() -> CapturingContextManager:
    return CapturingContextManager()


@pytest.fixture
def session() -> SimpleNamespace:
    return SimpleNamespace(
        id="session_123",
        history=[],
        history_sequence=0,
        context_profile=None,
    )


@pytest.fixture
def runtime_limits() -> RuntimeLimits:
    return RuntimeLimits(
        token_budget=TokenBudget(
            config=TokenBudgetConfig(input_max_tokens=10_000, output_max_tokens=10_000)
        ),
        loop_guard=LoopGuard(config=LoopDetectionConfig(enabled=False)),
    )


# ============================================================================
# Helper utilities
# ============================================================================


def event_types(events: list[RuntimeEvent]) -> list[str]:
    return [e.type for e in events]


def find_event(events: list[RuntimeEvent], event_type) -> Any | None:
    return next((e for e in events if isinstance(e, event_type)), None)


def make_snapshot(turn_id: str = "turn_1") -> ContextSnapshot:
    return ContextSnapshot(
        id=f"ctx_{turn_id}",
        turn_id=turn_id,
        tool_set_ref=ToolSetRef(),
        template_set_ref=TemplateSetRef(),
        repo_snapshot_ref=None,
        memory_snapshot_ref=None,
        message_boundary=None,
        content_hash="sha256:abc",
        created_at_ms=int(datetime.now(UTC).timestamp() * 1000),
    )


def make_turn(
    turn_id: str = "turn_1",
    session_id: str = "session_1",
    trace_id: str = "trace_1",
    limits: RuntimeLimits | None = None,
    max_frames_per_turn: int | None = None,
) -> RuntimeTurn:
    snapshot = make_snapshot(turn_id)
    limits_config = RuntimeLimitsConfig()
    if max_frames_per_turn is not None:
        limits_config.continuation_policy = {"max_frames_per_turn": max_frames_per_turn}
    return RuntimeTurn(
        turn_id=turn_id,
        session_id=session_id,
        trace_id=trace_id,
        turn_index=0,
        user_input="Test input",
        context_snapshot=snapshot,
        context_snapshot_id=snapshot.id,
        context_config=RuntimeContextConfig(),
        limits_config=limits_config,
        limits=limits or RuntimeLimits(),
    )


async def make_frame_agent(
    engine: IExecutionEngine,
    context_manager: CapturingContextManager,
    event_pipeline: MockEventPipeline,
) -> FrameAgent:
    return FrameAgent(
        engine=engine,
        context_manager=context_manager,
        event_pipeline=event_pipeline,
    )


async def make_frame_agent_with_engine(
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


# ============================================================================
# TestFrameAgentRunTurnBasic - run_turn basic scenarios
# ============================================================================


class TestFrameAgentRunTurnBasic:
    async def test_simple_llm_only_completion(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        llm_service = MockLLMService(
            responses=[{"content": "Hello, world!", "tool_calls": []}]
        )
        tool_gateway = MockToolGateway()
        frame_agent = await make_frame_agent_with_engine(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn(limits=runtime_limits)

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.COMPLETED
        assert turn.final_answer == "Hello, world!"
        assert len(turn.frame_ids) == 1
        assert len(tool_gateway.executed_calls) == 0

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "completed"
        assert turn_complete.final_answer == "Hello, world!"
        assert turn_complete.total_frames == 1

        turn_start = find_event(mock_event_pipeline.events, TurnStartEvent)
        assert turn_start is not None
        assert turn_start.turn_id == turn.turn_id

    async def test_multi_frame_continue_to_final_answer(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        llm_service = MockLLMService(
            responses=[
                {
                    "content": "Let me check that for you.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "test_tool",
                            "arguments": {"param": "value"},
                        }
                    ],
                    "total_tokens": 100,
                },
                {
                    "content": "Here is the answer based on the tool result.",
                    "tool_calls": [],
                    "total_tokens": 50,
                },
            ]
        )
        tool_gateway = MockToolGateway()
        frame_agent = await make_frame_agent_with_engine(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn(limits=runtime_limits)

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.COMPLETED
        assert turn.final_answer == "Here is the answer based on the tool result."
        assert len(turn.frame_ids) == 2
        assert len(tool_gateway.executed_calls) == 1
        assert tool_gateway.executed_calls[0].tool_id == "test_tool"

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "completed"
        assert turn_complete.total_frames == 2

        decision_events = [
            e
            for e in mock_event_pipeline.events
            if isinstance(e, ContinuationDecisionEvent)
        ]
        assert len(decision_events) == 2
        assert decision_events[0].decision == "continue"
        assert decision_events[1].decision == "stop"

    async def test_event_ordering(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        llm_service = MockLLMService(
            responses=[{"content": "Simple answer.", "tool_calls": []}]
        )
        tool_gateway = MockToolGateway()
        frame_agent = await make_frame_agent_with_engine(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn(limits=runtime_limits)

        await frame_agent.run_turn(turn, session)

        types = event_types(mock_event_pipeline.events)
        assert types[0] == "turn_start"
        assert "frame_start" in types
        assert "llm_start" in types
        assert "llm_complete" in types
        assert "frame_complete" in types
        assert "continuation_decision" in types
        assert types[-1] == "turn_complete"
        assert types.index("turn_start") < types.index("frame_start")
        assert types.index("turn_complete") > types.index("frame_complete")

    async def test_turn_complete_event_metrics(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        llm_service = MockLLMService(
            responses=[
                {
                    "content": "Frame one.",
                    "tool_calls": [],
                    "total_tokens": 200,
                    "prompt_tokens": 100,
                    "completion_tokens": 100,
                }
            ]
        )
        tool_gateway = MockToolGateway()
        frame_agent = await make_frame_agent_with_engine(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn(limits=runtime_limits)

        await frame_agent.run_turn(turn, session)

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "completed"
        assert turn_complete.total_frames == 1
        assert turn_complete.total_input_tokens == 100
        assert turn_complete.total_output_tokens == 100
        assert turn_complete.finish_reason == "final_answer"


# ============================================================================
# TestFrameAgentContinuation - continuation decisions and directives
# ============================================================================


class TestFrameAgentContinuation:
    async def test_retry_generation_transient_error(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        llm_service = TransientThenSuccessLLMService(
            error_message="Temporarily unavailable"
        )
        tool_gateway = MockToolGateway()
        frame_agent = await make_frame_agent_with_engine(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn(limits=runtime_limits)

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.COMPLETED
        assert turn.final_answer == "Recovered after transient error"
        assert llm_service.call_count == 2

        decision_events = [
            e
            for e in mock_event_pipeline.events
            if isinstance(e, ContinuationDecisionEvent)
        ]
        retry_decisions = [
            e for e in decision_events if e.decision == "retry_generation"
        ]
        assert len(retry_decisions) == 1
        stop_decisions = [e for e in decision_events if e.decision == "stop"]
        assert len(stop_decisions) == 1

        # Continuation captured for the retry frame carries retry params
        assert len(mock_context_manager.captured_continuations) >= 2
        retry_continuation = mock_context_manager.captured_continuations[1]
        assert retry_continuation.generation_retry_count == 1

    async def test_frame_cap_limit(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        llm_service = MockLLMService(
            responses=[
                {
                    "content": "I'll use a tool",
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "name": "test_tool",
                            "arguments": {},
                        }
                    ],
                    "total_tokens": 50,
                }
                for i in range(20)
            ]
        )
        tool_gateway = MockToolGateway()
        frame_agent = await make_frame_agent_with_engine(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn(limits=runtime_limits, max_frames_per_turn=3)

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.FAILED
        assert len(turn.frame_ids) == 3

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "failed"
        assert turn_complete.finish_reason == "frame_limit_reached"

    async def test_continuation_tool_observations(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        llm_service = MockLLMService(
            responses=[
                {
                    "content": "Let me check that for you.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "test_tool",
                            "arguments": {"param": "value"},
                        }
                    ],
                    "total_tokens": 100,
                },
                {
                    "content": "Here is the answer.",
                    "tool_calls": [],
                    "total_tokens": 50,
                },
            ]
        )
        tool_gateway = MockToolGateway()
        frame_agent = await make_frame_agent_with_engine(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn(limits=runtime_limits)

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.COMPLETED
        # Two resolve_context calls: one per frame
        assert len(mock_context_manager.captured_continuations) == 2
        # First continuation is empty (initial), second carries tool observations
        first_cont = mock_context_manager.captured_continuations[0]
        assert first_cont.tool_observations == []
        second_cont = mock_context_manager.captured_continuations[1]
        assert len(second_cont.tool_observations) == 1
        obs = second_cont.tool_observations[0]
        assert obs.tool_name == "test_tool"
        assert obs.call_id == "call_1"
        assert obs.success is True

    async def test_continuation_compaction_directive(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        # Frame 1: FAILED with finish_reason length -> CONTEXT_OVERFLOW ->
        # COMPACT_AND_CONTINUE. Frame 2: COMPLETED -> FINAL_RESPONSE -> STOP.
        frame1 = scripted_frame(
            ExecutionStatus.FAILED,
            finish_reason="length",
            final_answer="partial content",
        )
        frame2 = scripted_frame(
            ExecutionStatus.COMPLETED,
            finish_reason="stop",
            final_answer="Final answer after compaction.",
        )
        engine = ScriptedEngine([frame1, frame2])
        frame_agent = await make_frame_agent(
            engine, mock_context_manager, mock_event_pipeline
        )
        turn = make_turn(limits=runtime_limits)

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.COMPLETED
        assert turn.final_answer == "Final answer after compaction."
        assert len(turn.frame_ids) == 2

        # Continuation for frame 2 carries the compaction instruction
        assert len(mock_context_manager.captured_continuations) == 2
        compact_cont = mock_context_manager.captured_continuations[1]
        assert compact_cont.compaction_instruction is not None
        assert compact_cont.compaction_count == 1

        decision_events = [
            e
            for e in mock_event_pipeline.events
            if isinstance(e, ContinuationDecisionEvent)
        ]
        compact_decisions = [
            e for e in decision_events if e.decision == "compact_and_continue"
        ]
        assert len(compact_decisions) == 1

    async def test_continuation_retry_directive(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        llm_service = TransientThenSuccessLLMService(
            error_message="Temporarily unavailable"
        )
        tool_gateway = MockToolGateway()
        frame_agent = await make_frame_agent_with_engine(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn(limits=runtime_limits)

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.COMPLETED
        # Continuation for the retry frame carries retry params
        assert len(mock_context_manager.captured_continuations) == 2
        retry_cont = mock_context_manager.captured_continuations[1]
        assert retry_cont.generation_retry_count == 1
        assert retry_cont.delay_before_retry is not None

        decision_events = [
            e
            for e in mock_event_pipeline.events
            if isinstance(e, ContinuationDecisionEvent)
        ]
        retry_decisions = [
            e for e in decision_events if e.decision == "retry_generation"
        ]
        assert len(retry_decisions) == 1


# ============================================================================
# TestFrameAgentLimits - turn-level limit enforcement
# ============================================================================


class TestFrameAgentLimits:
    async def test_deadline_exceeded(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        llm_service = MockLLMService(
            responses=[{"content": "Frame one response.", "tool_calls": []}]
        )
        tool_gateway = MockToolGateway()

        runtime_limits.deadline_tracker.reset(timeout_seconds=0)

        frame_agent = await make_frame_agent_with_engine(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn(limits=runtime_limits)

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.FAILED

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "failed"
        assert turn_complete.finish_reason == "deadline_exceeded"

    async def test_token_budget_depleted(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
    ):
        llm_service = MockLLMService(
            responses=[
                {
                    "content": "Frame one.",
                    "tool_calls": [],
                    "total_tokens": 200,
                    "prompt_tokens": 100,
                    "completion_tokens": 100,
                },
                {
                    "content": "Frame two.",
                    "tool_calls": [],
                    "total_tokens": 200,
                    "prompt_tokens": 100,
                    "completion_tokens": 100,
                },
            ]
        )
        tool_gateway = MockToolGateway()

        limits = RuntimeLimits(
            token_budget=TokenBudget(
                config=TokenBudgetConfig(input_max_tokens=50, output_max_tokens=50)
            ),
            loop_guard=LoopGuard(config=LoopDetectionConfig(enabled=False)),
        )

        frame_agent = await make_frame_agent_with_engine(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn(limits=limits)

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.FAILED

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "failed"
        assert turn_complete.finish_reason == "token_budget_depleted"

    async def test_loop_detection_frame_failed(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
    ):
        llm_service = MockLLMService(
            responses=[
                {
                    "content": "I'll repeat the same tool.",
                    "tool_calls": [
                        {
                            "id": "call_repeat",
                            "name": "same_tool",
                            "arguments": {"param": "same"},
                        }
                    ],
                    "total_tokens": 50,
                },
            ]
        )
        tool_gateway = MockToolGateway()

        limits = RuntimeLimits(
            token_budget=TokenBudget(
                config=TokenBudgetConfig(
                    input_max_tokens=100_000, output_max_tokens=100_000
                )
            ),
            loop_guard=LoopGuard(
                config=LoopDetectionConfig(
                    enabled=True,
                    max_repetitions=2,
                    window_size=5,
                    check_consecutive=True,
                    check_window_repetitions=True,
                )
            ),
        )
        limits.loop_guard.record_tool_call("pre_1", "same_tool", {"param": "same"})

        frame_agent = await make_frame_agent_with_engine(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn(limits=limits)

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.FAILED


# ============================================================================
# TestFrameAgentResumeTurn - resume_turn scenarios
# ============================================================================


class TestFrameAgentResumeTurn:
    async def test_approval_pause_and_resume(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        llm_service = MockLLMService(
            responses=[
                {
                    "content": "I need to use a dangerous tool.",
                    "tool_calls": [
                        {
                            "id": "call_danger",
                            "name": "dangerous_tool",
                            "arguments": {"action": "delete"},
                        }
                    ],
                    "total_tokens": 100,
                },
                {
                    "content": "Tool completed after approval.",
                    "tool_calls": [],
                    "total_tokens": 50,
                },
            ]
        )
        tool_gateway = MockToolGatewayWithApproval(require_approval=True)
        engine = RuntimeEngine(
            llm_service=llm_service,
            tool_gateway=tool_gateway,
            event_pipeline=mock_event_pipeline,
        )
        capturing_engine = CapturingEngine(engine)
        frame_agent = FrameAgent(
            engine=capturing_engine,
            context_manager=mock_context_manager,
            event_pipeline=mock_event_pipeline,
        )
        turn = make_turn(limits=runtime_limits)

        # First run: LLM produces tool call, gateway requires approval -> pause
        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.RUNNING
        assert turn.pause_reason == "approval_pending"
        assert len(turn.pending_approvals) == 1
        assert len(tool_gateway.executed_calls) == 1

        approval_events = [
            e
            for e in mock_event_pipeline.events
            if isinstance(e, ApprovalRequiredEvent)
        ]
        assert len(approval_events) == 1
        approval_token = approval_events[0].approval_requests[0].approval_token
        tool_call_id = approval_events[0].approval_requests[0].tool_call_id

        paused_frame = capturing_engine.last_frame
        assert paused_frame is not None
        assert paused_frame.status == ExecutionStatus.RUNNING_TOOLS

        # Apply approval response
        turn.apply_approval_response(
            ApprovalResponseForm(
                approval_token=approval_token,
                tool_call_id=tool_call_id,
                response="yes_once",
            )
        )
        assert (
            turn.pending_approvals[tool_call_id].response
            == ToolApprovalResponse.YES_ONCE
        )
        assert turn.pending_approvals[tool_call_id].status == ApprovalStatus.GRANTED

        # Resume the paused frame
        turn.interrupt_event.clear()
        await frame_agent.resume_turn(turn, session, frame_to_resume=paused_frame)

        assert turn.status == TurnStatus.COMPLETED
        assert len(tool_gateway.executed_calls) == 2

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "completed"


# ============================================================================
# TestFrameAgentErrorHandling - transient error resumability
# ============================================================================


class TestFrameAgentErrorHandling:
    async def test_transient_error_resumability(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        llm_service = TransientThenSuccessLLMService(
            error_message="Temporarily unavailable"
        )
        tool_gateway = MockToolGateway()
        frame_agent = await make_frame_agent_with_engine(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn(limits=runtime_limits)

        # run_turn should swallow the transient exception internally via the
        # continuation policy and complete the turn on retry.
        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.COMPLETED
        assert turn.final_answer == "Recovered after transient error"
        assert llm_service.call_count == 2

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "completed"

    async def test_non_transient_error_turn_failed(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        class AlwaysFailingLLMService:
            async def stream(self, task, *, options=None):
                raise Exception("Authentication failed: invalid API key")
                yield  # pragma: no cover - make async generator

        llm_service = AlwaysFailingLLMService()
        tool_gateway = MockToolGateway()
        frame_agent = await make_frame_agent_with_engine(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn(limits=runtime_limits)

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.FAILED
        # Non-transient errors carry no final answer; the error is in frame.result
        assert turn.final_answer is None

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "failed"
        assert turn_complete.finish_reason == "provider_non_transient_error"
