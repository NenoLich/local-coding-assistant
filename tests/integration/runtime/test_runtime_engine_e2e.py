"""End-to-end tests for RuntimeEngine frame execution and FrameAgent turn loop.

Tests the complete frame execution state machine:
- RUNNING_LLM -> RUNNING_TOOLS -> COMPLETED/FAILED/PARTIAL/BLOCKED
- Event emissions at each phase
- Limits enforcement (token budget, deadline, loop detection)
- Approval pause/resume flow
- FrameAgent multi-frame turn loop (CONTINUE, RETRY_GENERATION)
- Turn finalization and event ordering
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from local_coding_assistant.agent.llm import LLMOptions, LLMTask, LLMToolCall
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
    ApprovalRequest,
    ApprovalStatus,
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
    FrameCompleteEvent,
    FrameStartEvent,
    RuntimeEvent,
    TurnCompleteEvent,
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
    ToolErrorCategory,
    ToolErrorCode,
    ToolEvent,
    ToolEventType,
    ToolExecutionError,
    ToolExecutionResult,
    ToolExecutionStatus,
)

# ============================================================================
# Mock LLM Service
# ============================================================================


class MockLLMChunk:
    """Mock LLM chunk that works with RuntimeEngine's _update_chunk_metadata."""

    def __init__(
        self,
        content: str = "",
        reasoning: str = "",
        tool_calls: list[LLMToolCall] | None = None,
        model: str = "mock-model",
        provider: str = "mock-provider",
        is_final: bool = False,
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
        self.is_final = is_final
        self.total_tokens = total_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.latency_ms = latency_ms
        self.finish_reason = finish_reason


class MockLLMService:
    """Mock LLM service for testing."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = responses
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
        tool_calls = response_data.get("tool_calls", [])
        reasoning = response_data.get("reasoning", "")
        finish_reason = response_data.get("finish_reason", "stop")
        total_tokens = response_data.get("total_tokens", 100 if tool_calls else 50)
        prompt_tokens = response_data.get("prompt_tokens", 50 if tool_calls else 25)
        completion_tokens = response_data.get(
            "completion_tokens", 50 if tool_calls else 25
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
                is_final=False,
            )

        if tool_calls:
            tc_objects = [
                LLMToolCall(
                    id=tc.get("id", str(uuid.uuid4())),
                    name=tc.get("name", "unknown"),
                    arguments=tc.get("arguments", {}),
                )
                for tc in tool_calls
            ]
            yield MockLLMChunk(
                content=content,
                tool_calls=tc_objects,
                model="mock-model",
                provider="mock-provider",
                is_final=True,
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
                is_final=True,
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
            )


class FailingLLMService:
    """Mock LLM service that raises an error on every call."""

    def __init__(self, error_message: str = "LLM service error"):
        self.error_message = error_message
        self.call_count = 0

    async def stream(
        self, task: LLMTask, *, options: LLMOptions | None = None
    ) -> AsyncIterator[MockLLMChunk]:
        self.call_count += 1
        raise Exception(self.error_message)
        yield  # Make this an async generator


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
            is_final=True,
            total_tokens=50,
            prompt_tokens=25,
            completion_tokens=25,
            finish_reason="stop",
        )


# ============================================================================
# Mock Tool Gateway
# ============================================================================


class MockToolGateway:
    """Mock tool gateway for testing."""

    def __init__(self):
        self._event_callback = None
        self.executed_calls = []

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
    """Mock tool gateway that requires approval."""

    def __init__(self, require_approval: bool = True):
        self._event_callback = None
        self.executed_calls = []
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

                results.append(result)
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


class MockToolGatewayWithFailure:
    """Mock tool gateway that fails tool execution."""

    def __init__(self, fail_tool: str | None = None):
        self._event_callback = None
        self.executed_calls = []
        self.fail_tool = fail_tool

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
            if self.fail_tool and request.tool_id == self.fail_tool:
                result = ToolExecutionResult(
                    status=ToolExecutionStatus.FAILED,
                    output=None,
                    backend=BackendType.HOST,
                    timing={"elapsed_seconds": 0.1},
                    error=ToolExecutionError(
                        code=ToolErrorCode.BACKEND_FAILURE,
                        category=ToolErrorCategory.INFRASTRUCTURE,
                        message=f"Tool {request.tool_id} failed",
                    ),
                )
            else:
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

                if result.status == ToolExecutionStatus.COMPLETED:
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
                else:
                    await self._event_callback(
                        ToolEvent(
                            type=ToolEventType.TOOL_CALL_ERROR,
                            call_id=request.call_id,
                            session_id=request.session_id,
                            frame_id=request.frame_id,
                            trace_id=str(uuid.uuid4()),
                            data={
                                "tool_id": request.tool_id,
                                "status": "failed",
                                "error": result.error.model_dump()
                                if result.error
                                else None,
                                "timing": result.timing,
                                "backend": "classic",
                            },
                        )
                    )

            results.append(result)

        return results


# ============================================================================
# Mock Context Manager (for FrameAgent integration tests)
# ============================================================================


class MockContextManager:
    """Mock ContextManager that returns a fixed rendered prompt for every frame."""

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
        prompt_ctx = PromptContext(
            session_id=getattr(session, "id", "session_123"),
            execution_mode=ExecutionMode.CLASSIC_TOOLS,
            tool_call_mode="classic",
        )
        return prompt_ctx, self._rendered

    async def clear_turn_cache(self, turn_id: str) -> None:
        self.cleared_turns.append(turn_id)


class CapturingEngine(IExecutionEngine):
    """Wraps IExecutionEngine to capture the last frame passed to execute_frame."""

    def __init__(self, engine: RuntimeEngine):
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


# ============================================================================
# Mock Event Pipeline
# ============================================================================


class MockEventPipeline:
    """Mock event pipeline that captures events."""

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
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_service():
    responses = [
        {"content": "Hello, world!", "tool_calls": []},
        {
            "content": "I'll use a tool",
            "tool_calls": [
                {"id": "call_1", "name": "test_tool", "arguments": {"param": "value"}}
            ],
        },
    ]
    return MockLLMService(responses.copy())


@pytest.fixture
def mock_tool_gateway():
    return MockToolGateway()


@pytest.fixture
def mock_tool_gateway_with_approval():
    return MockToolGatewayWithApproval(require_approval=True)


@pytest.fixture
def mock_tool_gateway_fails():
    return MockToolGatewayWithFailure(fail_tool="failing_tool")


@pytest.fixture
def mock_event_pipeline():
    return MockEventPipeline()


@pytest.fixture
def mock_context_manager():
    return MockContextManager()


@pytest.fixture
def runtime_limits():
    from local_coding_assistant.config.schemas import TokenBudgetConfig

    return RuntimeLimits(
        token_budget=TokenBudget(
            config=TokenBudgetConfig(input_max_tokens=10_000, output_max_tokens=10_000)
        ),
        loop_guard=LoopGuard(config=LoopDetectionConfig(enabled=False)),
    )


@pytest.fixture
def turn_context():
    from local_coding_assistant.runtime.context.profile import (
        ContextSnapshot,
        RuntimeContextConfig,
        TemplateSetRef,
        ToolSetRef,
    )

    snapshot = ContextSnapshot(
        id="ctx_123",
        turn_id="turn_123",
        tool_set_ref=ToolSetRef(),
        template_set_ref=TemplateSetRef(),
        repo_snapshot_ref=None,
        memory_snapshot_ref=None,
        message_boundary=None,
        content_hash="sha256:abc123",
        created_at_ms=int(datetime.now(UTC).timestamp() * 1000),
    )

    return RuntimeTurn(
        turn_id="turn_123",
        session_id="session_123",
        trace_id="trace_123",
        turn_index=0,
        user_input="Test user input",
        context_snapshot=snapshot,
        context_snapshot_id="ctx_123",
        context_config=RuntimeContextConfig(),
        limits_config=RuntimeLimitsConfig(),
    )


@pytest.fixture
def session():
    return SimpleNamespace(
        id="session_123",
        history=[],
        history_sequence=0,
        context_profile=None,
    )


@pytest.fixture
def sample_rendered():
    return RenderedPrompt(
        system_messages=["You are a helpful assistant."],
        user_messages=["Test user input"],
        tool_schemas=[],
        history=[],
        metadata={"model": "gpt-4", "temperature": 0.7},
    )


# ============================================================================
# Helper utilities
# ============================================================================


def event_types(events: list[RuntimeEvent]) -> list[str]:
    return [e.type for e in events]


def find_event(events: list[RuntimeEvent], event_type) -> Any | None:
    return next((e for e in events if isinstance(e, event_type)), None)


def make_turn(
    turn_id: str = "turn_1",
    session_id: str = "session_1",
    trace_id: str = "trace_1",
    limits: RuntimeLimits | None = None,
) -> RuntimeTurn:
    from local_coding_assistant.runtime.context.profile import (
        ContextSnapshot,
        RuntimeContextConfig,
        TemplateSetRef,
        ToolSetRef,
    )

    snapshot = ContextSnapshot(
        id="ctx_1",
        turn_id=turn_id,
        tool_set_ref=ToolSetRef(),
        template_set_ref=TemplateSetRef(),
        repo_snapshot_ref=None,
        memory_snapshot_ref=None,
        message_boundary=None,
        content_hash="sha256:abc",
        created_at_ms=int(datetime.now(UTC).timestamp() * 1000),
    )

    return RuntimeTurn(
        turn_id=turn_id,
        session_id=session_id,
        trace_id=trace_id,
        turn_index=0,
        user_input="Test input",
        context_snapshot=snapshot,
        context_snapshot_id="ctx_1",
        context_config=RuntimeContextConfig(),
        limits_config=RuntimeLimitsConfig(),
        limits=limits or RuntimeLimits(),
    )


async def make_frame_agent(
    llm_service,
    tool_gateway,
    event_pipeline,
    context_manager,
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
# E2E Tests - RuntimeEngine (single frame)
# ============================================================================


class TestRuntimeEngineE2E:
    """End-to-end tests for RuntimeEngine single-frame execution."""

    async def test_simple_frame_execution_llm_only(
        self,
        mock_llm_service,
        mock_tool_gateway,
        mock_event_pipeline,
        runtime_limits,
        turn_context,
        sample_rendered,
    ):
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=mock_tool_gateway,
            event_pipeline=mock_event_pipeline,
        )

        frame = RuntimeFrame(
            id="frame_123",
            session_id="session_123",
            turn_id="turn_123",
            trace_id="trace_123",
            iteration=1,
            status=ExecutionStatus.RUNNING_LLM,
        )

        result_frame = await engine.execute_frame(
            frame, turn_context, None, sample_rendered
        )
        await asyncio.sleep(0)

        assert result_frame.status == ExecutionStatus.COMPLETED
        assert result_frame.finished_at is not None
        assert result_frame.result is not None
        assert result_frame.result.final_answer == "Hello, world!"

        event_types_list = event_types(mock_event_pipeline.events)
        assert "frame_start" in event_types_list
        assert "llm_start" in event_types_list
        assert "llm_chunk" in event_types_list
        assert "llm_complete" in event_types_list
        assert "frame_complete" in event_types_list

        frame_start = find_event(mock_event_pipeline.events, FrameStartEvent)
        assert frame_start is not None
        assert frame_start.frame_id == "frame_123"
        assert frame_start.turn_id == "turn_123"
        assert frame_start.iteration == 1

        frame_complete = find_event(mock_event_pipeline.events, FrameCompleteEvent)
        assert frame_complete is not None
        assert frame_complete.status == "completed"
        assert frame_complete.llm_message == "Hello, world!"

    async def test_frame_execution_with_tool_calls(
        self,
        mock_llm_service,
        mock_tool_gateway,
        mock_event_pipeline,
        runtime_limits,
        turn_context,
        sample_rendered,
    ):
        mock_llm_service.responses = [
            {
                "content": "I'll use a tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "test_tool",
                        "arguments": {"param": "value"},
                    }
                ],
                "total_tokens": 100,
            },
        ]
        mock_llm_service.call_count = 0

        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=mock_tool_gateway,
            event_pipeline=mock_event_pipeline,
        )

        frame = RuntimeFrame(
            id="frame_tool",
            session_id="session_tool",
            turn_id="turn_tool",
            trace_id="trace_tool",
            iteration=1,
            status=ExecutionStatus.RUNNING_LLM,
        )

        result_frame = await engine.execute_frame(
            frame, turn_context, None, sample_rendered
        )
        await asyncio.sleep(0)

        assert result_frame.status == ExecutionStatus.COMPLETED
        assert result_frame.finished_at is not None
        assert result_frame.result is not None
        assert "I'll use a tool" in result_frame.result.final_answer

        assert len(mock_tool_gateway.executed_calls) == 1
        executed_call = mock_tool_gateway.executed_calls[0]
        assert executed_call.tool_id == "test_tool"
        assert executed_call.raw_arguments == {"param": "value"}

        event_types_list = event_types(mock_event_pipeline.events)
        assert "frame_start" in event_types_list
        assert "llm_start" in event_types_list
        assert "llm_chunk" in event_types_list
        assert "llm_complete" in event_types_list
        assert "tool_start" in event_types_list
        assert "tool_result" in event_types_list
        assert "frame_complete" in event_types_list

        frame_complete = find_event(mock_event_pipeline.events, FrameCompleteEvent)
        assert frame_complete is not None
        assert frame_complete.status == "completed"
        assert frame_complete.llm_message is not None
        assert "I'll use a tool" in frame_complete.llm_message

    async def test_frame_execution_approval_pause_resume(
        self,
        mock_llm_service,
        mock_tool_gateway_with_approval,
        mock_event_pipeline,
        runtime_limits,
        turn_context,
        sample_rendered,
    ):
        mock_llm_service.responses = [
            {
                "content": "I'll use a tool that needs approval",
                "tool_calls": [
                    {
                        "id": "call_approval",
                        "name": "dangerous_tool",
                        "arguments": {"action": "delete"},
                    }
                ],
                "total_tokens": 100,
            },
            {
                "content": "Tool completed after approval",
                "tool_calls": [],
                "total_tokens": 50,
            },
        ]
        mock_llm_service.call_count = 0

        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=mock_tool_gateway_with_approval,
            event_pipeline=mock_event_pipeline,
        )

        frame = RuntimeFrame(
            id="frame_approval",
            session_id="session_approval",
            turn_id="turn_approval",
            trace_id="trace_approval",
            iteration=1,
            status=ExecutionStatus.RUNNING_LLM,
        )

        result_frame = await engine.execute_frame(
            frame, turn_context, None, sample_rendered
        )
        await asyncio.sleep(0)

        assert result_frame.status == ExecutionStatus.RUNNING_TOOLS
        assert result_frame.finished_at is None

        approval_events = [
            e
            for e in mock_event_pipeline.events
            if isinstance(e, ApprovalRequiredEvent)
        ]
        assert len(approval_events) == 1
        approval_event = approval_events[0]
        assert len(approval_event.approval_requests) == 1
        assert approval_event.approval_requests[0].tool_name == "dangerous_tool"
        assert approval_event.approval_requests[0].arguments == {"action": "delete"}
        assert (
            approval_event.approval_requests[0].approval_token == "token_call_approval"
        )  # noqa: S105

        approval_token = approval_event.approval_requests[0].approval_token

        turn_context.pending_approvals["call_approval"] = ApprovalRequest(
            token=approval_token,
            tool_call_id="call_approval",
            tool_name="dangerous_tool",
            arguments={"action": "delete"},
            risk_class="high",
            session_id="session_approval",
            frame_id="frame_approval",
            turn_id="turn_approval",
            status=ApprovalStatus.GRANTED,
            timeout_seconds=300,
            response=ToolApprovalResponse("yes_once"),
            responded_at=datetime.now(UTC),
        )

        result_frame = await engine.execute_frame(
            result_frame, turn_context, None, sample_rendered
        )
        await asyncio.sleep(0)

        event_types_list = event_types(mock_event_pipeline.events)
        assert "approval_required" in event_types_list
        assert "tool_start" in event_types_list
        assert "tool_result" in event_types_list
        assert "frame_complete" in event_types_list

    async def test_frame_execution_token_limit_enforcement(
        self,
        mock_llm_service,
        mock_tool_gateway,
        mock_event_pipeline,
        turn_context,
        sample_rendered,
    ):
        from local_coding_assistant.config.schemas import TokenBudgetConfig

        token_budget = TokenBudget(
            config=TokenBudgetConfig(input_max_tokens=50, output_max_tokens=50)
        )
        loop_limits = RuntimeLimits(
            token_budget=token_budget,
            loop_guard=LoopGuard(config=LoopDetectionConfig(enabled=False)),
        )
        turn_context.limits = loop_limits

        mock_llm_service.responses = [
            {
                "content": "This is a very long response that uses many tokens",
                "tool_calls": [],
                "total_tokens": 100,
                "prompt_tokens": 20,
                "completion_tokens": 80,
            },
        ]
        mock_llm_service.call_count = 0

        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=mock_tool_gateway,
            event_pipeline=mock_event_pipeline,
        )

        frame = RuntimeFrame(
            id="frame_token",
            session_id="session_token",
            turn_id="turn_token",
            trace_id="trace_token",
            iteration=1,
            status=ExecutionStatus.RUNNING_LLM,
        )

        result_frame = await engine.execute_frame(
            frame, turn_context, None, sample_rendered
        )
        await asyncio.sleep(0)

        assert result_frame.status == ExecutionStatus.FAILED
        assert result_frame.finished_at is not None
        assert result_frame.result is not None
        assert "token budget" in result_frame.result.error_message.lower()

        frame_complete = find_event(mock_event_pipeline.events, FrameCompleteEvent)
        assert frame_complete is not None
        assert frame_complete.status == "failed"
        assert "token budget" in frame_complete.error_message.lower()

    async def test_frame_execution_loop_detection(
        self,
        mock_llm_service,
        mock_tool_gateway,
        mock_event_pipeline,
        turn_context,
        sample_rendered,
    ):
        from local_coding_assistant.config.schemas import TokenBudgetConfig

        loop_limits = RuntimeLimits(
            token_budget=TokenBudget(
                config=TokenBudgetConfig(
                    input_max_tokens=10_000, output_max_tokens=10_000
                )
            ),
            loop_guard=LoopGuard(
                config=LoopDetectionConfig(
                    enabled=True,
                    max_repetitions=2,
                    window_size=3,
                )
            ),
        )
        turn_context.limits = loop_limits

        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=mock_tool_gateway,
            event_pipeline=mock_event_pipeline,
        )

        frame = RuntimeFrame(
            id="frame_loop",
            session_id="session_loop",
            turn_id="turn_loop",
            trace_id="trace_loop",
            iteration=1,
            status=ExecutionStatus.RUNNING_LLM,
        )

        loop_limits.loop_guard.record_tool_call("123", "test_tool", {"param": "value"})
        loop_limits.loop_guard.record_tool_call("456", "test_tool", {"param": "value"})

        mock_llm_service.responses = [
            {
                "content": "I'll use a tool",
                "tool_calls": [
                    {
                        "id": "789",
                        "name": "test_tool",
                        "arguments": {"param": "value"},
                    }
                ],
                "total_tokens": 100,
            },
        ]
        mock_llm_service.call_count = 0

        result_frame = await engine.execute_frame(
            frame, turn_context, None, sample_rendered
        )

        assert result_frame.status == ExecutionStatus.FAILED
        assert result_frame.result.error_message is not None

    async def test_frame_execution_error_handling(
        self,
        mock_tool_gateway,
        mock_event_pipeline,
        runtime_limits,
        turn_context,
        sample_rendered,
    ):
        failing_llm = FailingLLMService("Authentication failed: invalid API key")

        engine = RuntimeEngine(
            llm_service=failing_llm,
            tool_gateway=mock_tool_gateway,
            event_pipeline=mock_event_pipeline,
        )

        frame = RuntimeFrame(
            id="frame_error",
            session_id="session_error",
            turn_id="turn_error",
            trace_id="trace_error",
            iteration=1,
            status=ExecutionStatus.RUNNING_LLM,
        )

        result_frame = await engine.execute_frame(
            frame, turn_context, None, sample_rendered
        )

        assert result_frame.status == ExecutionStatus.FAILED
        assert result_frame.finished_at is not None
        assert result_frame.result is not None
        assert "Authentication failed" in result_frame.result.error_message

        event_types_list = event_types(mock_event_pipeline.events)
        assert "frame_start" in event_types_list
        assert "llm_start" in event_types_list
        assert "frame_complete" in event_types_list

        frame_complete = find_event(mock_event_pipeline.events, FrameCompleteEvent)
        assert frame_complete is not None
        assert frame_complete.status == "failed"
        assert frame_complete.error_message is not None
        assert "Authentication failed" in frame_complete.error_message

    async def test_frame_execution_resume_from_intermediate_state(
        self,
        mock_llm_service,
        mock_tool_gateway,
        mock_event_pipeline,
        runtime_limits,
        turn_context,
    ):
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=mock_tool_gateway,
            event_pipeline=mock_event_pipeline,
        )

        frame = RuntimeFrame(
            id="frame_resume",
            session_id="session_resume",
            turn_id="turn_resume",
            trace_id="trace_resume",
            iteration=1,
            status=ExecutionStatus.RUNNING_TOOLS,
        )

        llm_action = frame.add_action(ActionKind.LLM_MESSAGE, name="llm_generation")
        frame.complete_action(
            llm_action.id,
            "I'll use a tool",
            metadata={
                "tool_calls": [
                    {
                        "id": "call_resume",
                        "name": "test_tool",
                        "arguments": {"param": "value"},
                    }
                ],
                "model": "mock-model",
                "total_tokens": 100,
            },
        )

        result_frame = await engine.execute_frame(
            frame, turn_context, None, sample_rendered
        )

        assert result_frame.status == ExecutionStatus.COMPLETED
        assert result_frame.finished_at is not None
        assert len(mock_tool_gateway.executed_calls) == 1

    async def test_event_sequence_ordering(
        self,
        mock_llm_service,
        mock_tool_gateway,
        mock_event_pipeline,
        turn_context,
        sample_rendered,
    ):
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=mock_tool_gateway,
            event_pipeline=mock_event_pipeline,
        )

        frame = RuntimeFrame(
            id="frame_order",
            session_id="session_order",
            turn_id="turn_order",
            trace_id="trace_order",
            iteration=1,
            status=ExecutionStatus.RUNNING_LLM,
        )

        await engine.execute_frame(frame, turn_context, None, sample_rendered)

        event_types_list = event_types(mock_event_pipeline.events)

        assert event_types_list[0] == "frame_start"
        assert event_types_list.index("llm_start") > event_types_list.index(
            "frame_start"
        )
        assert event_types_list.index("llm_complete") > event_types_list.index(
            "llm_start"
        )
        assert event_types_list[-1] == "frame_complete"

    async def test_frame_metrics_tracking(
        self,
        mock_llm_service,
        mock_tool_gateway,
        mock_event_pipeline,
        turn_context,
        sample_rendered,
    ):
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=mock_tool_gateway,
            event_pipeline=mock_event_pipeline,
        )

        frame = RuntimeFrame(
            id="frame_metrics",
            session_id="session_metrics",
            turn_id="turn_metrics",
            trace_id="trace_metrics",
            iteration=1,
            status=ExecutionStatus.RUNNING_LLM,
        )

        result_frame = await engine.execute_frame(
            frame, turn_context, None, sample_rendered
        )
        await asyncio.sleep(0)

        assert result_frame.duration_ms is not None
        assert result_frame.duration_ms >= 0

        llm_metrics = result_frame.get_llm_metrics()
        assert llm_metrics is not None
        assert llm_metrics.model == "mock-model"

        frame_complete = find_event(mock_event_pipeline.events, FrameCompleteEvent)
        assert frame_complete is not None
        assert frame_complete.duration_ms is None or frame_complete.duration_ms >= 0
        assert frame_complete.model_name == "mock-model"


# ============================================================================
# E2E Tests - FrameAgent (multi-frame turn loop)
# ============================================================================


class TestFrameAgentLoop:
    """FrameAgent integration tests covering multi-frame turn execution."""

    async def test_multi_frame_continue(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
    ):
        """Multi-frame turn: LLM tool calls -> CONTINUE -> final answer -> STOP."""
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

        frame_agent = await make_frame_agent(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn()

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

    async def test_retry_generation_transient_error(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
    ):
        """Transient LLM error -> RETRY_GENERATION -> success -> STOP."""
        llm_service = TransientThenSuccessLLMService(
            error_message="Temporarily unavailable"
        )
        tool_gateway = MockToolGateway()

        frame_agent = await make_frame_agent(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn()

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

    async def test_approval_pause_and_resume(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
    ):
        """Approval pause/resume end-to-end through FrameAgent."""
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
        turn = make_turn()

        # First run: LLM produces tool calls, gateway requires approval -> pause
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

    async def test_frame_agent_turn_frame_cap(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
    ):
        """Frame cap limits the number of frames per turn."""
        from local_coding_assistant.config.schemas import TokenBudgetConfig

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

        limits = RuntimeLimits(
            token_budget=TokenBudget(
                config=TokenBudgetConfig(
                    input_max_tokens=100_000, output_max_tokens=100_000
                )
            ),
            loop_guard=LoopGuard(config=LoopDetectionConfig(enabled=False)),
        )
        turn = make_turn(limits=limits)
        turn.limits_config.continuation_policy = {
            "max_frames_per_turn": 3,
        }

        frame_agent = await make_frame_agent(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.FAILED
        assert len(turn.frame_ids) == 3

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "failed"
        assert turn_complete.finish_reason == "frame_limit_reached"


# ============================================================================
# E2E Tests - FrameAgent limits across frames
# ============================================================================


class TestFrameAgentLimits:
    """Limit enforcement tests across multiple frames."""

    def _make_turn_with_limits(self, limits: RuntimeLimits) -> RuntimeTurn:
        from local_coding_assistant.runtime.context.profile import (
            ContextSnapshot,
            RuntimeContextConfig,
            TemplateSetRef,
            ToolSetRef,
        )

        snapshot = ContextSnapshot(
            id="ctx_limit",
            turn_id="turn_limit",
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
            repo_snapshot_ref=None,
            memory_snapshot_ref=None,
            message_boundary=None,
            content_hash="sha256:limit",
            created_at_ms=int(datetime.now(UTC).timestamp() * 1000),
        )

        return RuntimeTurn(
            turn_id="turn_limit",
            session_id="session_limit",
            trace_id="trace_limit",
            turn_index=0,
            user_input="Limit test input",
            context_snapshot=snapshot,
            context_snapshot_id="ctx_limit",
            context_config=RuntimeContextConfig(),
            limits_config=RuntimeLimitsConfig(),
            limits=limits,
        )

    async def test_deadline_exceeded(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
    ):
        """Turn deadline exceeded across frames -> DEADLINE_EXCEEDED -> STOP."""
        from local_coding_assistant.config.schemas import TokenBudgetConfig

        llm_service = MockLLMService(
            responses=[
                {
                    "content": "Frame one response.",
                    "tool_calls": [],
                    "total_tokens": 10,
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
            loop_guard=LoopGuard(config=LoopDetectionConfig(enabled=False)),
        )
        limits.deadline_tracker.reset(timeout_seconds=0)

        turn = self._make_turn_with_limits(limits)

        frame_agent = await make_frame_agent(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.FAILED
        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "failed"
        assert turn_complete.finish_reason == "deadline_exceeded"

    async def test_token_budget_across_frames(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
    ):
        """Token budget exceeded across frames -> TOKEN_BUDGET_DEPLETED -> STOP."""
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

        from local_coding_assistant.config.schemas import TokenBudgetConfig

        limits = RuntimeLimits(
            token_budget=TokenBudget(
                config=TokenBudgetConfig(input_max_tokens=50, output_max_tokens=50)
            ),
            loop_guard=LoopGuard(config=LoopDetectionConfig(enabled=False)),
        )

        turn = self._make_turn_with_limits(limits)

        frame_agent = await make_frame_agent(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )

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
        """LoopGuard detects repeated tool calls -> LOOP_DETECTED -> FAILED."""
        from local_coding_assistant.config.schemas import TokenBudgetConfig

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

        turn = self._make_turn_with_limits(limits)

        frame_agent = await make_frame_agent(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )

        await frame_agent.run_turn(turn, session)

        assert turn.status == TurnStatus.FAILED


# ============================================================================
# E2E Tests - Event pipeline integration through FrameAgent
# ============================================================================


class TestFrameAgentEvents:
    """Verify event ordering and types emitted through the FrameAgent loop."""

    async def test_turn_lifecycle_events(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
    ):
        llm_service = MockLLMService(
            responses=[
                {
                    "content": "Simple answer.",
                    "tool_calls": [],
                    "total_tokens": 30,
                },
            ]
        )
        tool_gateway = MockToolGateway()

        frame_agent = await make_frame_agent(
            llm_service, tool_gateway, mock_event_pipeline, mock_context_manager
        )
        turn = make_turn()

        await frame_agent.run_turn(turn, session)

        event_types_list = event_types(mock_event_pipeline.events)

        assert "turn_start" in event_types_list
        assert "frame_start" in event_types_list
        assert "llm_start" in event_types_list
        assert "llm_complete" in event_types_list
        assert "frame_complete" in event_types_list
        assert "continuation_decision" in event_types_list
        assert "turn_complete" in event_types_list

        assert event_types_list.index("turn_start") < event_types_list.index(
            "frame_start"
        )
        assert event_types_list[-1] == "turn_complete"
