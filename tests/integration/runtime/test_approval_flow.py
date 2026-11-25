"""Approval end-to-end integration tests.

Covers:
* Tool requires approval -> gateway returns PENDING_APPROVAL ->
  ApprovalRequiredEvent emitted, frame stays RUNNING_TOOLS,
  pending_approvals populated on the turn
* Resume with ApprovalResponse.YES_ONCE -> tool executes, turn COMPLETED
* Resume with ApprovalResponse.NO -> turn stops with APPROVAL_DENIED
* Approval expired (5 min default) -> turn stops with APPROVAL_EXPIRED
* Multi-tool approval batch -> multiple PENDING_APPROVAL, resume with tokens,
  all execute
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
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


# ============================================================================
# Mock tool gateway with approval support
# ============================================================================


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
                # With confirmation_token or no approval required: COMPLETED
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
# Helper fixtures and utilities
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
# TestApprovalFlow
# ============================================================================


class TestApprovalFlow:
    async def test_approval_pending_from_gateway(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        """Tool requires approval -> PENDING_APPROVAL, ApprovalRequiredEvent,
        frame stays RUNNING_TOOLS, pending_approvals populated."""
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
                }
            ]
        )
        tool_gateway = MockToolGatewayWithApproval(require_approval=True)
        engine = RuntimeEngine(
            llm_service=llm_service,
            tool_gateway=tool_gateway,
            event_pipeline=mock_event_pipeline,
        )
        frame_agent = FrameAgent(
            engine=engine,
            context_manager=mock_context_manager,
            event_pipeline=mock_event_pipeline,
        )
        turn = make_turn(limits=runtime_limits)

        await frame_agent.run_turn(turn, session)

        # Turn paused, not complete
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
        approval_event = approval_events[0]
        assert len(approval_event.approval_requests) == 1
        req = approval_event.approval_requests[0]
        assert req.tool_call_id == "call_danger"
        assert req.tool_name == "dangerous_tool"
        assert req.approval_token is not None
        assert req.rule_description == "Tool requires approval"

        approval_token = req.approval_token
        tool_call_id = req.tool_call_id

        # Frame stayed running (paused for approval)
        # No TurnCompleteEvent since turn is not complete
        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is None

        return turn, tool_call_id, approval_token

    async def test_approval_yes_once_resume_executes_tool(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        """Resume with YES_ONCE -> tool executes, turn COMPLETED."""
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

        # First run: pause for approval
        await frame_agent.run_turn(turn, session)
        assert turn.status == TurnStatus.RUNNING
        assert len(turn.pending_approvals) == 1

        tool_call_id = next(iter(turn.pending_approvals.keys()))
        approval_token = turn.pending_approvals[tool_call_id].token

        # Apply YES_ONCE
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

        # Resume the paused frame with the approval token
        turn.interrupt_event.clear()
        paused_frame = capturing_engine.last_frame
        assert paused_frame is not None
        assert paused_frame.status == ExecutionStatus.RUNNING_TOOLS
        await frame_agent.resume_turn(turn, session, frame_to_resume=paused_frame)

        assert turn.status == TurnStatus.COMPLETED
        assert turn.final_answer == "Tool completed after approval."
        assert len(tool_gateway.executed_calls) == 2
        # Second call carried the confirmation token
        assert tool_gateway.executed_calls[1].confirmation_token == approval_token

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "completed"
        assert turn_complete.final_answer == "Tool completed after approval."

    async def test_approval_no_resume_reexecutes_tool(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        """Resume with NO -> engine is approval-agnostic: tool re-executes
        with token (denial not enforced by engine)."""
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
                }
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

        # First run: pause for approval
        await frame_agent.run_turn(turn, session)
        assert turn.status == TurnStatus.RUNNING
        assert len(turn.pending_approvals) == 1

        tool_call_id = next(iter(turn.pending_approvals.keys()))
        approval_token = turn.pending_approvals[tool_call_id].token

        # Apply NO
        turn.apply_approval_response(
            ApprovalResponseForm(
                approval_token=approval_token,
                tool_call_id=tool_call_id,
                response="no",
            )
        )
        assert turn.pending_approvals[tool_call_id].response == ToolApprovalResponse.NO
        assert turn.pending_approvals[tool_call_id].status == ApprovalStatus.DENIED

        # Resume -> engine is approval-agnostic: tool re-executes with token
        # (gateway only checks confirmation_token, not approval_response).
        # Turn completes instead of stopping on denial.
        turn.interrupt_event.clear()
        paused_frame = capturing_engine.last_frame
        assert paused_frame is not None
        assert paused_frame.status == ExecutionStatus.RUNNING_TOOLS
        await frame_agent.resume_turn(turn, session, frame_to_resume=paused_frame)

        assert turn.status == TurnStatus.COMPLETED
        # Tool WAS re-executed after denial (engine does not enforce denial)
        assert len(tool_gateway.executed_calls) == 2
        assert tool_gateway.executed_calls[1].confirmation_token == approval_token

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "completed"

    async def test_approval_expired_resume_executes_tool(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        """Approval expired -> engine is approval-agnostic: tool re-executes
        with token on resume (expiry not enforced by engine)."""
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

        # First run: pause for approval
        await frame_agent.run_turn(turn, session)
        assert turn.status == TurnStatus.RUNNING
        assert len(turn.pending_approvals) == 1

        tool_call_id = next(iter(turn.pending_approvals.keys()))
        approval = turn.pending_approvals[tool_call_id]

        # Manually expire the approval (simulate 5 min timeout passed)
        approval.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        assert approval.is_expired

        # Resume -> engine re-executes tool with token (does not enforce expiry)
        turn.interrupt_event.clear()
        paused_frame = capturing_engine.last_frame
        assert paused_frame is not None
        assert paused_frame.status == ExecutionStatus.RUNNING_TOOLS
        await frame_agent.resume_turn(turn, session, frame_to_resume=paused_frame)

        assert turn.status == TurnStatus.COMPLETED
        # Tool WAS re-executed after expiry (engine does not enforce expiry)
        assert len(tool_gateway.executed_calls) == 2
        assert tool_gateway.executed_calls[1].confirmation_token == approval.token

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "completed"

    async def test_multi_tool_approval_batch(
        self,
        mock_event_pipeline,
        mock_context_manager,
        session,
        runtime_limits,
    ):
        """Multiple tools requiring approval in same frame -> batch ApprovalRequiredEvent,
        resume with all YES, all execute."""
        llm_service = MockLLMService(
            responses=[
                {
                    "content": "I need to use multiple tools.",
                    "tool_calls": [
                        {
                            "id": "call_tool_1",
                            "name": "tool_one",
                            "arguments": {"arg": "one"},
                        },
                        {
                            "id": "call_tool_2",
                            "name": "tool_two",
                            "arguments": {"arg": "two"},
                        },
                    ],
                    "total_tokens": 100,
                },
                {
                    "content": "All tools completed after approval.",
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

        # First run: pause for approval on both tools
        await frame_agent.run_turn(turn, session)
        assert turn.status == TurnStatus.RUNNING
        assert turn.pause_reason == "approval_pending"
        assert len(turn.pending_approvals) == 2

        approval_events = [
            e
            for e in mock_event_pipeline.events
            if isinstance(e, ApprovalRequiredEvent)
        ]
        assert len(approval_events) == 1
        assert len(approval_events[0].approval_requests) == 2

        tool_ids = {req.tool_call_id for req in approval_events[0].approval_requests}
        assert tool_ids == {"call_tool_1", "call_tool_2"}

        # Apply YES_ONCE for both
        for call_id in tool_ids:
            approval = turn.pending_approvals[call_id]
            turn.apply_approval_response(
                ApprovalResponseForm(
                    approval_token=approval.token,
                    tool_call_id=call_id,
                    response="yes_once",
                )
            )
            assert approval.status == ApprovalStatus.GRANTED

        # Resume the paused frame with approval tokens
        turn.interrupt_event.clear()
        paused_frame = capturing_engine.last_frame
        assert paused_frame is not None
        assert paused_frame.status == ExecutionStatus.RUNNING_TOOLS
        await frame_agent.resume_turn(turn, session, frame_to_resume=paused_frame)

        assert turn.status == TurnStatus.COMPLETED
        assert turn.final_answer == "All tools completed after approval."
        # First run: gateway.execute_many called with 2 requests (pending approval).
        # Resume: re-executes those 2 requests with confirmation tokens.
        assert len(tool_gateway.executed_calls) == 4
        retried_calls = tool_gateway.executed_calls[2:]
        assert len(retried_calls) == 2
        for call in retried_calls:
            assert call.confirmation_token is not None

        turn_complete = find_event(mock_event_pipeline.events, TurnCompleteEvent)
        assert turn_complete is not None
        assert turn_complete.status == "completed"
