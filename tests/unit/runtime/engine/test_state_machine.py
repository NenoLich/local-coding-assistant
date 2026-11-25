"""Unit tests for RuntimeEngine state machine transitions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.agent.llm import LLMToolCall
from local_coding_assistant.config.schemas import LoopDetectionConfig
from local_coding_assistant.runtime.context.profile import (
    ContextSnapshot,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.engine.engine import RuntimeEngine
from local_coding_assistant.runtime.engine.frame import (
    ActionKind,
    ExecutionStatus,
    RuntimeFrame,
)
from local_coding_assistant.runtime.engine.limits import LoopGuard
from local_coding_assistant.runtime.session.turn import RuntimeTurn
from local_coding_assistant.runtime.types import RenderedPrompt
from local_coding_assistant.tools_platform.contracts import (
    ApprovalResponse,
    ToolDecision,
    ToolDecisionOutcome,
    ToolExecutionResult,
    ToolExecutionStatus,
)


@pytest.fixture
def mock_llm_service():
    """Mock LLM service whose stream() returns an async generator directly."""
    service = MagicMock()
    service.stream = MagicMock()
    return service


@pytest.fixture
def mock_tool_gateway():
    """Mock tool gateway."""
    gateway = MagicMock()
    gateway.execute_many = AsyncMock()
    return gateway


@pytest.fixture
def mock_event_pipeline():
    """Mock event pipeline."""
    pipeline = MagicMock()
    pipeline.emit = AsyncMock()
    return pipeline


@pytest.fixture
def sample_frame():
    """Sample runtime frame."""
    return RuntimeFrame(
        id="frame_123",
        session_id="session_123",
        turn_id="turn_123",
        trace_id="trace_123",
        iteration=1,
        context_snapshot_id="ctx_123",
        status=ExecutionStatus.RUNNING_LLM,
    )


@pytest.fixture
def sample_resolved():
    """Sample resolved context."""
    return SimpleNamespace(
        metadata={"model": "test_model", "temperature": 0.7, "streaming": True},
        streaming=True,
    )


@pytest.fixture
def sample_rendered():
    """Sample rendered prompt."""
    return RenderedPrompt(
        messages=[{"role": "user", "content": "test input"}],
        system_messages=["You are a helpful assistant"],
        user_messages=["test input"],
        history=[],
        tool_schemas=[],
    )


@pytest.fixture
def sample_runtime_turn():
    """Sample runtime turn."""
    turn = RuntimeTurn(
        turn_id="turn_123",
        session_id="session_123",
        trace_id="trace_123",
        turn_index=1,
        user_input="test input",
        context_snapshot=ContextSnapshot(
            id="ctx_123",
            turn_id="turn_123",
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
            repo_snapshot_ref=None,
            memory_snapshot_ref=None,
            message_boundary=None,
            content_hash="hash123",
            created_at_ms=1000,
        ),
        context_snapshot_id="ctx_123",
        context_config=RuntimeContextConfig(),
        limits_config=RuntimeLimitsConfig(),
    )

    return turn


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str = "",
    reasoning: str = "",
    tool_calls: list[LLMToolCall] | None = None,
    model: str = "mock-model",
    provider: str = "mock-provider",
    finish_reason: str = "stop",
    total_tokens: int | None = 100,
    prompt_tokens: int | None = 50,
    completion_tokens: int | None = 50,
    reasoning_tokens: int | None = 0,
    latency_ms: float | None = 10.0,
) -> SimpleNamespace:
    """Build a mock LLM stream chunk readable by RuntimeEngine._update_chunk_metadata."""
    return SimpleNamespace(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls or [],
        model=model,
        provider=provider,
        finish_reason=finish_reason,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        latency_ms=latency_ms,
        usage=None,
        metadata={},
    )


async def _stream(chunks: list[SimpleNamespace]):
    """Async generator yielding mock chunks."""
    for chunk in chunks:
        yield chunk


def _make_engine(
    llm_service: MagicMock | None = None,
    tool_gateway: MagicMock | None = None,
    event_pipeline: MagicMock | None = None,
) -> RuntimeEngine:
    return RuntimeEngine(
        llm_service=llm_service or MagicMock(),
        tool_gateway=tool_gateway or MagicMock(),
        event_pipeline=event_pipeline or MagicMock(),
    )


def _make_turn(
    turn_id: str = "turn_123",
    session_id: str = "session_123",
    trace_id: str = "trace_123",
    limits_config: RuntimeLimitsConfig | None = None,
    interrupted: bool = False,
) -> RuntimeTurn:
    turn = RuntimeTurn(
        turn_id=turn_id,
        session_id=session_id,
        trace_id=trace_id,
        turn_index=1,
        user_input="test input",
        context_snapshot=ContextSnapshot(
            id="ctx_123",
            turn_id=turn_id,
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
            repo_snapshot_ref=None,
            memory_snapshot_ref=None,
            message_boundary=None,
            content_hash="hash123",
            created_at_ms=1000,
        ),
        context_snapshot_id="ctx_123",
        context_config=RuntimeContextConfig(),
        limits_config=limits_config or RuntimeLimitsConfig(),
    )
    if interrupted:
        turn.interrupt_event.set()
    return turn


def _completed_tool_result(
    call_id: str, tool_id: str = "read_file"
) -> ToolExecutionResult:
    return ToolExecutionResult(
        status=ToolExecutionStatus.COMPLETED.value,
        output="ok",
        call_id=call_id,
        tool_id=tool_id,
    )


def _failed_tool_result(
    call_id: str, tool_id: str = "read_file"
) -> ToolExecutionResult:
    return ToolExecutionResult(
        status=ToolExecutionStatus.FAILED.value,
        output=None,
        error=None,
        call_id=call_id,
        tool_id=tool_id,
    )


def _approval_tool_result(call_id: str, ct: str = "tok_1") -> ToolExecutionResult:
    return ToolExecutionResult(
        status=ToolExecutionStatus.PENDING_APPROVAL.value,
        output=None,
        decision=ToolDecision(
            outcome=ToolDecisionOutcome.ASK,
            reason="requires approval",
            confirmation_token=ct,
        ),
        call_id=call_id,
    )


def _frame_with_tool_calls(
    status: ExecutionStatus = ExecutionStatus.RUNNING_TOOLS,
    call_ids: list[str] | None = None,
) -> RuntimeFrame:
    call_ids = call_ids or ["call_1", "call_2"]
    frame = RuntimeFrame(
        id="frame_1",
        session_id="session_1",
        turn_id="turn_1",
        trace_id="trace_1",
        iteration=1,
        context_snapshot_id="ctx_1",
        status=status,
    )
    llm_action = frame.add_action(ActionKind.LLM_MESSAGE, name="llm_generation")
    llm_action.tool_calls = [
        LLMToolCall(id=cid, name="read_file", arguments={"path": f"/fake/{cid}.txt"})
        for cid in call_ids
    ]
    frame.complete_action(
        llm_action.id,
        "I will call tools",
        metadata={
            "tool_calls": [tc.model_dump() for tc in llm_action.tool_calls],
            "reasoning": None,
            "model": "mock-model",
            "provider": "mock-provider",
        },
    )
    return frame


class TestRuntimeEngineStateMachine:
    """Frame execution state machine transitions.

    RUNNING_LLM -> RUNNING_TOOLS -> COMPLETED/FAILED/PARTIAL/BLOCKED,
    resume from any state, and terminal state absorption.
    """

    # -- LLM phase ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_llm_completes_without_tools_transitions_to_completed(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [_make_chunk(content="Hello, world!", finish_reason="stop")]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert result.result is not None
        assert result.result.final_answer == "Hello, world!"
        assert result.model_response_raw == "Hello, world!"
        assert mock_event_pipeline.emit.await_count > 0

    @pytest.mark.asyncio
    async def test_llm_with_tool_calls_transitions_tools_then_completed(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        tool_calls = [
            LLMToolCall(
                id="call_1", name="read_file", arguments={"path": "/fake/a.txt"}
            )
        ]
        mock_llm_service.stream.return_value = _stream(
            [
                _make_chunk(
                    content="reading", tool_calls=tool_calls, finish_reason="stop"
                )
            ]
        )
        mock_tool_gateway.execute_many.return_value = [
            _completed_tool_result("call_1"),
        ]
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert result.model_response_raw == "reading"
        mock_tool_gateway.execute_many.assert_awaited_once()
        llm_action = next(a for a in result.actions if a.kind == ActionKind.LLM_MESSAGE)
        assert len(llm_action.tool_calls) == 1
        assert llm_action.tool_calls[0].id == "call_1"

    @pytest.mark.asyncio
    async def test_llm_length_finish_reason_transitions_to_partial(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [_make_chunk(content="partial", finish_reason="length")]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.PARTIAL
        assert result.result is not None
        assert result.result.final_answer == "partial"
        assert "truncated" in (result.result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_llm_content_filter_transitions_to_blocked(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [_make_chunk(content="", finish_reason="content_filter")]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.BLOCKED
        assert result.result is not None
        assert "blocked" in (result.result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_llm_transient_error_keeps_frame_running_llm(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        from local_coding_assistant.providers.exceptions import ProviderRateLimitError

        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.side_effect = ProviderRateLimitError("rate limited")
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.RUNNING_LLM
        assert result.result is not None
        assert "rate limited" in (result.result.error_message or "")

    @pytest.mark.asyncio
    async def test_llm_permanent_error_transitions_to_failed(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        from local_coding_assistant.providers.exceptions import ProviderAuthError

        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.side_effect = ProviderAuthError("bad key")
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.FAILED
        assert result.result is not None
        assert "bad key" in (result.result.error_message or "")

    @pytest.mark.asyncio
    async def test_token_budget_exceeded_transitions_to_failed(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        from local_coding_assistant.config.schemas import TokenBudgetConfig
        from local_coding_assistant.runtime.engine.limits import TokenBudget

        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        turn.limits.token_budget = TokenBudget(
            config=TokenBudgetConfig(output_max_tokens=10)
        )
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [_make_chunk(content="too long", completion_tokens=50, total_tokens=100)]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.FAILED
        assert result.result is not None
        assert "budget" in (result.result.error_message or "").lower()

    # -- Tools phase --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_running_tools_all_completed_transitions_to_completed(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = _frame_with_tool_calls(
            ExecutionStatus.RUNNING_TOOLS, ["call_1", "call_2"]
        )
        mock_tool_gateway.execute_many.return_value = [
            _completed_tool_result("call_1"),
            _completed_tool_result("call_2"),
        ]
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.COMPLETED
        mock_tool_gateway.execute_many.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_running_tools_some_fail_non_retryable_transitions_to_partial(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = _frame_with_tool_calls(
            ExecutionStatus.RUNNING_TOOLS, ["call_1", "call_2"]
        )
        mock_tool_gateway.execute_many.return_value = [
            _completed_tool_result("call_1"),
            _failed_tool_result("call_2"),
        ]
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.PARTIAL
        assert result.result is not None
        assert result.result.error_message is not None

    @pytest.mark.asyncio
    async def test_loop_detected_transitions_to_failed_without_executing_tools(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        # Two identical tool calls (same tool + args, different call ids) => consecutive loop.
        frame = _frame_with_tool_calls(
            ExecutionStatus.RUNNING_TOOLS, ["call_1", "call_2"]
        )
        # Force both calls to have identical signature by sharing name+args.
        llm_action = next(a for a in frame.actions if a.kind == ActionKind.LLM_MESSAGE)
        llm_action.tool_calls = [
            LLMToolCall(
                id="call_1", name="read_file", arguments={"path": "/fake/x.txt"}
            ),
            LLMToolCall(
                id="call_2", name="read_file", arguments={"path": "/fake/x.txt"}
            ),
        ]
        mock_tool_gateway.execute_many.return_value = []
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.FAILED
        mock_tool_gateway.execute_many.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_loop_state_persists_across_frames_within_turn(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        """LoopGuard accumulates across frames within a turn; reset only at turn boundary.

        Two identical tool signatures split across two frames on the same turn must
        trigger loop detection on the second frame, proving the guard is not reset
        between frames.
        """
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        turn.limits.loop_guard = LoopGuard(
            config=LoopDetectionConfig(
                max_repetitions=2,
                window_size=10,
                check_consecutive=False,
                check_window_repetitions=True,
            )
        )

        def _frame(call_id: str) -> RuntimeFrame:
            frame = _frame_with_tool_calls(ExecutionStatus.RUNNING_TOOLS, [call_id])
            llm_action = next(
                a for a in frame.actions if a.kind == ActionKind.LLM_MESSAGE
            )
            llm_action.tool_calls = [
                LLMToolCall(id=call_id, name="read_file", arguments={"path": "/x.txt"})
            ]
            return frame

        # First frame: single call, below the repetition threshold -> completes.
        mock_tool_gateway.execute_many.return_value = [
            _completed_tool_result("call_1"),
        ]
        result1 = await engine.execute_frame(
            _frame("call_1"), turn, sample_resolved, sample_rendered
        )
        assert result1.status == ExecutionStatus.COMPLETED
        mock_tool_gateway.execute_many.reset_mock()

        # Second frame on the same turn with the identical signature: the call
        # recorded in frame 1 is still present, so count reaches 2 -> loop detected.
        mock_tool_gateway.execute_many.return_value = []
        result2 = await engine.execute_frame(
            _frame("call_2"), turn, sample_resolved, sample_rendered
        )
        assert result2.status == ExecutionStatus.FAILED
        mock_tool_gateway.execute_many.assert_not_awaited()
        assert len(turn.limits.loop_guard._tool_call_history) == 2

    @pytest.mark.asyncio
    async def test_approval_pending_keeps_frame_running_tools(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = _frame_with_tool_calls(ExecutionStatus.RUNNING_TOOLS, ["call_1"])
        mock_tool_gateway.execute_many.return_value = [_approval_tool_result("call_1")]
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.RUNNING_TOOLS
        assert "call_1" in turn.pending_approvals
        assert mock_event_pipeline.emit.await_count > 0

    @pytest.mark.asyncio
    async def test_resume_with_approval_token_transitions_to_completed(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = _frame_with_tool_calls(ExecutionStatus.RUNNING_TOOLS, ["call_1"])
        mock_tool_gateway.execute_many.return_value = [_approval_tool_result("call_1")]
        first = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )
        assert first.status == ExecutionStatus.RUNNING_TOOLS

        # Grant the pending approval and re-run (resume from RUNNING_TOOLS).
        approval = turn.pending_approvals["call_1"]
        approval.response = ApprovalResponse.YES_ONCE
        mock_tool_gateway.execute_many.return_value = [_completed_tool_result("call_1")]
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.COMPLETED

    # -- Resume / terminal absorption ---------------------------------------

    @pytest.mark.asyncio
    async def test_resume_from_running_llm_reenters_llm_phase(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [_make_chunk(content="resumed", finish_reason="stop")]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert result.result is not None
        assert result.result.final_answer == "resumed"

    @pytest.mark.asyncio
    async def test_resume_from_running_tools_executes_tools(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = _frame_with_tool_calls(ExecutionStatus.RUNNING_TOOLS, ["call_1"])
        mock_tool_gateway.execute_many.return_value = [_completed_tool_result("call_1")]
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.COMPLETED
        mock_tool_gateway.execute_many.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_terminal_states_are_absorbing_no_op(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        for status in (
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.PARTIAL,
            ExecutionStatus.BLOCKED,
        ):
            frame = RuntimeFrame(
                id="frame_1",
                session_id="session_1",
                turn_id="turn_1",
                trace_id="trace_1",
                iteration=1,
                context_snapshot_id="ctx_1",
                status=status,
            )
            before_status = frame.status
            result = await engine.execute_frame(
                frame, turn, sample_resolved, sample_rendered
            )
            assert result.status == before_status, f"{status} was not absorbed"
            mock_tool_gateway.execute_many.assert_not_awaited()
            mock_llm_service.stream.assert_not_called()
            mock_llm_service.stream.reset_mock()
            mock_tool_gateway.execute_many.reset_mock()

    # -- Interrupt / deadline -----------------------------------------------

    @pytest.mark.asyncio
    async def test_interrupt_during_llm_stays_running_llm_preserves_partial_content(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )

        async def _interrupting_stream():
            yield _make_chunk(content="partial", finish_reason="stop")
            turn.interrupt_event.set()
            yield _make_chunk(content=" more", finish_reason="stop")

        mock_llm_service.stream.return_value = _interrupting_stream()
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.RUNNING_LLM
        # Partial content accumulated before the interrupt is preserved on the frame.
        assert result.model_response_raw == "partial more"
        assert mock_llm_service.stream.call_count == 1

    @pytest.mark.asyncio
    async def test_interrupt_pauses_frame_before_entering_a_phase(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        """Interrupt set before execute_frame pauses at loop start; phase is not entered."""
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn(interrupted=True)

        llm_frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        result = await engine.execute_frame(
            llm_frame, turn, sample_resolved, sample_rendered
        )
        assert result.status == ExecutionStatus.RUNNING_LLM
        mock_llm_service.stream.assert_not_called()

        tools_frame = _frame_with_tool_calls(ExecutionStatus.RUNNING_TOOLS, ["call_1"])
        result = await engine.execute_frame(
            tools_frame, turn, sample_resolved, sample_rendered
        )
        assert result.status == ExecutionStatus.RUNNING_TOOLS
        mock_tool_gateway.execute_many.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deadline_exceeded_at_loop_start_transitions_to_failed(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        tracker = turn.limits.deadline_tracker
        now = datetime.now(UTC)
        tracker._start_time = now - timedelta(seconds=2)
        tracker._deadline = now - timedelta(seconds=1)
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.FAILED
        assert result.result is not None
        assert "deadline" in (result.result.error_message or "").lower()
