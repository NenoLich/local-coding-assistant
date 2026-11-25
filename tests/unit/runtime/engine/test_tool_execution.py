"""Unit tests for RuntimeEngine tools phase execution.

Covers the plan's section 6.3 Tools Phase scenarios:

- All tools COMPLETED -> frame COMPLETED
- Some tools FAILED (non-retryable) -> frame PARTIAL
- Retryable failure + retries remaining -> frame RUNNING_TOOLS, re-enter
- Retryable failure + retries exhausted -> frame PARTIAL
- PENDING_APPROVAL without token -> frame RUNNING_TOOLS, ApprovalRequiredEvent
- PENDING_APPROVAL with token (resume) -> COMPLETED / PARTIAL
- Loop detection before execution -> frame FAILED
- Recorded terminal outcome skip on resume
- Synthetic events for prepare-phase failures
- No tool calls -> COMPLETED
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.agent.llm import LLMToolCall
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
from local_coding_assistant.runtime.events.contracts import (
    ApprovalRequiredEvent,
    ToolResultEvent,
)
from local_coding_assistant.runtime.session.turn import RuntimeTurn
from local_coding_assistant.runtime.types import RenderedPrompt
from local_coding_assistant.tools_platform.contracts import (
    ApprovalResponse,
    ToolDecision,
    ToolDecisionOutcome,
    ToolExecutionError,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from local_coding_assistant.tools_platform.errors import (
    ToolErrorCategory,
    ToolErrorCode,
)


@pytest.fixture
def mock_llm_service():
    """Mock LLM service (unused by tools phase but required by engine)."""
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
def sample_resolved():
    """Sample resolved context."""
    return MagicMock(
        metadata={"model": "test_model", "streaming": True}, streaming=True
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


def _make_turn(
    turn_id: str = "turn_123",
    session_id: str = "session_123",
    trace_id: str = "trace_123",
    tool_retry_max_retries: int | None = None,
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
        limits_config=RuntimeLimitsConfig(),
    )
    if tool_retry_max_retries is not None:
        turn.limits.tool_retry_guard.config.tool_retry_max_retries = (
            tool_retry_max_retries
        )
    if interrupted:
        turn.interrupt_event.set()
    return turn


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


def _completed_tool_result(
    call_id: str, tool_id: str = "read_file"
) -> ToolExecutionResult:
    return ToolExecutionResult(
        status=ToolExecutionStatus.COMPLETED.value,
        output=f"out:{tool_id}",
        call_id=call_id,
        tool_id=tool_id,
    )


def _failed_tool_result(
    call_id: str, tool_id: str = "read_file"
) -> ToolExecutionResult:
    return ToolExecutionResult(
        status=ToolExecutionStatus.FAILED.value,
        output=None,
        error=ToolExecutionError(
            code=ToolErrorCode.BACKEND_FAILURE,
            category=ToolErrorCategory.INFRASTRUCTURE,
            message="tool broke",
            retryable=False,
        ),
        call_id=call_id,
        tool_id=tool_id,
    )


def _retryable_tool_result(
    call_id: str, tool_id: str = "read_file"
) -> ToolExecutionResult:
    return ToolExecutionResult(
        status=ToolExecutionStatus.FAILED.value,
        output=None,
        error=ToolExecutionError(
            code=ToolErrorCode.BACKEND_FAILURE,
            category=ToolErrorCategory.INFRASTRUCTURE,
            message="backend flake",
            retryable=True,
        ),
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
        tool_id="read_file",
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


def _attach_completed_tool_action(
    frame: RuntimeFrame, tool_id: str, call_id: str, output: str
) -> None:
    action = frame.add_action(ActionKind.TOOL_CALL, name=tool_id)
    action.tool_execution_data.update(
        {
            "tool_id": tool_id,
            "call_id": call_id,
            "input": {"path": "/fake/x.txt"},
            "status": "completed",
            "output": output,
        }
    )
    action.finished_at = None


class TestToolsExecution:
    """Tools phase: execute_many, approval flow, retries, loop detection,
    recorded outcome skip, and synthetic events for prepare-phase failures."""

    # -- success / failure outcomes ------------------------------------------

    @pytest.mark.asyncio
    async def test_all_tools_completed_transitions_to_completed(
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
        requests = mock_tool_gateway.execute_many.await_args[0][0]
        assert [r.call_id for r in requests] == ["call_1", "call_2"]

    @pytest.mark.asyncio
    async def test_some_tools_fail_non_retryable_transitions_to_partial(
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
        assert "tool broke" in (result.result.error_message or "")

    @pytest.mark.asyncio
    async def test_no_tool_calls_transitions_to_completed(
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
            status=ExecutionStatus.RUNNING_TOOLS,
        )
        llm_action = frame.add_action(ActionKind.LLM_MESSAGE, name="llm_generation")
        llm_action.tool_calls = []
        frame.complete_action(
            llm_action.id,
            "done",
            metadata={"tool_calls": [], "model": "mock-model", "provider": "mock"},
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.COMPLETED
        mock_tool_gateway.execute_many.assert_not_awaited()

    # -- retryable failures --------------------------------------------------

    @pytest.mark.asyncio
    async def test_retryable_failure_retries_remaining_reenters_tools_phase(
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

        gateway_calls = {"n": 0}

        async def execute_many_side_effect(
            requests: list[Any], interrupt_event: Any = None
        ) -> list[ToolExecutionResult]:
            gateway_calls["n"] += 1
            if gateway_calls["n"] == 1:
                return [_retryable_tool_result("call_1")]
            return [_completed_tool_result("call_1")]

        mock_tool_gateway.execute_many.side_effect = execute_many_side_effect
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        # Retried then succeeded on re-entry
        assert result.status == ExecutionStatus.COMPLETED
        assert mock_tool_gateway.execute_many.await_count == 2

    @pytest.mark.asyncio
    async def test_retryable_failure_retries_exhausted_transitions_to_partial(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn(tool_retry_max_retries=1)
        frame = _frame_with_tool_calls(ExecutionStatus.RUNNING_TOOLS, ["call_1"])
        mock_tool_gateway.execute_many.return_value = [_retryable_tool_result("call_1")]
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.PARTIAL
        # Initial attempt + one retry
        assert mock_tool_gateway.execute_many.await_count == 2

    # -- approval flow -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_pending_approval_without_token_pauses_for_approval(
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
        approval = turn.pending_approvals["call_1"]
        assert approval.token == "tok_1"
        assert approval.status.value == "pending"
        # ApprovalRequiredEvent emitted
        assert any(
            isinstance(c.args[0], ApprovalRequiredEvent)
            for c in mock_event_pipeline.emit.await_args_list
        )

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
        # The resumed request carries the confirmation token and response.
        requests = mock_tool_gateway.execute_many.await_args[0][0]
        assert requests[0].confirmation_token == "tok_1"
        assert requests[0].approval_response == ApprovalResponse.YES_ONCE

    @pytest.mark.asyncio
    async def test_approval_denied_transitions_to_partial(
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

        # Deny the approval and re-run.
        approval = turn.pending_approvals["call_1"]
        approval.response = ApprovalResponse.NO
        mock_tool_gateway.execute_many.return_value = [
            ToolExecutionResult(
                status=ToolExecutionStatus.DENIED.value,
                call_id="call_1",
                tool_id="read_file",
            )
        ]
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.PARTIAL

    # -- loop detection ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_loop_detected_before_execution_transitions_to_failed(
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
        # Identical tool + arguments => consecutive loop on the second call.
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

    # -- recorded outcome skip on resume -------------------------------------

    @pytest.mark.asyncio
    async def test_recorded_terminal_outcome_skipped_on_resume(
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
        # call_1 completed before the crash; only call_2 should be dispatched.
        _attach_completed_tool_action(frame, "read_file", "call_1", "out:read_file")
        mock_tool_gateway.execute_many.return_value = [_completed_tool_result("call_2")]
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert mock_tool_gateway.execute_many.await_count == 1
        requests = mock_tool_gateway.execute_many.await_args[0][0]
        assert [r.call_id for r in requests] == ["call_2"]
        # Both actions terminal; no duplicate for call_1.
        tool_actions = [a for a in result.actions if a.kind == ActionKind.TOOL_CALL]
        statuses = {
            a.tool_execution_data["call_id"]: a.tool_execution_data["status"]
            for a in tool_actions
        }
        assert statuses == {"call_1": "completed", "call_2": "completed"}

    # -- synthetic events for prepare-phase failures -------------------------

    @pytest.mark.asyncio
    async def test_prepare_phase_failure_synthesizes_terminal_event(
        self,
        mock_llm_service: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        # Gateway returns a failed result without emitting any tool events
        # (prepare-phase / validation failure). The engine must backstop with
        # synthetic events so the transcript records the attempt and failure.
        class SilentFailingGateway:
            def __init__(self) -> None:
                self.executed: list[str] = []
                self._callback: Any = None

            def register_event_callback(self, callback: Any) -> None:
                self._callback = callback

            async def execute_many(
                self, requests: list[Any], interrupt_event: Any = None
            ) -> list[ToolExecutionResult]:
                self.executed = [r.call_id for r in requests]
                return [
                    ToolExecutionResult(
                        status=ToolExecutionStatus.FAILED.value,
                        output=None,
                        error=ToolExecutionError(
                            code=ToolErrorCode.INVALID_ARGUMENTS,
                            category=ToolErrorCategory.VALIDATION,
                            message="invalid arguments: missing path",
                            retryable=False,
                        ),
                        call_id=r.call_id,
                        tool_id=r.tool_id,
                    )
                    for r in requests
                ]

        silent_gateway = SilentFailingGateway()
        engine = _make_engine(mock_llm_service, silent_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = _frame_with_tool_calls(ExecutionStatus.RUNNING_TOOLS, ["call_1"])
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.PARTIAL
        # Action closed as failed via the synthetic-event backstop.
        tool_actions = [a for a in result.actions if a.kind == ActionKind.TOOL_CALL]
        assert len(tool_actions) == 1
        assert tool_actions[0].tool_execution_data["status"] == "failed"
        assert "missing path" in (
            tool_actions[0].tool_execution_data.get("error", {}).get("message", "")
        )
        # Synthetic ToolResultEvent emitted for the eventless failure.
        assert any(
            isinstance(c.args[0], ToolResultEvent)
            for c in mock_event_pipeline.emit.await_args_list
        )
