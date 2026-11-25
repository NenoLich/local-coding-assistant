"""Unit tests for ContinuationPolicy.decide().

Covers every FrameOutcome branch, limit/approval/interrupt precedence,
and exact ContinuationAction / StopReason pairs asserted against the
current implementation in runtime/engine/policy.py.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.config.schemas import ContinuationPolicyConfig
from local_coding_assistant.runtime.context.profile import (
    ContextSnapshot,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.engine.approval import (
    ApprovalRequest,
    ApprovalStatus,
)
from local_coding_assistant.runtime.engine.config import (
    LimitCheckResult,
    LimitViolationReason,
    RuntimeLimitsConfig,
)
from local_coding_assistant.runtime.engine.frame import (
    FrameOutcome,
    FrameOutcomeDetail,
    ToolCallInfo,
)
from local_coding_assistant.runtime.engine.policy import (
    ContinuationAction,
    ContinuationPolicy,
    StopReason,
)
from local_coding_assistant.runtime.session.turn import RuntimeTurn
from local_coding_assistant.tools_platform.contracts import ApprovalResponse


@pytest.fixture
def policy_config() -> ContinuationPolicyConfig:
    return ContinuationPolicyConfig(
        max_frames_per_turn=10,
        max_generation_retries=2,
        max_compaction_retries=1,
    )


@pytest.fixture
def policy(policy_config: ContinuationPolicyConfig) -> ContinuationPolicy:
    return ContinuationPolicy(config=policy_config)


def _make_turn(
    *,
    turn_id: str = "turn_123",
    session_id: str = "session_123",
    trace_id: str = "trace_123",
    turn_index: int = 1,
    user_input: str = "hi",
    frame_ids: list[str] | None = None,
    metadata: dict | None = None,
    pending_approvals: dict[str, ApprovalRequest] | None = None,
    interrupt: bool = False,
) -> RuntimeTurn:
    turn = RuntimeTurn(
        turn_id=turn_id,
        session_id=session_id,
        trace_id=trace_id,
        turn_index=turn_index,
        user_input=user_input,
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
        frame_ids=list(frame_ids or []),
        metadata=dict(metadata or {}),
        pending_approvals=dict(pending_approvals or {}),
    )
    if interrupt:
        turn.interrupt_event.set()
    return turn


def _make_outcome(
    outcome: FrameOutcome = FrameOutcome.FINAL_RESPONSE,
    *,
    finish_reason: str | None = "stop",
    final_content: str | None = "final answer",
    tool_calls: list[ToolCallInfo] | None = None,
    metadata: dict | None = None,
) -> FrameOutcomeDetail:
    return FrameOutcomeDetail(
        outcome=outcome,
        finish_reason=finish_reason,
        final_content=final_content,
        tool_calls=list(tool_calls or []),
        metadata=dict(metadata or {}),
    )


def _violated(reason: LimitViolationReason) -> LimitCheckResult:
    return LimitCheckResult.violated(
        reason, detail="limit hit", metadata={"reason": reason.value}
    )


class TestFinalResponse:
    def test_final_response_with_content(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn()
        decision = policy.decide(_make_outcome(FrameOutcome.FINAL_RESPONSE), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.FINAL_ANSWER
        assert decision.metadata["content_length"] == len("final answer")

    def test_final_response_empty_content(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn()
        decision = policy.decide(
            _make_outcome(FrameOutcome.FINAL_RESPONSE, final_content=""), turn
        )

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.UNKNOWN
        assert decision.metadata["reason"] == "empty_final_response"

    def test_final_response_takes_precedence_over_interrupt(
        self, policy: ContinuationPolicy
    ) -> None:
        turn = _make_turn(interrupt=True)
        decision = policy.decide(_make_outcome(FrameOutcome.FINAL_RESPONSE), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.FINAL_ANSWER

    def test_final_response_takes_precedence_over_limits(
        self, policy: ContinuationPolicy
    ) -> None:
        turn = _make_turn()
        turn.check_limits = MagicMock(
            return_value=_violated(LimitViolationReason.TOKEN_BUDGET_EXCEEDED)
        )  # type: ignore[method-assign]
        decision = policy.decide(_make_outcome(FrameOutcome.FINAL_RESPONSE), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.FINAL_ANSWER


class TestInterrupt:
    def test_interrupt(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn(interrupt=True)
        decision = policy.decide(_make_outcome(FrameOutcome.TOOL_ROUND_COMPLETED), turn)

        assert decision.action == ContinuationAction.PAUSE_FOR_INTERRUPT
        assert decision.reason == StopReason.INTERRUPTED
        assert decision.metadata["interrupt"] is True

    def test_interrupt_takes_precedence_over_limits(
        self, policy: ContinuationPolicy
    ) -> None:
        turn = _make_turn(interrupt=True)
        turn.check_limits = MagicMock(
            return_value=_violated(LimitViolationReason.DEADLINE_EXCEEDED)
        )  # type: ignore[method-assign]
        decision = policy.decide(_make_outcome(FrameOutcome.TOOL_ROUND_COMPLETED), turn)

        assert decision.action == ContinuationAction.PAUSE_FOR_INTERRUPT
        assert decision.reason == StopReason.INTERRUPTED


class TestLimits:
    def test_token_budget_exceeded(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn()
        turn.check_limits = MagicMock(
            return_value=_violated(LimitViolationReason.TOKEN_BUDGET_EXCEEDED)
        )  # type: ignore[method-assign]
        decision = policy.decide(_make_outcome(FrameOutcome.TOOL_ROUND_COMPLETED), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.TOKEN_BUDGET_DEPLETED

    def test_deadline_exceeded(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn()
        turn.check_limits = MagicMock(
            return_value=_violated(LimitViolationReason.DEADLINE_EXCEEDED)
        )  # type: ignore[method-assign]
        decision = policy.decide(_make_outcome(FrameOutcome.TOOL_ROUND_COMPLETED), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.DEADLINE_EXCEEDED

    def test_loop_detected(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn()
        turn.check_limits = MagicMock(
            return_value=_violated(LimitViolationReason.LOOP_DETECTED)
        )  # type: ignore[method-assign]
        decision = policy.decide(_make_outcome(FrameOutcome.TOOL_ROUND_COMPLETED), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.LOOP_DETECTED

    def test_generation_retry_exhausted(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn()
        turn.check_limits = MagicMock(
            return_value=_violated(LimitViolationReason.GENERATION_RETRY_EXHAUSTED)
        )  # type: ignore[method-assign]
        decision = policy.decide(_make_outcome(FrameOutcome.TOOL_ROUND_COMPLETED), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.GENERATION_RETRY_EXHAUSTED

    def test_compaction_retry_exhausted(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn()
        turn.check_limits = MagicMock(
            return_value=_violated(LimitViolationReason.COMPACTION_RETRY_EXHAUSTED)
        )  # type: ignore[method-assign]
        decision = policy.decide(_make_outcome(FrameOutcome.TOOL_ROUND_COMPLETED), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.TRUNCATION_RETRY_EXHAUSTED

    def test_limits_take_precedence_over_outcome(
        self, policy: ContinuationPolicy
    ) -> None:
        turn = _make_turn()
        turn.check_limits = MagicMock(
            return_value=_violated(LimitViolationReason.LOOP_DETECTED)
        )  # type: ignore[method-assign]
        decision = policy.decide(_make_outcome(FrameOutcome.TOOL_ROUND_COMPLETED), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.LOOP_DETECTED


class TestFrameCap:
    def test_frame_limit_reached(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn(frame_ids=[f"frame_{i}" for i in range(10)])
        decision = policy.decide(_make_outcome(FrameOutcome.TOOL_ROUND_COMPLETED), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.FRAME_LIMIT_REACHED
        assert decision.metadata["frames_executed"] == 10
        assert decision.metadata["max_frames"] == 10

    def test_frame_cap_takes_precedence_over_outcome(
        self, policy: ContinuationPolicy
    ) -> None:
        turn = _make_turn(frame_ids=[f"frame_{i}" for i in range(10)])
        decision = policy.decide(_make_outcome(FrameOutcome.TOOL_ROUND_COMPLETED), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.FRAME_LIMIT_REACHED

    def test_frame_cap_not_reached(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn(frame_ids=[f"frame_{i}" for i in range(3)])
        decision = policy.decide(_make_outcome(FrameOutcome.TOOL_ROUND_COMPLETED), turn)

        assert decision.action == ContinuationAction.CONTINUE
        assert decision.reason == StopReason.TOOL_ROUND_COMPLETE


class TestToolRoundCompleted:
    def test_tool_round_completed(self, policy: ContinuationPolicy) -> None:
        tool_calls = [
            ToolCallInfo(call_id="c1", name="read_file", arguments={}, success=True),
            ToolCallInfo(call_id="c2", name="write_file", arguments={}, success=True),
        ]
        turn = _make_turn()
        decision = policy.decide(
            _make_outcome(FrameOutcome.TOOL_ROUND_COMPLETED, tool_calls=tool_calls),
            turn,
        )

        assert decision.action == ContinuationAction.CONTINUE
        assert decision.reason == StopReason.TOOL_ROUND_COMPLETE
        assert decision.metadata["tools_executed"] == 2


class TestTruncation:
    def test_output_truncated_with_content_stops(
        self, policy: ContinuationPolicy
    ) -> None:
        turn = _make_turn(metadata={"truncation_retry_count": 0})
        decision = policy.decide(
            _make_outcome(
                FrameOutcome.OUTPUT_TRUNCATED,
                final_content="partial answer here",
            ),
            turn,
        )

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.OUTPUT_TRUNCATED
        assert decision.metadata["reason"] == "truncated_with_partial_content"
        assert decision.metadata["partial_content_length"] == len("partial answer here")

    def test_output_truncated_with_content_retries_ignored(
        self, policy: ContinuationPolicy
    ) -> None:
        turn = _make_turn(metadata={"truncation_retry_count": 1})
        decision = policy.decide(
            _make_outcome(
                FrameOutcome.OUTPUT_TRUNCATED,
                final_content="partial answer here",
            ),
            turn,
        )

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.OUTPUT_TRUNCATED

    def test_output_truncated_no_content(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn(metadata={"truncation_retry_count": 0})
        decision = policy.decide(
            _make_outcome(FrameOutcome.OUTPUT_TRUNCATED, final_content=None), turn
        )

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.TRUNCATION_RETRY_EXHAUSTED
        assert decision.metadata["retry_count"] == 0
        assert decision.metadata["max_retries"] == 2

    def test_output_truncated_with_tool_calls(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn()
        decision = policy.decide(
            _make_outcome(
                FrameOutcome.OUTPUT_TRUNCATED_WITH_TOOL_CALLS,
                final_content="partial",
                tool_calls=[
                    ToolCallInfo(
                        call_id="c1", name="run_cmd", arguments={}, success=False
                    )
                ],
            ),
            turn,
        )

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.OUTPUT_TRUNCATED_WITH_TOOLS
        assert decision.metadata["reason"] == "truncated_with_incomplete_tool_calls"


class TestContextOverflow:
    def test_context_overflow_compact_and_continue(
        self, policy: ContinuationPolicy
    ) -> None:
        turn = _make_turn(metadata={"compaction_count": 0})
        decision = policy.decide(
            _make_outcome(
                FrameOutcome.CONTEXT_OVERFLOW,
                metadata={"estimated_input_tokens": 5000},
            ),
            turn,
        )

        assert decision.action == ContinuationAction.COMPACT_AND_CONTINUE
        assert decision.reason == StopReason.OUTPUT_TRUNCATED
        assert decision.retry_count == 1
        assert decision.compact_instruction is not None
        assert decision.metadata["compaction_count"] == 1
        assert decision.metadata["max_compactions"] == 1

    def test_context_overflow_exhausted(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn(metadata={"compaction_count": 1})
        decision = policy.decide(_make_outcome(FrameOutcome.CONTEXT_OVERFLOW), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.TOKEN_BUDGET_DEPLETED
        assert decision.metadata["compaction_count"] == 1
        assert decision.metadata["max_compactions"] == 1


class TestToolFailure:
    def test_tool_failure_recoverable(self, policy: ContinuationPolicy) -> None:
        tool_calls = [
            ToolCallInfo(call_id="c1", name="read_file", arguments={}, success=True),
            ToolCallInfo(call_id="c2", name="delete_file", arguments={}, success=False),
        ]
        turn = _make_turn()
        decision = policy.decide(
            _make_outcome(FrameOutcome.TOOL_FAILURE_RECOVERABLE, tool_calls=tool_calls),
            turn,
        )

        assert decision.action == ContinuationAction.CONTINUE
        assert decision.reason == StopReason.TOOL_FAILURE_RECOVERABLE
        assert decision.metadata["failed_tools"] == ["delete_file"]
        assert decision.metadata["tools_executed"] == 2

    def test_tool_failure_non_recoverable(self, policy: ContinuationPolicy) -> None:
        tool_calls = [
            ToolCallInfo(call_id="c1", name="delete_file", arguments={}, success=False),
        ]
        turn = _make_turn()
        decision = policy.decide(
            _make_outcome(
                FrameOutcome.TOOL_FAILURE_NON_RECOVERABLE, tool_calls=tool_calls
            ),
            turn,
        )

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.TOOL_FAILURE_NON_RECOVERABLE
        assert decision.metadata["failed_tools"] == ["delete_file"]


class TestIncompleteToolCall:
    def test_incomplete_tool_call_retries_left(
        self, policy: ContinuationPolicy
    ) -> None:
        turn = _make_turn(metadata={"generation_retry_count": 0})
        decision = policy.decide(
            _make_outcome(
                FrameOutcome.INCOMPLETE_TOOL_CALL,
                tool_calls=[
                    ToolCallInfo(
                        call_id="c1", name="run_cmd", arguments={}, success=False
                    )
                ],
            ),
            turn,
        )

        assert decision.action == ContinuationAction.RETRY_GENERATION
        assert decision.reason == StopReason.INCOMPLETE_TOOL_CALL
        assert decision.retry_count == 1
        assert decision.metadata["incomplete_tools"] == ["run_cmd"]

    def test_incomplete_tool_call_exhausted(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn(metadata={"generation_retry_count": 2})
        decision = policy.decide(_make_outcome(FrameOutcome.INCOMPLETE_TOOL_CALL), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.GENERATION_RETRY_EXHAUSTED
        assert decision.metadata["retry_count"] == 2
        assert decision.metadata["max_retries"] == 2


class TestProviderTransientError:
    def test_provider_transient_error_retries_left(
        self, policy: ContinuationPolicy
    ) -> None:
        turn = _make_turn(metadata={"generation_retry_count": 0})
        decision = policy.decide(
            _make_outcome(
                FrameOutcome.PROVIDER_TRANSIENT_ERROR,
                metadata={
                    "error_kind": "timeout",
                    "status_code": 503,
                    "retry_after": 2.0,
                },
            ),
            turn,
        )

        assert decision.action == ContinuationAction.RETRY_GENERATION
        assert decision.reason == StopReason.PROVIDER_TRANSIENT_ERROR
        assert decision.retry_count == 1
        assert decision.metadata["error_kind"] == "timeout"
        assert decision.metadata["status_code"] == 503
        assert decision.metadata["retry_after"] == 2.0

    def test_provider_transient_error_exhausted(
        self, policy: ContinuationPolicy
    ) -> None:
        turn = _make_turn(metadata={"generation_retry_count": 2})
        decision = policy.decide(
            _make_outcome(
                FrameOutcome.PROVIDER_TRANSIENT_ERROR,
                metadata={"error_kind": "timeout", "status_code": 503},
            ),
            turn,
        )

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.PROVIDER_TRANSIENT_ERROR
        assert decision.metadata["retry_count"] == 2
        assert decision.metadata["max_retries"] == 2

    def test_transient_retry_delay_exponential(
        self, policy: ContinuationPolicy
    ) -> None:
        # retry_count=0 -> base 1.0 * 2**0 = 1.0
        assert policy._transient_retry_delay(0, None) == 1.0
        # retry_count=1 -> base 1.0 * 2**1 = 2.0
        assert policy._transient_retry_delay(1, None) == 2.0
        # retry_count=5 -> 32.0, below the max so not clamped
        assert policy._transient_retry_delay(5, None) == 32.0
        # retry_count=6 -> 64.0, clamped to max 60.0
        assert policy._transient_retry_delay(6, None) == 60.0

    def test_transient_retry_delay_respects_retry_after(
        self, policy: ContinuationPolicy
    ) -> None:
        assert policy._transient_retry_delay(0, 10.0) == 10.0
        # Clamped to max when retry_after exceeds max
        assert policy._transient_retry_delay(0, 120.0) == 60.0


class TestProviderNonTransientError:
    def test_provider_non_transient_error(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn()
        decision = policy.decide(
            _make_outcome(
                FrameOutcome.PROVIDER_NON_TRANSIENT_ERROR,
                metadata={"error_message": "unauthorized"},
            ),
            turn,
        )

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.PROVIDER_NON_TRANSIENT_ERROR
        assert decision.metadata["error"] == "unauthorized"


class TestContentBlocked:
    def test_content_blocked(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn()
        decision = policy.decide(
            _make_outcome(FrameOutcome.CONTENT_BLOCKED, finish_reason="content_filter"),
            turn,
        )

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.CONTENT_BLOCKED
        assert decision.metadata["finish_reason"] == "content_filter"


class TestApproval:
    def _pending_approval(self, **overrides) -> ApprovalRequest:
        data = dict(
            token="tok_1",
            tool_call_id="call_1",
            tool_name="run_cmd",
            arguments={"cmd": "echo hi"},
            risk_class="medium",
            session_id="session_123",
            frame_id="frame_123",
            turn_id="turn_123",
        )
        data.update(overrides)
        return ApprovalRequest(**data)

    def test_approval_pending(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn(
            pending_approvals={
                "call_1": self._pending_approval(status=ApprovalStatus.PENDING)
            }
        )
        decision = policy.decide(_make_outcome(FrameOutcome.APPROVAL_PENDING), turn)

        assert decision.action == ContinuationAction.PAUSE_FOR_APPROVAL
        assert decision.reason == StopReason.APPROVAL_PENDING
        assert decision.metadata["tool"] == "run_cmd"
        assert decision.metadata["call_id"] == "call_1"
        assert decision.metadata["token"] == "tok_1"

    def test_approval_denied(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn(
            pending_approvals={
                "call_1": self._pending_approval(
                    status=ApprovalStatus.DENIED, response=ApprovalResponse.NO
                )
            }
        )
        decision = policy.decide(_make_outcome(FrameOutcome.APPROVAL_DENIED), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.APPROVAL_DENIED
        assert decision.metadata["tool"] == "run_cmd"

    def test_approval_expired(self, policy: ContinuationPolicy) -> None:
        req = self._pending_approval(status=ApprovalStatus.PENDING)
        req.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        turn = _make_turn(pending_approvals={"call_1": req})
        decision = policy.decide(_make_outcome(FrameOutcome.APPROVAL_EXPIRED), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.APPROVAL_EXPIRED
        assert decision.metadata["tool"] == "run_cmd"

    def test_pending_approval_takes_precedence_over_outcome(
        self, policy: ContinuationPolicy
    ) -> None:
        turn = _make_turn(
            pending_approvals={
                "call_1": self._pending_approval(status=ApprovalStatus.PENDING)
            }
        )
        decision = policy.decide(_make_outcome(FrameOutcome.TOOL_ROUND_COMPLETED), turn)

        assert decision.action == ContinuationAction.PAUSE_FOR_APPROVAL
        assert decision.reason == StopReason.APPROVAL_PENDING


class TestUnknown:
    def test_unknown_outcome(self, policy: ContinuationPolicy) -> None:
        turn = _make_turn()
        decision = policy.decide(_make_outcome(FrameOutcome.UNKNOWN), turn)

        assert decision.action == ContinuationAction.STOP
        assert decision.reason == StopReason.UNKNOWN
        assert decision.metadata["outcome"] == "unknown"
