"""Continuation policy for frame execution.

Purely maps a frame outcome and turn state to a typed action and reason.
Does not call providers, tools, mutate prompts, or keep hidden retry state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from local_coding_assistant.config.schemas import ContinuationPolicyConfig
from local_coding_assistant.runtime.engine.config import (
    LimitCheckResult,
    LimitViolationReason,
)
from local_coding_assistant.runtime.engine.frame import FrameOutcome, FrameOutcomeDetail
from local_coding_assistant.runtime.session.turn import RuntimeTurn


class ContinuationAction(str, Enum):
    """Action decided by the continuation policy."""

    STOP = "stop"  # Turn is complete, finalize
    CONTINUE = "continue"  # Start a new model frame after observations
    COMPACT_AND_CONTINUE = "compact_and_continue"  # Compact history then continue
    RETRY_GENERATION = (
        "retry_generation"  # Retry the LLM generation with adjusted params
    )
    PAUSE_FOR_APPROVAL = (
        "pause_for_approval"  # Wait for approval (frame already paused)
    )
    PAUSE_FOR_INTERRUPT = (
        "pause_for_interrupt"  # Wait for user interrupt (frame already paused)
    )
    PAUSE_FOR_DEADLINE = (
        "pause_for_deadline"  # Wait for deadline extension (frame already paused)
    )
    PAUSE_FOR_DENIED = "pause_for_denied"  # Approval denied, frame paused


class StopReason(str, Enum):
    """Stable reason codes for terminal decisions."""

    # Success
    FINAL_ANSWER = "final_answer"  # Model produced final user-facing response
    TURN_COMPLETE = "turn_complete"  # Turn completed normally
    TOOL_ROUND_COMPLETE = "tool_round_complete"  # Tool round completed normally

    # Truncation
    OUTPUT_TRUNCATED = "output_truncated"  # Output truncated, partial preserved
    OUTPUT_TRUNCATED_WITH_TOOLS = (
        "output_truncated_with_tools"  # Truncated mid-tool-call
    )
    TRUNCATION_RETRY_EXHAUSTED = (
        "truncation_retry_exhausted"  # Retried truncation, still truncated
    )

    # Tool failures
    TOOL_FAILURE_RECOVERABLE = (
        "tool_failure_recoverable"  # Tool failed, model can correct
    )
    TOOL_FAILURE_NON_RECOVERABLE = (
        "tool_failure_non_recoverable"  # Tool failed, not safe to retry
    )
    INCOMPLETE_TOOL_CALL = "incomplete_tool_call"  # Malformed tool call from model

    # Provider errors
    PROVIDER_TRANSIENT_ERROR = (
        "provider_transient_error"  # Network/timeout, retried but exhausted
    )
    PROVIDER_NON_TRANSIENT_ERROR = (
        "provider_non_transient_error"  # Auth/permission/invalid request
    )

    # Content
    CONTENT_BLOCKED = "content_blocked"  # Safety filter blocked response

    # Approval
    APPROVAL_PENDING = "approval_pending"  # Tool requires approval, paused
    APPROVAL_DENIED = "approval_denied"  # Approval explicitly denied
    APPROVAL_EXPIRED = "approval_expired"  # Approval timed out

    # Limits
    FRAME_LIMIT_REACHED = "frame_limit_reached"  # Max frames per turn exceeded
    TOKEN_BUDGET_DEPLETED = (
        "token_budget_depleted"  # Turn token budget exceeded  # noqa: S105
    )
    DEADLINE_EXCEEDED = "deadline_exceeded"  # Turn deadline exceeded
    LOOP_DETECTED = "loop_detected"  # LoopGuard detected repetition
    GENERATION_RETRY_EXHAUSTED = (
        "generation_retry_exhausted"  # Max generation retries reached
    )

    # Interrupt
    INTERRUPTED = "interrupted"  # User interrupt (Ctrl+C, stop button)

    # Unknown
    UNKNOWN = "unknown"  # Unknown or malformed state


@dataclass(frozen=True)
class ContinuationDecision:
    """Decision from the continuation policy."""

    action: ContinuationAction
    reason: StopReason
    # Parameters for the action
    max_tokens_adjustment: int | None = (
        None  # For RETRY_GENERATION/COMPACT_AND_CONTINUE
    )
    temperature_adjustment: float | None = None
    compact_instruction: str | None = None  # For COMPACT_AND_CONTINUE
    retry_count: int = 0  # Current retry attempt
    metadata: dict[str, Any] | None = None

    def to_event_payload(self) -> dict[str, Any]:
        """Convert to event payload for CONTINUATION_DECISION event."""
        return {
            "action": self.action.value,
            "reason": self.reason.value,
            "max_tokens_adjustment": self.max_tokens_adjustment,
            "temperature_adjustment": self.temperature_adjustment,
            "compact_instruction": self.compact_instruction,
            "retry_count": self.retry_count,
            "metadata": self.metadata or {},
        }


class ContinuationPolicy:
    """Pure policy for deciding continuation after a frame completes.

    Consumes typed semantic outcomes; does not infer from ExecutionStatus alone.
    All decisions are deterministic and testable.
    Uses config from RuntimeTurn for resume/replay compatibility.
    """

    def __init__(self, config: ContinuationPolicyConfig) -> None:
        self.config = config

    def decide(  # noqa: C901
        self,
        frame_outcome: FrameOutcomeDetail,
        turn: RuntimeTurn,
    ) -> ContinuationDecision:
        """Decide next action from frame outcome and turn state.

        Applies rules in order; terminal/safety/limit conditions take precedence
        over recovery. The policy does not mutate turn or perform actions.
        """
        outcome = frame_outcome.outcome

        # A final response completes the turn; it takes precedence over
        # interrupts and limits so the produced answer is not discarded.
        if outcome == FrameOutcome.FINAL_RESPONSE:
            return self._decide_final_response(frame_outcome)

        # Check for interrupt first (highest priority for pause)
        if turn.interrupt_event.is_set():
            return ContinuationDecision(
                action=ContinuationAction.PAUSE_FOR_INTERRUPT,
                reason=StopReason.INTERRUPTED,
                metadata={"interrupt": True},
            )

        # Check turn limits first (these are terminal) - use structured result
        limit_result = turn.check_limits()
        if not limit_result.within_limits:
            return self._limit_result_to_decision(limit_result)

        # Check frame cap
        if len(turn.frame_ids) >= self.config.max_frames_per_turn:
            return ContinuationDecision(
                action=ContinuationAction.STOP,
                reason=StopReason.FRAME_LIMIT_REACHED,
                metadata={
                    "frames_executed": len(turn.frame_ids),
                    "max_frames": self.config.max_frames_per_turn,
                },
            )

        # Check for pending approval
        if turn.pending_approvals:
            return self._decide_approval(turn)

        # Route based on semantic outcome
        if outcome == FrameOutcome.TOOL_ROUND_COMPLETED:
            return self._decide_tool_round_completed(frame_outcome)

        if outcome in (
            FrameOutcome.OUTPUT_TRUNCATED,
            FrameOutcome.OUTPUT_TRUNCATED_WITH_TOOL_CALLS,
        ):
            return self._decide_truncation(frame_outcome, turn)

        if outcome == FrameOutcome.CONTEXT_OVERFLOW:
            return self._decide_context_overflow(frame_outcome, turn)

        if outcome == FrameOutcome.TOOL_FAILURE_RECOVERABLE:
            return self._decide_tool_failure_recoverable(frame_outcome)

        if outcome == FrameOutcome.TOOL_FAILURE_NON_RECOVERABLE:
            return self._decide_tool_failure_non_recoverable(frame_outcome)

        if outcome == FrameOutcome.INCOMPLETE_TOOL_CALL:
            return self._decide_incomplete_tool_call(frame_outcome, turn)

        if outcome == FrameOutcome.PROVIDER_TRANSIENT_ERROR:
            return self._decide_provider_transient_error(frame_outcome, turn)

        if outcome == FrameOutcome.PROVIDER_NON_TRANSIENT_ERROR:
            return self._decide_provider_non_transient_error(frame_outcome)

        if outcome == FrameOutcome.CONTENT_BLOCKED:
            return self._decide_content_blocked(frame_outcome)

        # Unknown/malformed
        return ContinuationDecision(
            action=ContinuationAction.STOP,
            reason=StopReason.UNKNOWN,
            metadata={
                "outcome": outcome.value,
                "execution_status": frame_outcome.metadata.get("execution_status"),
            },
        )

    def _limit_result_to_decision(
        self, result: LimitCheckResult
    ) -> ContinuationDecision:
        """Convert LimitCheckResult to ContinuationDecision."""
        reason_map = {
            LimitViolationReason.TOKEN_BUDGET_EXCEEDED: StopReason.TOKEN_BUDGET_DEPLETED,
            LimitViolationReason.DEADLINE_EXCEEDED: StopReason.DEADLINE_EXCEEDED,
            LimitViolationReason.LOOP_DETECTED: StopReason.LOOP_DETECTED,
            LimitViolationReason.FRAME_LIMIT_REACHED: StopReason.FRAME_LIMIT_REACHED,
            LimitViolationReason.GENERATION_RETRY_EXHAUSTED: StopReason.GENERATION_RETRY_EXHAUSTED,
            LimitViolationReason.COMPACTION_RETRY_EXHAUSTED: StopReason.TRUNCATION_RETRY_EXHAUSTED,
        }
        if result.reason is None:
            reason = StopReason.UNKNOWN
        else:
            reason = reason_map.get(result.reason, StopReason.UNKNOWN)
        return ContinuationDecision(
            action=ContinuationAction.STOP,
            reason=reason,
            metadata=result.metadata,
        )

    def _decide_approval(self, turn: RuntimeTurn) -> ContinuationDecision:
        pending = next(iter(turn.pending_approvals.values()))
        if pending.is_expired:
            return ContinuationDecision(
                action=ContinuationAction.STOP,
                reason=StopReason.APPROVAL_EXPIRED,
                metadata={
                    "tool": pending.tool_name,
                    "call_id": pending.tool_call_id,
                },
            )
        # Check if approval was explicitly denied
        from local_coding_assistant.tools_platform.contracts import ApprovalResponse

        if pending.response == ApprovalResponse.NO:
            return ContinuationDecision(
                action=ContinuationAction.STOP,
                reason=StopReason.APPROVAL_DENIED,
                metadata={
                    "tool": pending.tool_name,
                    "call_id": pending.tool_call_id,
                    "token": pending.token,
                },
            )
        return ContinuationDecision(
            action=ContinuationAction.PAUSE_FOR_APPROVAL,
            reason=StopReason.APPROVAL_PENDING,
            metadata={
                "tool": pending.tool_name,
                "call_id": pending.tool_call_id,
                "token": pending.token,
            },
        )

    def _decide_final_response(
        self, frame_outcome: FrameOutcomeDetail
    ) -> ContinuationDecision:
        if not frame_outcome.final_content or not frame_outcome.final_content.strip():
            return ContinuationDecision(
                action=ContinuationAction.STOP,
                reason=StopReason.UNKNOWN,
                metadata={"reason": "empty_final_response"},
            )
        return ContinuationDecision(
            action=ContinuationAction.STOP,
            reason=StopReason.FINAL_ANSWER,
            metadata={"content_length": len(frame_outcome.final_content or "")},
        )

    def _decide_tool_round_completed(
        self, frame_outcome: FrameOutcomeDetail
    ) -> ContinuationDecision:
        return ContinuationDecision(
            action=ContinuationAction.CONTINUE,
            reason=StopReason.TOOL_ROUND_COMPLETE,
            metadata={"tools_executed": len(frame_outcome.tool_calls)},
        )

    def _decide_tool_failure_recoverable(
        self, frame_outcome: FrameOutcomeDetail
    ) -> ContinuationDecision:
        return ContinuationDecision(
            action=ContinuationAction.CONTINUE,
            reason=StopReason.TOOL_FAILURE_RECOVERABLE,
            metadata={
                "failed_tools": [
                    tc.name for tc in frame_outcome.tool_calls if not tc.success
                ],
                "tools_executed": len(frame_outcome.tool_calls),
            },
        )

    def _decide_tool_failure_non_recoverable(
        self, frame_outcome: FrameOutcomeDetail
    ) -> ContinuationDecision:
        return ContinuationDecision(
            action=ContinuationAction.STOP,
            reason=StopReason.TOOL_FAILURE_NON_RECOVERABLE,
            metadata={
                "failed_tools": [
                    tc.name for tc in frame_outcome.tool_calls if not tc.success
                ],
            },
        )

    def _decide_incomplete_tool_call(
        self, frame_outcome: FrameOutcomeDetail, turn: RuntimeTurn
    ) -> ContinuationDecision:
        retry_count = turn.metadata.get("generation_retry_count", 0)
        if retry_count < self.config.max_generation_retries:
            return ContinuationDecision(
                action=ContinuationAction.RETRY_GENERATION,
                reason=StopReason.INCOMPLETE_TOOL_CALL,
                retry_count=retry_count + 1,
                metadata={
                    "incomplete_tools": [tc.name for tc in frame_outcome.tool_calls]
                },
            )
        return ContinuationDecision(
            action=ContinuationAction.STOP,
            reason=StopReason.GENERATION_RETRY_EXHAUSTED,
            metadata={
                "retry_count": retry_count,
                "max_retries": self.config.max_generation_retries,
            },
        )

    def _decide_provider_transient_error(
        self, frame_outcome: FrameOutcomeDetail, turn: RuntimeTurn
    ) -> ContinuationDecision:
        retry_count = turn.metadata.get("generation_retry_count", 0)
        retry_after = frame_outcome.metadata.get("retry_after")
        delay = self._transient_retry_delay(retry_count, retry_after)
        if retry_count < self.config.max_generation_retries:
            return ContinuationDecision(
                action=ContinuationAction.RETRY_GENERATION,
                reason=StopReason.PROVIDER_TRANSIENT_ERROR,
                retry_count=retry_count + 1,
                metadata={
                    "error_kind": frame_outcome.metadata.get("error_kind"),
                    "status_code": frame_outcome.metadata.get("status_code"),
                    "retry_after": retry_after,
                    "delay_before_retry": delay,
                },
            )
        return ContinuationDecision(
            action=ContinuationAction.STOP,
            reason=StopReason.PROVIDER_TRANSIENT_ERROR,
            metadata={
                "retry_count": retry_count,
                "max_retries": self.config.max_generation_retries,
                "error_kind": frame_outcome.metadata.get("error_kind"),
                "status_code": frame_outcome.metadata.get("status_code"),
                "retry_after": retry_after,
            },
        )

    def _transient_retry_delay(
        self, retry_count: int, retry_after: float | None
    ) -> float:
        """Compute the backoff before retrying a transient provider error.

        Honors an explicit Retry-After hint when configured to do so;
        otherwise applies exponential backoff from the configured base
        delay, clamped to the configured maximum.
        """
        if retry_after is not None and self.config.respect_retry_after:
            return max(
                0.0, min(float(retry_after), self.config.transient_retry_max_delay)
            )
        delay = self.config.transient_retry_base_delay * (2**retry_count)
        return max(0.0, min(delay, self.config.transient_retry_max_delay))

    def _decide_provider_non_transient_error(
        self, frame_outcome: FrameOutcomeDetail
    ) -> ContinuationDecision:
        return ContinuationDecision(
            action=ContinuationAction.STOP,
            reason=StopReason.PROVIDER_NON_TRANSIENT_ERROR,
            metadata={"error": frame_outcome.metadata.get("error_message")},
        )

    def _decide_content_blocked(
        self, frame_outcome: FrameOutcomeDetail
    ) -> ContinuationDecision:
        return ContinuationDecision(
            action=ContinuationAction.STOP,
            reason=StopReason.CONTENT_BLOCKED,
            metadata={"finish_reason": frame_outcome.finish_reason},
        )

    def _decide_truncation(
        self, frame_outcome: FrameOutcomeDetail, turn: RuntimeTurn
    ) -> ContinuationDecision:
        """Decide action for output truncation.

        Compaction is not yet implemented, so truncation always stops
        (no COMPACT_AND_CONTINUE loop).
        """
        retry_count = turn.metadata.get("truncation_retry_count", 0)

        # No useful content or retries exhausted
        if frame_outcome.outcome == FrameOutcome.OUTPUT_TRUNCATED_WITH_TOOL_CALLS:
            return ContinuationDecision(
                action=ContinuationAction.STOP,
                reason=StopReason.OUTPUT_TRUNCATED_WITH_TOOLS,
                metadata={
                    "reason": "truncated_with_incomplete_tool_calls",
                    "partial_content_length": len(frame_outcome.final_content or ""),
                },
            )

        if frame_outcome.final_content and frame_outcome.final_content.strip():
            # Useful partial content preserved - stop with truncation reason
            return ContinuationDecision(
                action=ContinuationAction.STOP,
                reason=StopReason.OUTPUT_TRUNCATED,
                metadata={
                    "reason": "truncated_with_partial_content",
                    "truncation_retry_count": retry_count,
                    "max_retries": self.config.max_generation_retries,
                    "partial_content_length": len(frame_outcome.final_content or ""),
                },
            )

        return ContinuationDecision(
            action=ContinuationAction.STOP,
            reason=StopReason.TRUNCATION_RETRY_EXHAUSTED,
            metadata={
                "retry_count": retry_count,
                "max_retries": self.config.max_generation_retries,
                "partial_content_length": len(frame_outcome.final_content or ""),
            },
        )

    def _decide_context_overflow(
        self, frame_outcome: FrameOutcomeDetail, turn: RuntimeTurn
    ) -> ContinuationDecision:
        """Decide action for context overflow (prompt too large)."""
        compaction_count = turn.metadata.get("compaction_count", 0)

        if compaction_count < self.config.max_compaction_retries:
            return ContinuationDecision(
                action=ContinuationAction.COMPACT_AND_CONTINUE,
                reason=StopReason.OUTPUT_TRUNCATED,  # Reuse truncation reason for context overflow
                retry_count=compaction_count + 1,
                compact_instruction="Reduce history to essential context: preserve user request, tool results, and recent decisions.",
                metadata={
                    "compaction_count": compaction_count + 1,
                    "max_compactions": self.config.max_compaction_retries,
                    "estimated_tokens": frame_outcome.metadata.get(
                        "estimated_input_tokens", 0
                    ),
                },
            )

        return ContinuationDecision(
            action=ContinuationAction.STOP,
            reason=StopReason.TOKEN_BUDGET_DEPLETED,
            metadata={
                "compaction_count": compaction_count,
                "max_compactions": self.config.max_compaction_retries,
                "reason": "context_overflow_no_progress",
            },
        )
