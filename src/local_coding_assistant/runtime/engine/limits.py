"""Runtime limits enforcement: TokenBudget, DeadlineTracker, LoopGuard, ToolRetryGuard.

These components enforce execution limits at the frame/turn level to prevent
unbounded resource consumption and infinite loops.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from local_coding_assistant.config.schemas import (
    ContinuationPolicyConfig,
    DeadlineTrackerConfig,
    LoopDetectionConfig,
    TokenBudgetConfig,
)
from local_coding_assistant.runtime.engine.config import (
    LimitCheckResult,
    LimitViolationReason,
    RuntimeLimitsConfig,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.engine.limits")


@dataclass
class LoopGuard:
    """Detects repeated tool calls to prevent infinite loops.

    Tracks tool call signatures (tool name + normalized arguments) within a turn
    and detects when the same signature is repeated within a bounded sliding window.

    Detection modes:
    - Consecutive: identical tool+args signature called twice in a row
    - Window-based: same tool+args signature appears N times within sliding window

    The guard counts exact signatures, not just tool names. Different arguments
    to the same tool are treated as distinct calls. The sliding window properly
    evicts old entries and decrements their counts.
    """

    config: LoopDetectionConfig = field(default_factory=LoopDetectionConfig)
    # History stores (call_id, tool_name, normalized_args) tuples in chronological order
    _tool_call_history: list[tuple[str, str, dict[str, Any]]] = field(
        default_factory=list
    )
    # Counts signatures (tool_name, args_hash) within the current window
    _pattern_counts: defaultdict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record_tool_call(
        self, call_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        """Record a tool call for loop detection.

        Args:
            call_id: Unique identifier for the tool call
            tool_name: Name of the tool being called
            arguments: Tool arguments (normalized for comparison)
        """
        if not self.config.enabled:
            return

        # Skip recording if the tool with the same call_id already present
        if any(
            tool_call_id == call_id for tool_call_id, _, _ in self._tool_call_history
        ):
            return

        # Normalize arguments for comparison (sort keys only, preserve values exactly)
        normalized_args = self._normalize_arguments(arguments)
        call_signature = (tool_name, normalized_args)

        # Compute a hashable key for the signature to use in pattern_counts
        sig_key = self._signature_key(call_signature)

        self._tool_call_history.append((call_id, tool_name, normalized_args))
        self._pattern_counts[sig_key] += 1

        # Evict old entries if window exceeded
        if len(self._tool_call_history) > self.config.window_size:
            _, tool_name, normalized_args = self._tool_call_history.pop(0)
            evicted_key = self._signature_key((tool_name, normalized_args))
            self._pattern_counts[evicted_key] -= 1
            if self._pattern_counts[evicted_key] <= 0:
                del self._pattern_counts[evicted_key]

        logger.debug(
            "Tool call recorded",
            tool=tool_name,
            history_length=len(self._tool_call_history),
            pattern_count=self._pattern_counts[sig_key],
            window_size=self.config.window_size,
        )

    def check_loop(self) -> LimitCheckResult:
        """Check if a loop pattern is detected.

        Returns:
            LimitCheckResult with structured violation info
        """
        if not self.config.enabled:
            return LimitCheckResult.ok()

        # Check for identical consecutive calls (exact signature match)
        if self.config.check_consecutive and len(self._tool_call_history) >= 2:
            _, last_tool_call_name, last_tool_call_args = self._tool_call_history[-1]
            _, prev_tool_call_name, prev_tool_call_args = self._tool_call_history[-2]
            if (
                last_tool_call_name == prev_tool_call_name
                and last_tool_call_args == prev_tool_call_args
            ):
                return LimitCheckResult.violated(
                    reason=LimitViolationReason.LOOP_DETECTED,
                    detail=f"Identical consecutive calls to {last_tool_call_name}",
                    metadata={
                        "tool": last_tool_call_name,
                        "detection_mode": "consecutive",
                        "arguments": last_tool_call_args,
                    },
                )

        # Check for excessive repetitions of same signature within window
        if self.config.check_window_repetitions:
            for sig_key, count in self._pattern_counts.items():
                if count >= self.config.max_repetitions:
                    # Find the signature for detail
                    tool_name, _ = sig_key
                    matching_sig = next(
                        (
                            (recorded_tool_name, normalized_args)
                            for _, recorded_tool_name, normalized_args in self._tool_call_history
                            if self._signature_key(
                                (recorded_tool_name, normalized_args)
                            )
                            == sig_key
                        ),
                        None,
                    )
                    return LimitCheckResult.violated(
                        reason=LimitViolationReason.LOOP_DETECTED,
                        detail=f"Tool {tool_name} with same arguments called {count} times within window (max: {self.config.max_repetitions})",
                        metadata={
                            "tool": tool_name,
                            "detection_mode": "window",
                            "count": count,
                            "max_repetitions": self.config.max_repetitions,
                            "window_size": self.config.window_size,
                            "arguments": matching_sig[1] if matching_sig else None,
                        },
                    )

        return LimitCheckResult.ok()

    def reset(self) -> None:
        """Reset loop detection state for new turn."""
        self._tool_call_history.clear()
        self._pattern_counts.clear()

    def _signature_key(self, signature: tuple[str, dict[str, Any]]) -> tuple[str, str]:
        """Create a hashable key from a signature for use in pattern_counts."""
        tool_name, normalized_args = signature
        # Serialize normalized args to a stable string key
        import json

        args_str = json.dumps(normalized_args, sort_keys=True, separators=(",", ":"))
        return tool_name, args_str

    def _normalize_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Normalize arguments for comparison.

        Only sorts dictionary keys to ensure consistent ordering.
        Does NOT modify string values (no whitespace collapsing) or sort lists,
        as those can change the semantic meaning of code, queries, or
        order-sensitive arrays.
        """
        normalized = {}
        for key in sorted(arguments.keys()):
            value = arguments[key]
            if isinstance(value, dict):
                value = self._normalize_arguments(value)
            elif isinstance(value, list):
                # Do NOT sort lists - order matters for arrays
                # Recursively normalize dict elements within lists
                value = [
                    self._normalize_arguments(item) if isinstance(item, dict) else item
                    for item in value
                ]
            # Strings, numbers, bools, None are kept as-is
            normalized[key] = value
        return normalized

    def get_state_for_replay(self) -> dict[str, Any]:
        """Get state for serialization/resume."""
        return {
            "tool_call_history": [
                {"call_id": call_id, "tool": tool, "arguments": args}
                for call_id, tool, args in self._tool_call_history
            ],
            "pattern_counts": {
                f"{tool}|{args_hash}": count
                for (tool, args_hash), count in self._pattern_counts.items()
            },
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore state from serialization."""
        self._tool_call_history = [
            (item["call_id"], item["tool"], item["arguments"])
            for item in state.get("tool_call_history", [])
        ]
        # Rebuild pattern_counts from history to ensure consistency with window
        self._pattern_counts.clear()
        for _, tool_name, normalized_args in self._tool_call_history:
            sig_key = self._signature_key((tool_name, normalized_args))
            self._pattern_counts[sig_key] += 1


@dataclass
class ToolRetryGuard:
    """Tracks retry attempts for retryable tool failures.

    When a tool fails with a retryable error (timeout, backend failure,
    connection error), the engine can re-enter the tools phase to retry.
    This guard tracks retry counts per call_id and enforces the max retries limit.

    Unlike LoopGuard which prevents infinite loops of successful calls,
    this guard manages controlled retries for transient infrastructure failures.
    """

    config: ContinuationPolicyConfig = field(default_factory=ContinuationPolicyConfig)
    # Tracks retry count per call_id: {call_id: attempt_count}
    _retry_counts: dict[str, int] = field(default_factory=dict)
    # Tracks the last result for each call_id to avoid re-executing on resume
    _last_results: dict[str, Any] = field(default_factory=dict)

    @property
    def max_retries(self) -> int:
        """Get max retries from config."""
        return self.config.tool_retry_max_retries

    @property
    def base_delay(self) -> float:
        """Get base delay from config."""
        return self.config.tool_retry_base_delay

    @property
    def max_delay(self) -> float:
        """Get max delay from config."""
        return self.config.tool_retry_max_delay

    def record_retry(self, call_id: str) -> int:
        """Record a retry attempt for a call_id.

        Returns:
            The new retry count (1-indexed)
        """
        count = self._retry_counts.get(call_id, 0) + 1
        self._retry_counts[call_id] = count
        return count

    def get_retry_count(self, call_id: str) -> int:
        """Get current retry count for a call_id."""
        return self._retry_counts.get(call_id, 0)

    def can_retry(self, call_id: str) -> bool:
        """Check if a call_id can be retried."""
        return self.get_retry_count(call_id) < self.max_retries

    def store_result(self, call_id: str, result: Any) -> None:
        """Store the last result for a call_id (for resume)."""
        self._last_results[call_id] = result

    def get_last_result(self, call_id: str) -> Any | None:
        """Get the last stored result for a call_id."""
        return self._last_results.get(call_id)

    def clear_call(self, call_id: str) -> None:
        """Clear retry state for a call_id (after success or non-retryable failure)."""
        self._retry_counts.pop(call_id, None)
        self._last_results.pop(call_id, None)

    def reset(self) -> None:
        """Reset all retry state for new turn."""
        self._retry_counts.clear()
        self._last_results.clear()

    def get_state_for_replay(self) -> dict[str, Any]:
        """Get state for serialization/resume."""
        return {
            "retry_counts": self._retry_counts,
            "last_results": self._last_results,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore state from serialization."""
        self._retry_counts = state.get("retry_counts", {})
        self._last_results = state.get("last_results", {})


@dataclass
class TokenBudget:
    """Manages token budget for a turn/session.

    Tracks input and output token usage across frames and enforces
    separate limits for each.
    """

    config: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    _input_tokens_used: int = field(default=0, init=False)
    _output_tokens_used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._input_tokens_used = 0
        self._output_tokens_used = 0

    def record_tokens(
        self, input_tokens: int = 0, output_tokens: int = 0
    ) -> LimitCheckResult:
        """Record token usage.

        Args:
            input_tokens: Number of input (prompt) tokens used
            output_tokens: Number of output (completion + reasoning) tokens used

        Returns:
            LimitCheckResult indicating if budget was exceeded
        """
        self._input_tokens_used += input_tokens
        self._output_tokens_used += output_tokens

        if self._input_tokens_used > self.config.input_max_tokens:
            return LimitCheckResult.violated(
                reason=LimitViolationReason.TOKEN_BUDGET_EXCEEDED,
                detail=(
                    f"Input token budget exceeded: "
                    f"{self._input_tokens_used} > {self.config.input_max_tokens}"
                ),
                metadata={
                    "input_tokens_used": self._input_tokens_used,
                    "output_tokens_used": self._output_tokens_used,
                    "input_max_tokens": self.config.input_max_tokens,
                    "output_max_tokens": self.config.output_max_tokens,
                    "input_percentage": int(
                        self._input_tokens_used / self.config.input_max_tokens * 100
                    )
                    if self.config.input_max_tokens > 0
                    else 0,
                    "output_percentage": int(
                        self._output_tokens_used / self.config.output_max_tokens * 100
                    )
                    if self.config.output_max_tokens > 0
                    else 0,
                },
            )

        if self._output_tokens_used > self.config.output_max_tokens:
            return LimitCheckResult.violated(
                reason=LimitViolationReason.TOKEN_BUDGET_EXCEEDED,
                detail=(
                    f"Output token budget exceeded: "
                    f"{self._output_tokens_used} > {self.config.output_max_tokens}"
                ),
                metadata={
                    "input_tokens_used": self._input_tokens_used,
                    "output_tokens_used": self._output_tokens_used,
                    "input_max_tokens": self.config.input_max_tokens,
                    "output_max_tokens": self.config.output_max_tokens,
                    "input_percentage": int(
                        self._input_tokens_used / self.config.input_max_tokens * 100
                    )
                    if self.config.input_max_tokens > 0
                    else 0,
                    "output_percentage": int(
                        self._output_tokens_used / self.config.output_max_tokens * 100
                    )
                    if self.config.output_max_tokens > 0
                    else 0,
                },
            )

        # Log warning if approaching either limit
        if (
            self._input_tokens_used
            >= self.config.input_max_tokens * self.config.warn_threshold
        ) or (
            self._output_tokens_used
            >= self.config.output_max_tokens * self.config.warn_threshold
        ):
            logger.warning(
                "Token budget warning",
                input_tokens_used=self._input_tokens_used,
                output_tokens_used=self._output_tokens_used,
                input_max=self.config.input_max_tokens,
                output_max=self.config.output_max_tokens,
                input_percentage=int(
                    self._input_tokens_used / self.config.input_max_tokens * 100
                )
                if self.config.input_max_tokens > 0
                else 0,
                output_percentage=int(
                    self._output_tokens_used / self.config.output_max_tokens * 100
                )
                if self.config.output_max_tokens > 0
                else 0,
            )

        return LimitCheckResult.ok()

    def check_budget(
        self, input_additional: int = 0, output_additional: int = 0
    ) -> LimitCheckResult:
        """Check if additional tokens would exceed budget.

        Args:
            input_additional: Additional input tokens to check
            output_additional: Additional output tokens to check

        Returns:
            LimitCheckResult
        """
        if (
            self._input_tokens_used + input_additional <= self.config.input_max_tokens
            and self._output_tokens_used + output_additional
            <= self.config.output_max_tokens
        ):
            return LimitCheckResult.ok()

        return LimitCheckResult.violated(
            reason=LimitViolationReason.TOKEN_BUDGET_EXCEEDED,
            detail=(
                f"Token budget would be exceeded: "
                f"input {self._input_tokens_used + input_additional} > "
                f"{self.config.input_max_tokens} or "
                f"output {self._output_tokens_used + output_additional} > "
                f"{self.config.output_max_tokens}"
            ),
            metadata={
                "input_tokens_used": self._input_tokens_used,
                "output_tokens_used": self._output_tokens_used,
                "input_additional": input_additional,
                "output_additional": output_additional,
                "input_max_tokens": self.config.input_max_tokens,
                "output_max_tokens": self.config.output_max_tokens,
            },
        )

    @property
    def input_tokens_used(self) -> int:
        """Get input tokens used."""
        return self._input_tokens_used

    @property
    def output_tokens_used(self) -> int:
        """Get output tokens used."""
        return self._output_tokens_used

    @property
    def tokens_used(self) -> int:
        """Get total tokens used (input + output)."""
        return self._input_tokens_used + self._output_tokens_used

    @property
    def tokens_remaining(self) -> int:
        """Get remaining tokens (combined)."""
        return max(
            0,
            self.config.input_max_tokens - self._input_tokens_used,
        ) + max(
            0,
            self.config.output_max_tokens - self._output_tokens_used,
        )

    @property
    def usage_percentage(self) -> float:
        """Get usage as percentage (combined)."""
        total_max = self.config.input_max_tokens + self.config.output_max_tokens
        if total_max <= 0:
            return 0.0
        return self.tokens_used / total_max

    def reset(self) -> None:
        """Reset token budget for new turn."""
        self._input_tokens_used = 0
        self._output_tokens_used = 0

    def get_state_for_replay(self) -> dict[str, Any]:
        """Get state for serialization/resume."""
        return {
            "input_tokens_used": self._input_tokens_used,
            "output_tokens_used": self._output_tokens_used,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore state from serialization."""
        self._input_tokens_used = state.get("input_tokens_used", 0)
        self._output_tokens_used = state.get("output_tokens_used", 0)


@dataclass
class DeadlineTracker:
    """Tracks execution deadline for a turn/session.

    Enforces time limits to prevent runaway execution.
    """

    config: DeadlineTrackerConfig = field(default_factory=DeadlineTrackerConfig)
    _start_time: datetime = field(default_factory=lambda: datetime.now(UTC), init=False)
    _deadline: datetime = field(init=False)

    def __post_init__(self) -> None:
        self._start_time = datetime.now(UTC)
        self._deadline = self._start_time + timedelta(
            seconds=self.config.timeout_seconds
        )

    def check_deadline(self) -> LimitCheckResult:
        """Check if deadline has been exceeded.

        Returns:
            LimitCheckResult with structured violation info
        """
        now = datetime.now(UTC)
        seconds_remaining = (self._deadline - now).total_seconds()

        if seconds_remaining <= 0:
            return LimitCheckResult.violated(
                reason=LimitViolationReason.DEADLINE_EXCEEDED,
                detail=f"Execution deadline exceeded: {self.elapsed_seconds:.1f}s > {self.config.timeout_seconds}s",
                metadata={
                    "elapsed_seconds": self.elapsed_seconds,
                    "timeout_seconds": self.config.timeout_seconds,
                },
            )

        # Log warning if approaching deadline
        if seconds_remaining < self.config.timeout_seconds * self.config.warn_threshold:
            logger.warning(
                "Deadline approaching",
                seconds_remaining=int(seconds_remaining),
                timeout=self.config.timeout_seconds,
            )

        return LimitCheckResult.ok()

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since start."""
        return (datetime.now(UTC) - self._start_time).total_seconds()

    @property
    def seconds_remaining(self) -> float:
        """Get seconds left until the deadline (negative once exceeded)."""
        return (self._deadline - datetime.now(UTC)).total_seconds()

    @property
    def is_expired(self) -> bool:
        """Check if deadline has expired."""
        return self.check_deadline().within_limits is False

    def reset(self, timeout_seconds: int | None = None) -> None:
        """Reset deadline tracker.

        Args:
            timeout_seconds: Optional new timeout, uses existing if None
        """
        if timeout_seconds is not None:
            self.config.timeout_seconds = timeout_seconds
        self._start_time = datetime.now(UTC)
        self._deadline = self._start_time + timedelta(
            seconds=self.config.timeout_seconds
        )

    def get_state_for_replay(self) -> dict[str, Any]:
        """Get state for serialization/resume."""
        return {
            "start_time": self._start_time.isoformat(),
            "timeout_seconds": self.config.timeout_seconds,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore state from serialization."""
        if "start_time" in state:
            self._start_time = datetime.fromisoformat(state["start_time"])
        if "timeout_seconds" in state:
            self.config.timeout_seconds = state["timeout_seconds"]
        self._deadline = self._start_time + timedelta(
            seconds=self.config.timeout_seconds
        )


@dataclass
class RuntimeLimits:
    """Composite limits enforcement for frame execution.

    Combines token budget, deadline tracking, loop detection, and tool retry tracking
    into a single interface for the runtime engine.
    """

    token_budget: TokenBudget = field(default_factory=TokenBudget)
    deadline_tracker: DeadlineTracker = field(default_factory=DeadlineTracker)
    loop_guard: LoopGuard = field(default_factory=LoopGuard)
    tool_retry_guard: ToolRetryGuard = field(default_factory=ToolRetryGuard)

    def check_limits(
        self, input_tokens: int = 0, output_tokens: int = 0
    ) -> LimitCheckResult:
        """Check all limits.

        Args:
            input_tokens: Additional input tokens to check against budget
            output_tokens: Additional output tokens to check against budget

        Returns:
            LimitCheckResult with first violation found (if any)
        """
        # Check token budget
        budget_result = self.token_budget.check_budget(input_tokens, output_tokens)
        if not budget_result.within_limits:
            return budget_result

        # Check deadline
        deadline_result = self.deadline_tracker.check_deadline()
        if not deadline_result.within_limits:
            return deadline_result

        # Check loop detection
        loop_result = self.loop_guard.check_loop()
        if not loop_result.within_limits:
            return loop_result

        return LimitCheckResult.ok()

    def record_tool_call(
        self, call_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        """Record a tool call for loop detection."""
        self.loop_guard.record_tool_call(call_id, tool_name, arguments)

    def record_tool_retry(self, call_id: str) -> int:
        """Record a retry attempt for a tool call.

        Returns:
            The new retry count (1-indexed)
        """
        return self.tool_retry_guard.record_retry(call_id)

    def can_retry_tool(self, call_id: str) -> bool:
        """Check if a tool call can be retried."""
        return self.tool_retry_guard.can_retry(call_id)

    def get_tool_retry_count(self, call_id: str) -> int:
        """Get current retry count for a tool call."""
        return self.tool_retry_guard.get_retry_count(call_id)

    def store_tool_result(self, call_id: str, result: Any) -> None:
        """Store the last result for a tool call (for resume)."""
        self.tool_retry_guard.store_result(call_id, result)

    def get_last_tool_result(self, call_id: str) -> Any | None:
        """Get the last stored result for a tool call."""
        return self.tool_retry_guard.get_last_result(call_id)

    def clear_tool_retry(self, call_id: str) -> None:
        """Clear retry state for a tool call (after success or non-retryable failure)."""
        self.tool_retry_guard.clear_call(call_id)

    def record_tokens(
        self, input_tokens: int = 0, output_tokens: int = 0
    ) -> LimitCheckResult:
        """Record token usage."""
        return self.token_budget.record_tokens(input_tokens, output_tokens)

    def reset(self, timeout_seconds: int | None = None) -> None:
        """Reset all limits for new turn.

        Args:
            timeout_seconds: Optional new timeout for deadline tracker
        """
        self.token_budget.reset()
        self.loop_guard.reset()
        self.tool_retry_guard.reset()
        self.deadline_tracker.reset(timeout_seconds)

    @property
    def summary(self) -> dict[str, Any]:
        """Get summary of current limit state."""
        return {
            "tokens_used": self.token_budget.tokens_used,
            "tokens_remaining": self.token_budget.tokens_remaining,
            "usage_percentage": self.token_budget.usage_percentage,
            "elapsed_seconds": self.deadline_tracker.elapsed_seconds,
            "deadline_remaining": self.deadline_tracker.check_deadline().metadata.get(
                "seconds_remaining"
            ),
            "tool_call_history_length": len(self.loop_guard._tool_call_history),
        }

    def get_state_for_replay(self) -> dict[str, Any]:
        """Get combined state for serialization/resume."""
        return {
            "token_budget": self.token_budget.get_state_for_replay(),
            "deadline_tracker": self.deadline_tracker.get_state_for_replay(),
            "loop_guard": self.loop_guard.get_state_for_replay(),
            "tool_retry_guard": self.tool_retry_guard.get_state_for_replay(),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore state from serialization."""
        if "token_budget" in state:
            self.token_budget.restore_state(state["token_budget"])
        if "deadline_tracker" in state:
            self.deadline_tracker.restore_state(state["deadline_tracker"])
        if "loop_guard" in state:
            self.loop_guard.restore_state(state["loop_guard"])
        if "tool_retry_guard" in state:
            self.tool_retry_guard.restore_state(state["tool_retry_guard"])

    @classmethod
    def from_config(cls, limits_config: RuntimeLimitsConfig) -> RuntimeLimits:
        """Create fresh RuntimeLimits from config (for new turns).

        Args:
            limits_config: The RuntimeLimitsConfig with continuation policy settings

        Returns:
            Fresh RuntimeLimits ready for execution
        """
        # Convert dict to ContinuationPolicyConfig if needed
        policy_dict = limits_config.continuation_policy
        if isinstance(policy_dict, dict):
            policy_config = ContinuationPolicyConfig(**policy_dict)
        else:
            policy_config = policy_dict

        return cls(
            token_budget=TokenBudget(config=policy_config.token_budget),
            deadline_tracker=DeadlineTracker(config=policy_config.deadline),
            loop_guard=LoopGuard(config=policy_config.loop_detection),
            tool_retry_guard=ToolRetryGuard(config=policy_config),
        )

    @classmethod
    def from_restored(
        cls,
        limits_config: RuntimeLimitsConfig,
        limits_state: dict[str, Any] | None = None,
    ) -> RuntimeLimits:
        """Create RuntimeLimits from stored config and state for resume/replay.

        Args:
            limits_config: The RuntimeLimitsConfig with continuation policy settings
            limits_state: Optional state to restore (tokens_used, deadline start_time, loop history)

        Returns:
            Fully configured RuntimeLimits ready for execution
        """
        # Convert dict to ContinuationPolicyConfig if needed
        policy_dict = limits_config.continuation_policy
        if isinstance(policy_dict, dict):
            policy_config = ContinuationPolicyConfig(**policy_dict)
        else:
            policy_config = policy_dict

        limits = cls(
            token_budget=TokenBudget(config=policy_config.token_budget),
            deadline_tracker=DeadlineTracker(config=policy_config.deadline),
            loop_guard=LoopGuard(config=policy_config.loop_detection),
            tool_retry_guard=ToolRetryGuard(config=policy_config),
        )

        if limits_state:
            limits.restore_state(limits_state)

        return limits
