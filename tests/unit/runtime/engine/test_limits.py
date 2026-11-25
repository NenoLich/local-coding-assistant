"""Unit tests for runtime limits enforcement."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

from local_coding_assistant.config.schemas import (
    DeadlineTrackerConfig,
    LoopDetectionConfig,
    TokenBudgetConfig,
)
from local_coding_assistant.runtime.engine.config import (
    LimitViolationReason,
)
from local_coding_assistant.runtime.engine.limits import (
    DeadlineTracker,
    LoopGuard,
    RuntimeLimits,
    TokenBudget,
)


class TestTokenBudget:
    """Tests for TokenBudget."""

    def test_initial_state(self) -> None:
        budget = TokenBudget(
            config=TokenBudgetConfig(input_max_tokens=1000, output_max_tokens=1000)
        )
        assert budget.tokens_used == 0
        assert budget.tokens_remaining == 2000
        assert budget.usage_percentage == 0.0

    def test_record_tokens(self) -> None:
        budget = TokenBudget(
            config=TokenBudgetConfig(input_max_tokens=1000, output_max_tokens=1000)
        )
        result = budget.record_tokens(input_tokens=100)
        assert result.within_limits is True
        assert budget.tokens_used == 100
        assert budget.tokens_remaining == 1900
        assert budget.usage_percentage == 0.05

    def test_record_tokens_exceeds_budget(self) -> None:
        budget = TokenBudget(
            config=TokenBudgetConfig(input_max_tokens=1000, output_max_tokens=1000)
        )
        budget.record_tokens(input_tokens=900)
        result = budget.record_tokens(input_tokens=200)
        assert result.within_limits is False
        assert result.reason == LimitViolationReason.TOKEN_BUDGET_EXCEEDED

    def test_check_budget(self) -> None:
        budget = TokenBudget(
            config=TokenBudgetConfig(input_max_tokens=1000, output_max_tokens=1000)
        )
        result = budget.check_budget(input_additional=500)
        assert result.within_limits is True
        result = budget.check_budget(input_additional=1000)
        assert result.within_limits is True
        result = budget.check_budget(input_additional=1001)
        assert result.within_limits is False
        assert result.reason == LimitViolationReason.TOKEN_BUDGET_EXCEEDED

    def test_check_budget_with_existing_usage(self) -> None:
        budget = TokenBudget(
            config=TokenBudgetConfig(input_max_tokens=1000, output_max_tokens=1000)
        )
        budget.record_tokens(input_tokens=500)
        result = budget.check_budget(input_additional=400)
        assert result.within_limits is True
        result = budget.check_budget(input_additional=500)
        assert result.within_limits is True
        result = budget.check_budget(input_additional=501)
        assert result.within_limits is False
        assert result.reason == LimitViolationReason.TOKEN_BUDGET_EXCEEDED

    def test_reset(self) -> None:
        budget = TokenBudget(
            config=TokenBudgetConfig(input_max_tokens=1000, output_max_tokens=1000)
        )
        budget.record_tokens(input_tokens=500)
        budget.reset()
        assert budget.tokens_used == 0
        assert budget.tokens_remaining == 2000

    def test_warning_threshold(self) -> None:
        budget = TokenBudget(
            config=TokenBudgetConfig(
                input_max_tokens=1000, output_max_tokens=1000, warn_threshold=0.8
            )
        )
        budget.record_tokens(input_tokens=700)  # 70% - no warning
        budget.record_tokens(input_tokens=100)  # 80% - warning logged


class TestDeadlineTracker:
    """Tests for DeadlineTracker."""

    def test_initial_state(self) -> None:
        tracker = DeadlineTracker(config=DeadlineTrackerConfig(timeout_seconds=60))
        assert tracker.config.timeout_seconds == 60
        assert tracker.elapsed_seconds >= 0
        assert not tracker.is_expired

    def test_check_deadline_not_expired(self) -> None:
        tracker = DeadlineTracker(config=DeadlineTrackerConfig(timeout_seconds=60))
        result = tracker.check_deadline()
        assert result.within_limits is True
        assert result.reason is None
        assert result.detail is None

    def test_check_deadline_expired(self) -> None:
        tracker = DeadlineTracker(config=DeadlineTrackerConfig(timeout_seconds=1))
        # Manually expire by setting start time in the past
        tracker._start_time = datetime.now(UTC) - timedelta(seconds=2)
        tracker._deadline = tracker._start_time + timedelta(seconds=1)
        result = tracker.check_deadline()
        assert result.within_limits is False
        assert result.reason == LimitViolationReason.DEADLINE_EXCEEDED

    def test_elapsed_seconds(self) -> None:
        tracker = DeadlineTracker(config=DeadlineTrackerConfig(timeout_seconds=60))
        time.sleep(0.1)
        assert tracker.elapsed_seconds >= 0.1

    def test_is_expired(self) -> None:
        tracker = DeadlineTracker(config=DeadlineTrackerConfig(timeout_seconds=1))
        tracker._start_time = datetime.now(UTC) - timedelta(seconds=2)
        tracker._deadline = tracker._start_time + timedelta(seconds=1)
        assert tracker.is_expired

    def test_reset(self) -> None:
        tracker = DeadlineTracker(config=DeadlineTrackerConfig(timeout_seconds=60))
        time.sleep(0.1)
        tracker.reset(timeout_seconds=30)
        assert tracker.config.timeout_seconds == 30
        assert tracker.elapsed_seconds < 1.0


class TestLoopGuard:
    """Tests for LoopGuard."""

    def test_initial_state(self) -> None:
        config = LoopDetectionConfig(max_repetitions=3, window_size=5)
        guard = LoopGuard(config=config)
        assert len(guard._tool_call_history) == 0
        assert len(guard._pattern_counts) == 0

    def test_record_tool_call(self) -> None:
        guard = LoopGuard()
        guard.record_tool_call("432", "read_file", {"path": "/tmp/test.txt"})
        assert len(guard._tool_call_history) == 1
        # pattern_counts is keyed by signature (tool+args hash), not just tool name
        assert len(guard._pattern_counts) == 1
        # Get the count for the recorded signature
        sig_key = next(iter(guard._pattern_counts))
        assert guard._pattern_counts[sig_key] == 1

    def test_check_loop_no_loop(self) -> None:
        guard = LoopGuard()
        guard.record_tool_call("123", "read_file", {"path": "/tmp/test.txt"})
        guard.record_tool_call("432", "write_file", {"path": "/tmp/test.txt"})
        result = guard.check_loop()
        assert result.within_limits is True

    def test_check_loop_identical_consecutive(self) -> None:
        guard = LoopGuard()
        guard.record_tool_call("123", "read_file", {"path": "/tmp/test.txt"})
        guard.record_tool_call("234", "read_file", {"path": "/tmp/test.txt"})
        result = guard.check_loop()
        assert result.within_limits is False
        assert result.reason == LimitViolationReason.LOOP_DETECTED
        assert result.metadata["detection_mode"] == "consecutive"

    def test_check_loop_excessive_repetitions(self) -> None:
        config = LoopDetectionConfig(max_repetitions=3)
        guard = LoopGuard(config=config)
        guard.record_tool_call("123", "read_file", {"path": "/tmp/test.txt"})
        guard.record_tool_call("546", "write_file", {"path": "/tmp/test2.txt"})
        guard.record_tool_call("789", "read_file", {"path": "/tmp/test.txt"})
        guard.record_tool_call("763", "write_file", {"path": "/tmp/test2.txt"})
        guard.record_tool_call("987", "read_file", {"path": "/tmp/test.txt"})
        result = guard.check_loop()
        assert result.within_limits is False
        assert result.reason == LimitViolationReason.LOOP_DETECTED
        assert result.metadata["detection_mode"] == "window"
        assert result.metadata["count"] == 3

    def test_check_loop_disabled(self) -> None:
        config = LoopDetectionConfig(enabled=False)
        guard = LoopGuard(config=config)
        guard.record_tool_call("123", "read_file", {"path": "/tmp/test.txt"})
        guard.record_tool_call("456", "read_file", {"path": "/tmp/test.txt"})
        result = guard.check_loop()
        assert result.within_limits is True

    def test_window_size(self) -> None:
        config = LoopDetectionConfig(window_size=2)
        guard = LoopGuard(config=config)
        guard.record_tool_call("123", "read_file", {"path": "/tmp/test.txt"})
        guard.record_tool_call("234", "write_file", {"path": "/tmp/test.txt"})
        guard.record_tool_call("345", "read_file", {"path": "/tmp/test.txt"})
        # History should be limited to window_size
        assert len(guard._tool_call_history) == 2

    def test_normalize_arguments_preserves_whitespace_and_list_order(self) -> None:
        """Arguments are normalized by sorting keys only.
        Whitespace in strings and list order are preserved to avoid false positives.
        """
        guard = LoopGuard()
        # Different whitespace should be treated as different calls
        args1 = {"path": "  /tmp/test.txt  "}
        args2 = {"path": "/tmp/test.txt"}
        guard.record_tool_call("123", "read_file", args1)
        guard.record_tool_call("345", "read_file", args2)
        # Should NOT detect as identical (whitespace preserved)
        result = guard.check_loop()
        assert result.within_limits is True

        # List order matters - different order = different signature
        guard2 = LoopGuard()
        args3 = {"files": ["a.txt", "b.txt"]}
        args4 = {"files": ["b.txt", "a.txt"]}
        guard2.record_tool_call("543", "read_files", args3)
        guard2.record_tool_call("953", "read_files", args4)
        result2 = guard2.check_loop()
        assert result2.within_limits is True

    def test_same_signature_detected_as_loop(self) -> None:
        """Identical signatures (same tool + same args) are detected as loops."""
        guard = LoopGuard()
        args = {"path": "/tmp/test.txt"}
        guard.record_tool_call("123", "read_file", args)
        guard.record_tool_call("534", "read_file", args)
        result = guard.check_loop()
        assert result.within_limits is False
        assert result.reason == LimitViolationReason.LOOP_DETECTED

    def test_same_signature_and_same_call_id_not_detected_as_loop(self) -> None:
        """Identical signatures and call id treated as the same tool call and ignored by loop guard"""
        guard = LoopGuard()
        args = {"path": "/tmp/test.txt"}
        guard.record_tool_call("123", "read_file", args)
        guard.record_tool_call("123", "read_file", args)
        result = guard.check_loop()
        assert result.within_limits is True

    def test_window_eviction_decrements_count(self) -> None:
        """When entries fall out of the window, their pattern count is decremented."""
        config = LoopDetectionConfig(
            window_size=2, max_repetitions=2, check_consecutive=False
        )
        guard = LoopGuard(config=config)
        args = {"path": "/tmp/test.txt"}

        # First call
        guard.record_tool_call("123", "read_file", args)
        assert guard._pattern_counts[next(iter(guard._pattern_counts))] == 1

        # Second call - should trigger window repetition (count=2 >= max_repetitions=2)
        guard.record_tool_call("234", "read_file", args)
        result = guard.check_loop()
        assert result.within_limits is False
        assert result.metadata["detection_mode"] == "window"
        assert result.metadata["count"] == 2

        # Third call with different args - evicts first, count goes back to 1
        guard.record_tool_call("456", "read_file", {"path": "/tmp/other.txt"})
        result = guard.check_loop()
        # Should be within limits now (only 1 of each signature in window)
        assert result.within_limits is True

    def test_reset(self) -> None:
        guard = LoopGuard()
        guard.record_tool_call("123", "read_file", {"path": "/tmp/test.txt"})
        guard.reset()
        assert len(guard._tool_call_history) == 0
        assert len(guard._pattern_counts) == 0

    def test_get_state_for_replay_and_restore(self) -> None:
        """Loop guard state can be serialized and restored."""
        config = LoopDetectionConfig(max_repetitions=2)
        guard = LoopGuard(config=config)
        guard.record_tool_call("123", "read_file", {"path": "/tmp/test.txt"})
        guard.record_tool_call("321", "write_file", {"path": "/tmp/test2.txt"})

        state = guard.get_state_for_replay()
        assert "tool_call_history" in state
        assert "pattern_counts" in state
        assert len(state["tool_call_history"]) == 2

        # Restore to new guard
        guard2 = LoopGuard(config=config)
        guard2.restore_state(state)

        assert len(guard2._tool_call_history) == 2
        assert len(guard2._pattern_counts) == 2
        # Restored guard should detect loops correctly
        guard2.record_tool_call("765", "read_file", {"path": "/tmp/test.txt"})
        result = guard2.check_loop()
        assert result.within_limits is False


class TestRuntimeLimits:
    """Tests for RuntimeLimits composite."""

    def test_initial_state(self) -> None:
        limits = RuntimeLimits()
        assert limits.token_budget.config.input_max_tokens == 100_000
        assert limits.token_budget.config.output_max_tokens == 100_000
        assert limits.deadline_tracker.config.timeout_seconds == 300
        assert limits.loop_guard.config.enabled is True

    def test_check_limits_all_pass(self) -> None:
        limits = RuntimeLimits()
        result = limits.check_limits()
        assert result.within_limits is True

    def test_check_limits_token_budget_exceeded(self) -> None:
        limits = RuntimeLimits(
            token_budget=TokenBudget(
                config=TokenBudgetConfig(input_max_tokens=100, output_max_tokens=100)
            )
        )
        limits.token_budget.record_tokens(input_tokens=90)
        result = limits.check_limits(input_tokens=20)
        assert result.within_limits is False
        assert result.reason == LimitViolationReason.TOKEN_BUDGET_EXCEEDED

    def test_check_limits_deadline_exceeded(self) -> None:
        limits = RuntimeLimits(
            deadline_tracker=DeadlineTracker(
                config=DeadlineTrackerConfig(timeout_seconds=1)
            )
        )
        limits.deadline_tracker._start_time = datetime.now(UTC) - timedelta(seconds=2)
        limits.deadline_tracker._deadline = (
            limits.deadline_tracker._start_time + timedelta(seconds=1)
        )
        result = limits.check_limits()
        assert result.within_limits is False
        assert result.reason == LimitViolationReason.DEADLINE_EXCEEDED

    def test_check_limits_loop_detected(self) -> None:
        limits = RuntimeLimits()
        limits.loop_guard.record_tool_call(
            "123", "read_file", {"path": "/tmp/test.txt"}
        )
        limits.loop_guard.record_tool_call(
            "456", "read_file", {"path": "/tmp/test.txt"}
        )
        result = limits.check_limits()
        assert result.within_limits is False
        assert result.reason == LimitViolationReason.LOOP_DETECTED

    def test_record_tool_call(self) -> None:
        limits = RuntimeLimits()
        limits.record_tool_call("123", "read_file", {"path": "/tmp/test.txt"})
        # pattern_counts is keyed by signature, not tool name
        assert len(limits.loop_guard._pattern_counts) == 1
        sig_key = next(iter(limits.loop_guard._pattern_counts))
        assert limits.loop_guard._pattern_counts[sig_key] == 1

    def test_record_tokens(self) -> None:
        limits = RuntimeLimits()
        result = limits.record_tokens(input_tokens=100)
        assert result.within_limits is True
        assert limits.token_budget.tokens_used == 100

    def test_reset(self) -> None:
        limits = RuntimeLimits()
        limits.record_tokens(input_tokens=100)
        limits.record_tool_call("123", "read_file", {"path": "/tmp/test.txt"})
        limits.reset()
        assert limits.token_budget.tokens_used == 0
        assert len(limits.loop_guard._tool_call_history) == 0

    def test_summary(self) -> None:
        limits = RuntimeLimits()
        limits.record_tokens(input_tokens=1000)
        summary = limits.summary
        assert summary["tokens_used"] == 1000
        assert summary["tokens_remaining"] == 199_000
        assert summary["usage_percentage"] == 0.005
        assert "elapsed_seconds" in summary
        assert "deadline_remaining" in summary
