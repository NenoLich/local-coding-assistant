"""
Unit tests for tool error types.
"""

import pytest

from local_coding_assistant.runtime.handlers.tool_error_types import (
    ToolError,
    ToolErrorType,
    ToolFailureInfo,
    ToolRetryStrategy,
)


class TestToolErrorType:
    """Unit tests for ToolErrorType enum."""

    def test_tool_error_type_values(self):
        """Test that ToolErrorType has expected values."""
        assert ToolErrorType.NETWORK_ERROR == "network_error"
        assert ToolErrorType.SYNTAX_ERROR == "syntax_error"
        assert ToolErrorType.TOOL_NOT_FOUND == "tool_not_found"
        assert ToolErrorType.PERMISSION_ERROR == "permission_error"
        assert ToolErrorType.VALIDATION_ERROR == "validation_error"
        assert ToolErrorType.RATE_LIMIT_ERROR == "rate_limit_error"
        assert ToolErrorType.SERVICE_ERROR == "service_error"
        assert ToolErrorType.UNKNOWN_ERROR == "unknown_error"

    def test_tool_error_type_all_values(self):
        """Test that all expected error types are present."""
        expected = {
            "network_error",
            "syntax_error",
            "tool_not_found",
            "permission_error",
            "validation_error",
            "rate_limit_error",
            "service_error",
            "unknown_error",
        }
        actual = {e.value for e in ToolErrorType}
        assert actual == expected


class TestToolRetryStrategy:
    """Unit tests for ToolRetryStrategy enum."""

    def test_tool_retry_strategy_values(self):
        """Test that ToolRetryStrategy has expected values."""
        assert ToolRetryStrategy.RETRY_IMMEDIATE == "retry_immediate"
        assert ToolRetryStrategy.RETRY_WITH_BACKOFF == "retry_with_backoff"
        assert ToolRetryStrategy.NO_RETRY == "no_retry"
        assert ToolRetryStrategy.ESCALATE == "escalate"

    def test_tool_retry_strategy_all_values(self):
        """Test that all expected retry strategies are present."""
        expected = {
            "retry_immediate",
            "retry_with_backoff",
            "no_retry",
            "escalate",
        }
        actual = {e.value for e in ToolRetryStrategy}
        assert actual == expected


class TestToolError:
    """Unit tests for ToolError dataclass."""

    def test_tool_error_creation(self):
        """Test creating a ToolError instance."""
        error = ToolError(
            tool_name="search_files",
            tool_args={"pattern": "*.py"},
            error_type=ToolErrorType.NETWORK_ERROR,
            error_message="Connection timeout",
            retry_strategy=ToolRetryStrategy.RETRY_WITH_BACKOFF,
            can_retry=True,
        )

        assert error.tool_name == "search_files"
        assert error.tool_args == {"pattern": "*.py"}
        assert error.error_type == ToolErrorType.NETWORK_ERROR
        assert error.error_message == "Connection timeout"
        assert error.retry_strategy == ToolRetryStrategy.RETRY_WITH_BACKOFF
        assert error.can_retry == True
        assert error.original_exception is None

    def test_tool_error_with_exception(self):
        """Test ToolError with original exception."""
        exc = ValueError("Invalid argument")
        error = ToolError(
            tool_name="read_file",
            tool_args={"file_path": "/etc/passwd"},
            error_type=ToolErrorType.PERMISSION_ERROR,
            error_message="Permission denied",
            retry_strategy=ToolRetryStrategy.NO_RETRY,
            can_retry=False,
            original_exception=exc,
        )

        assert error.original_exception == exc


class TestToolFailureInfo:
    """Unit tests for ToolFailureInfo dataclass."""

    def test_tool_failure_info_creation(self):
        """Test creating a ToolFailureInfo instance."""
        failure = ToolFailureInfo(
            tool_name="search_files",
            tool_args={"pattern": "*.py"},
            error_message="Connection timeout",
            error_type=ToolErrorType.NETWORK_ERROR,
            retry_strategy=ToolRetryStrategy.RETRY_WITH_BACKOFF,
            can_retry=True,
            attempt_count=1,
        )

        assert failure.tool_name == "search_files"
        assert failure.tool_args == {"pattern": "*.py"}
        assert failure.error_message == "Connection timeout"
        assert failure.error_type == ToolErrorType.NETWORK_ERROR
        assert failure.retry_strategy == ToolRetryStrategy.RETRY_WITH_BACKOFF
        assert failure.can_retry == True
        assert failure.attempt_count == 1

    def test_tool_failure_info_defaults(self):
        """Test ToolFailureInfo default values."""
        failure = ToolFailureInfo(
            tool_name="test_tool",
            tool_args={},
            error_message="Error",
            error_type=ToolErrorType.UNKNOWN_ERROR,
            retry_strategy=ToolRetryStrategy.NO_RETRY,
            can_retry=False,
        )

        assert failure.attempt_count == 0
