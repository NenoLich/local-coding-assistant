"""
Unit tests for ToolErrorClassifier.
"""

import pytest

from local_coding_assistant.runtime.handlers.tool_error_classifier import (
    ToolErrorClassifier,
)
from local_coding_assistant.runtime.handlers.tool_error_types import (
    ToolError,
    ToolErrorType,
    ToolRetryStrategy,
)


class TestToolErrorClassifier:
    """Unit tests for ToolErrorClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create a ToolErrorClassifier instance."""
        return ToolErrorClassifier()

    def test_classify_network_error(self, classifier):
        """Test classification of network errors."""
        error = classifier.classify_error(
            "search_files", {"pattern": "*.py"}, "Connection timeout after 30 seconds"
        )

        assert error.tool_name == "search_files"
        assert error.tool_args == {"pattern": "*.py"}
        assert error.error_type == ToolErrorType.NETWORK_ERROR
        assert error.retry_strategy == ToolRetryStrategy.RETRY_WITH_BACKOFF
        assert error.can_retry == True

    def test_classify_rate_limit_error(self, classifier):
        """Test classification of rate limit errors."""
        error = classifier.classify_error(
            "read_file",
            {"file_path": "test.py"},
            "Rate limit exceeded, try again later",
        )

        assert error.error_type == ToolErrorType.RATE_LIMIT_ERROR
        assert error.retry_strategy == ToolRetryStrategy.RETRY_WITH_BACKOFF
        assert error.can_retry == True

    def test_classify_permission_error(self, classifier):
        """Test classification of permission errors."""
        error = classifier.classify_error(
            "read_file",
            {"file_path": "/etc/passwd"},
            "Permission denied: cannot access file",
        )

        assert error.error_type == ToolErrorType.PERMISSION_ERROR
        assert error.retry_strategy == ToolRetryStrategy.NO_RETRY
        assert error.can_retry == False

    def test_classify_validation_error(self, classifier):
        """Test classification of validation errors."""
        error = classifier.classify_error(
            "search_files",
            {"pattern": 123},
            "Invalid arguments: 'pattern' must be a string",
        )

        assert error.error_type == ToolErrorType.VALIDATION_ERROR
        assert error.retry_strategy == ToolRetryStrategy.RETRY_IMMEDIATE
        assert error.can_retry == True

    def test_classify_service_error(self, classifier):
        """Test classification of service errors."""
        error = classifier.classify_error(
            "api_call", {"endpoint": "/data"}, "Internal server error: 500"
        )

        assert error.error_type == ToolErrorType.SERVICE_ERROR
        assert error.retry_strategy == ToolRetryStrategy.RETRY_WITH_BACKOFF
        assert error.can_retry == True

    def test_classify_unknown_error(self, classifier):
        """Test classification of unknown errors."""
        error = classifier.classify_error(
            "unknown_tool", {}, "Some unexpected error occurred"
        )

        assert error.error_type == ToolErrorType.UNKNOWN_ERROR
        assert error.retry_strategy == ToolRetryStrategy.RETRY_IMMEDIATE
        assert error.can_retry == True

    def test_classify_with_exception(self, classifier):
        """Test classification when error is an exception."""
        exc = ValueError("Invalid value")
        error = classifier.classify_error("test_tool", {}, exc)

        assert error.error_message == "Invalid value"
        assert error.original_exception == exc
        assert (
            error.error_type == ToolErrorType.UNKNOWN_ERROR
        )  # "Invalid value" doesn't match validation patterns

    def test_classify_case_insensitive(self, classifier):
        """Test that classification is case insensitive."""
        error = classifier.classify_error("search_files", {}, "TIMEOUT occurred")

        assert error.error_type == ToolErrorType.NETWORK_ERROR

    def test_create_failure_info(self, classifier):
        """Test creating failure info dict."""
        tool_error = ToolError(
            tool_name="search_files",
            tool_args={"pattern": "*.py"},
            error_type=ToolErrorType.NETWORK_ERROR,
            error_message="Connection timeout",
            retry_strategy=ToolRetryStrategy.RETRY_WITH_BACKOFF,
            can_retry=True,
        )

        failure_info = classifier.create_failure_info(tool_error, attempt_count=2)

        expected = {
            "tool_name": "search_files",
            "tool_args": {"pattern": "*.py"},
            "error_message": "Connection timeout",
            "error_type": ToolErrorType.NETWORK_ERROR,
            "retry_strategy": ToolRetryStrategy.RETRY_WITH_BACKOFF,
            "can_retry": True,
            "attempt_count": 2,
        }

        assert failure_info == expected

    def test_create_failure_info_default_attempt_count(self, classifier):
        """Test create_failure_info with default attempt_count."""
        tool_error = ToolError(
            tool_name="test_tool",
            tool_args={},
            error_type=ToolErrorType.UNKNOWN_ERROR,
            error_message="Error",
            retry_strategy=ToolRetryStrategy.NO_RETRY,
            can_retry=False,
        )

        failure_info = classifier.create_failure_info(tool_error)

        assert failure_info["attempt_count"] == 0

    @pytest.mark.parametrize(
        "error_message,expected_type",
        [
            ("Connection timeout", ToolErrorType.NETWORK_ERROR),
            ("DNS resolution failed", ToolErrorType.NETWORK_ERROR),
            ("Rate limit exceeded", ToolErrorType.RATE_LIMIT_ERROR),
            ("Too many requests", ToolErrorType.RATE_LIMIT_ERROR),
            ("Permission denied", ToolErrorType.PERMISSION_ERROR),
            ("Access forbidden", ToolErrorType.PERMISSION_ERROR),
            ("Tool not found", ToolErrorType.TOOL_NOT_FOUND),
            ("Function not exposed", ToolErrorType.TOOL_NOT_FOUND),
            ("Invalid arguments", ToolErrorType.VALIDATION_ERROR),
            ("Syntax error in JSON", ToolErrorType.VALIDATION_ERROR),
            ("Internal server error", ToolErrorType.SERVICE_ERROR),
            ("API error occurred", ToolErrorType.SERVICE_ERROR),
            ("Random error message", ToolErrorType.UNKNOWN_ERROR),
            (
                "Unknown function",
                ToolErrorType.UNKNOWN_ERROR,
            ),  # "unknown" triggers UNKNOWN_ERROR
            (
                "Database connection failed",
                ToolErrorType.NETWORK_ERROR,
            ),  # "connection" triggers NETWORK_ERROR
        ],
    )
    def test_error_type_classification(self, classifier, error_message, expected_type):
        """Test various error messages map to correct types."""
        error = classifier.classify_error("test_tool", {}, error_message)
        assert error.error_type == expected_type

    @pytest.mark.parametrize(
        "error_type,expected_strategy",
        [
            (ToolErrorType.NETWORK_ERROR, ToolRetryStrategy.RETRY_WITH_BACKOFF),
            (ToolErrorType.RATE_LIMIT_ERROR, ToolRetryStrategy.RETRY_WITH_BACKOFF),
            (ToolErrorType.PERMISSION_ERROR, ToolRetryStrategy.NO_RETRY),
            (ToolErrorType.TOOL_NOT_FOUND, ToolRetryStrategy.RETRY_IMMEDIATE),
            (ToolErrorType.VALIDATION_ERROR, ToolRetryStrategy.RETRY_IMMEDIATE),
            (ToolErrorType.SERVICE_ERROR, ToolRetryStrategy.RETRY_WITH_BACKOFF),
            (ToolErrorType.UNKNOWN_ERROR, ToolRetryStrategy.RETRY_IMMEDIATE),
        ],
    )
    def test_retry_strategy_mapping(self, classifier, error_type, expected_strategy):
        """Test that error types map to correct retry strategies."""
        # Create a mock error to test the strategy mapping
        error = ToolError(
            tool_name="test",
            tool_args={},
            error_type=error_type,
            error_message="test",
            retry_strategy=ToolRetryStrategy.NO_RETRY,  # Will be overridden
            can_retry=False,
        )

        # Test the internal method
        strategy = classifier._determine_retry_strategy(error_type, "test message")
        assert strategy == expected_strategy
