"""
Tool error classification utilities.

This module provides utilities for classifying tool errors and determining
retry strategies based on error patterns and types.
"""

from typing import Any

from local_coding_assistant.runtime.handlers.tool_error_types import (
    ToolError,
    ToolErrorType,
    ToolRetryStrategy,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.tool_error_classifier")


class ToolErrorClassifier:
    """Classifier for tool errors with retry strategy determination."""

    def classify_error(
        self, tool_name: str, tool_args: dict[str, Any], error: str | Exception
    ) -> ToolError:
        """Classify a tool error and determine retry strategy."""
        error_message = str(error) if isinstance(error, Exception) else error

        error_type = self._determine_error_type(error_message)
        retry_strategy = self._determine_retry_strategy(error_type, error_message)
        can_retry = retry_strategy in [
            ToolRetryStrategy.RETRY_IMMEDIATE,
            ToolRetryStrategy.RETRY_WITH_BACKOFF,
        ]

        return ToolError(
            tool_name=tool_name,
            tool_args=tool_args,
            error_type=error_type,
            error_message=error_message,
            retry_strategy=retry_strategy,
            can_retry=can_retry,
            original_exception=error if isinstance(error, Exception) else None,
        )

    def _determine_error_type(self, error_message: str) -> ToolErrorType:
        """Determine the type of error based on message patterns."""
        error_lower = error_message.lower()

        # Network/timeout errors
        if any(
            pattern in error_lower
            for pattern in [
                "timeout",
                "connection",
                "network",
                "unreachable",
                "dns",
                "socket",
                "http error",
                "service unavailable",
            ]
        ):
            return ToolErrorType.NETWORK_ERROR

        # Rate limiting
        if any(
            pattern in error_lower
            for pattern in [
                "rate limit",
                "too many requests",
                "quota exceeded",
                "throttled",
                "429",
            ]
        ):
            return ToolErrorType.RATE_LIMIT_ERROR

        # Permission/access errors
        if any(
            pattern in error_lower
            for pattern in [
                "permission denied",
                "access denied",
                "unauthorized",
                "forbidden",
                "403",
                "401",
            ]
        ):
            return ToolErrorType.PERMISSION_ERROR

        # Tool not found errors
        if any(
            pattern in error_lower
            for pattern in [
                "tool not found",
                "unknown tool",
                "function not found",
                "tool does not exist",
                "invalid tool name",
                "not exposed",
            ]
        ):
            return ToolErrorType.TOOL_NOT_FOUND

        # Syntax/validation errors
        if any(
            pattern in error_lower
            for pattern in [
                "invalid arguments",
                "validation error",
                "syntax error",
                "malformed",
                "invalid json",
                "parse error",
                "missing required",
            ]
        ):
            return ToolErrorType.VALIDATION_ERROR

        # Service errors
        if any(
            pattern in error_lower
            for pattern in [
                "internal server error",
                "service error",
                "500",
                "database error",
                "api error",
            ]
        ):
            return ToolErrorType.SERVICE_ERROR

        return ToolErrorType.UNKNOWN_ERROR

    def _determine_retry_strategy(
        self, error_type: ToolErrorType, error_message: str
    ) -> ToolRetryStrategy:
        """Determine retry strategy based on error type."""
        strategies = {
            ToolErrorType.NETWORK_ERROR: ToolRetryStrategy.RETRY_WITH_BACKOFF,
            ToolErrorType.RATE_LIMIT_ERROR: ToolRetryStrategy.RETRY_WITH_BACKOFF,
            ToolErrorType.PERMISSION_ERROR: ToolRetryStrategy.NO_RETRY,
            ToolErrorType.TOOL_NOT_FOUND: ToolRetryStrategy.RETRY_IMMEDIATE,
            ToolErrorType.VALIDATION_ERROR: ToolRetryStrategy.RETRY_IMMEDIATE,
            ToolErrorType.SERVICE_ERROR: ToolRetryStrategy.RETRY_WITH_BACKOFF,
            ToolErrorType.UNKNOWN_ERROR: ToolRetryStrategy.RETRY_IMMEDIATE,
        }

        return strategies.get(error_type, ToolRetryStrategy.NO_RETRY)

    def create_failure_info(
        self, tool_error: ToolError, attempt_count: int = 0
    ) -> dict[str, Any]:
        """Create failure info dict for handler context."""
        return {
            "tool_name": tool_error.tool_name,
            "tool_args": tool_error.tool_args,
            "error_message": tool_error.error_message,
            "error_type": tool_error.error_type,
            "retry_strategy": tool_error.retry_strategy,
            "can_retry": tool_error.can_retry,
            "attempt_count": attempt_count,
        }
