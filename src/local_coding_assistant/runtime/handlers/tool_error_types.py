"""
Tool error types and contracts for structured error handling.

This module defines the contracts and error types for tool failures
to enable proper retry logic and error classification.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ToolErrorType(str, Enum):
    """Classification of tool errors for retry logic."""

    NETWORK_ERROR = "network_error"  # Timeout, connection issues
    SYNTAX_ERROR = "syntax_error"  # Invalid tool call syntax
    TOOL_NOT_FOUND = "tool_not_found"  # Tool doesn't exist
    PERMISSION_ERROR = "permission_error"  # Access denied
    VALIDATION_ERROR = "validation_error"  # Invalid arguments
    RATE_LIMIT_ERROR = "rate_limit_error"  # Too many requests
    SERVICE_ERROR = "service_error"  # External service failure
    UNKNOWN_ERROR = "unknown_error"  # Unclassified error


class ToolRetryStrategy(str, Enum):
    """Retry strategy for tool errors."""

    RETRY_IMMEDIATE = "retry_immediate"  # Retry immediately
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # Retry with delay
    NO_RETRY = "no_retry"  # Don't retry
    ESCALATE = "escalate"  # Escalate to higher level


@dataclass
class ToolError:
    """Structured tool error information."""

    tool_name: str
    tool_args: dict[str, Any]
    error_type: ToolErrorType
    error_message: str
    retry_strategy: ToolRetryStrategy
    can_retry: bool
    original_exception: Exception | None = None


@dataclass
class ToolFailureInfo:
    """Complete information about a tool failure."""

    tool_name: str
    tool_args: dict[str, Any]
    error_message: str
    error_type: ToolErrorType
    retry_strategy: ToolRetryStrategy
    can_retry: bool
    attempt_count: int = 0
