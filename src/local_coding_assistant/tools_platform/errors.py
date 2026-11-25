"""Shared error types for the tools platform.

This module contains the core error enums and types used across
contracts, exceptions, and error mapping.
"""

from __future__ import annotations

from enum import Enum


class ToolErrorCode(str, Enum):
    """Stable error codes for tool execution failures."""

    INVALID_ARGUMENTS = "invalid_arguments"
    UNAVAILABLE_TOOL = "unavailable_tool"
    DENIED = "denied"
    APPROVAL_REQUIRED = "approval_required"
    PATH_VIOLATION = "path_violation"
    CONFLICT = "conflict"
    TIMEOUT = "timeout"
    OUTPUT_LIMIT = "output_limit"
    BACKEND_FAILURE = "backend_failure"
    PROCESS_FAILURE = "process_failure"
    INTERNAL_FAILURE = "internal_failure"
    INTERRUPTED = "interrupted"


class ToolErrorCategory(str, Enum):
    """High-level error categories for grouping and policy."""

    VALIDATION = "validation"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    CONCURRENCY = "concurrency"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


__all__ = ["ToolErrorCategory", "ToolErrorCode"]
