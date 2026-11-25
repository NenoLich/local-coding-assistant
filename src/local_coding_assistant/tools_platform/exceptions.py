"""Exception taxonomy for the tools platform."""

from __future__ import annotations

from typing import Any

from local_coding_assistant.core.exceptions import LocalAssistantError
from local_coding_assistant.tools_platform.contracts import ToolExecutionError
from local_coding_assistant.tools_platform.errors import (
    ToolErrorCategory,
    ToolErrorCode,
)


def _normalize_code(value: ToolErrorCode | str | None) -> ToolErrorCode | None:
    if value is None:
        return None
    if isinstance(value, ToolErrorCode):
        return value
    try:
        return ToolErrorCode(value)
    except ValueError:
        return None


def _normalize_category(
    value: ToolErrorCategory | str | None,
) -> ToolErrorCategory | None:
    if value is None:
        return None
    if isinstance(value, ToolErrorCategory):
        return value
    try:
        return ToolErrorCategory(value)
    except ValueError:
        return None


class ToolPlatformError(LocalAssistantError):
    """Base exception for tools platform failures."""

    subsystem = "tools_platform"

    def __init__(
        self,
        message: str,
        *,
        code: ToolErrorCode | str | None = None,
        category: ToolErrorCategory | str | None = None,
        retryable: bool = False,
        remediation: str | None = None,
        details: dict[str, Any] | None = None,
        causal_error_type: str | None = None,
        tool_id: str | None = None,
        call_id: str | None = None,
        backend: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        super().__init__(message, subsystem=self.subsystem)
        self.code = _normalize_code(code) or ToolErrorCode.INTERNAL_FAILURE
        self.category = _normalize_category(category) or ToolErrorCategory.UNKNOWN
        self.retryable = retryable
        self.remediation = remediation
        self.details = details or {}
        self.causal_error_type = causal_error_type
        self.tool_id = tool_id
        self.call_id = call_id
        self.backend = backend
        self.workspace_id = workspace_id

    def to_tool_error(self) -> ToolExecutionError:
        """Convert platform error into the canonical external contract."""
        message = str(self).removeprefix(f"[{self.subsystem}] ")
        details = dict(self.details)
        if self.tool_id is not None:
            details.setdefault("tool_id", self.tool_id)
        if self.call_id is not None:
            details.setdefault("call_id", self.call_id)
        if self.backend is not None:
            details.setdefault("backend", self.backend)
        if self.workspace_id is not None:
            details.setdefault("workspace_id", self.workspace_id)
        return ToolExecutionError(
            code=self.code,
            category=self.category,
            message=message,
            retryable=self.retryable,
            remediation=self.remediation,
            details=details,
            causal_error_type=self.causal_error_type,
        )


class ToolValidationError(ToolPlatformError):
    """Raised for invalid tool arguments or malformed inputs.

    Subclasses (e.g. ToolPathViolationError) may override the default
    error code by passing ``code`` in ``kwargs``.
    """

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("code", ToolErrorCode.INVALID_ARGUMENTS)
        kwargs.setdefault("category", ToolErrorCategory.VALIDATION)
        super().__init__(message, **kwargs)


class ToolAuthorizationError(ToolPlatformError):
    """Raised when a tool invocation is denied or approval is required."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            code=ToolErrorCode.DENIED,
            category=ToolErrorCategory.AUTHORIZATION,
            **kwargs,
        )


class ToolPathViolationError(ToolValidationError):
    """Raised when a requested path leaves the workspace boundary."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            code=ToolErrorCode.PATH_VIOLATION,
            **kwargs,
        )


class ToolConflictError(ToolPlatformError):
    """Raised when a concurrent change or lock conflict occurs."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            code=ToolErrorCode.CONFLICT,
            category=ToolErrorCategory.CONCURRENCY,
            **kwargs,
        )


class ToolTimeoutError(ToolPlatformError):
    """Raised when an execution or I/O timeout is hit."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            code=ToolErrorCode.TIMEOUT,
            category=ToolErrorCategory.RESOURCE,
            retryable=True,
            **kwargs,
        )


class ToolBackendError(ToolPlatformError):
    """Raised when a backend fails or cannot be reached."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            code=ToolErrorCode.BACKEND_FAILURE,
            category=ToolErrorCategory.INFRASTRUCTURE,
            retryable=True,
            **kwargs,
        )


class ToolProcessError(ToolPlatformError):
    """Raised when a process lifecycle operation fails."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            code=ToolErrorCode.PROCESS_FAILURE,
            category=ToolErrorCategory.INFRASTRUCTURE,
            **kwargs,
        )


class ToolInternalError(ToolPlatformError):
    """Raised for unexpected platform errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            code=ToolErrorCode.INTERNAL_FAILURE,
            category=ToolErrorCategory.UNKNOWN,
            **kwargs,
        )


class ToolInterruptedError(ToolPlatformError):
    """Raised when tool execution is cancelled by a cooperative interrupt.

    Carries no retryability flag: the caller (turn interruption) decides
    whether the call is re-executed later, not the tool platform.
    """

    def __init__(
        self, message: str = "Tool execution interrupted", **kwargs: Any
    ) -> None:
        super().__init__(
            message,
            code=ToolErrorCode.INTERRUPTED,
            category=ToolErrorCategory.CONCURRENCY,
            retryable=False,
            **kwargs,
        )


__all__ = [
    "ToolAuthorizationError",
    "ToolBackendError",
    "ToolConflictError",
    "ToolInternalError",
    "ToolInterruptedError",
    "ToolPathViolationError",
    "ToolPlatformError",
    "ToolProcessError",
    "ToolTimeoutError",
    "ToolValidationError",
]
