"""Exception-to-ToolExecutionError mapping for the tools platform."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from local_coding_assistant.tools_platform.contracts import ToolExecutionError
from local_coding_assistant.tools_platform.errors import (
    ToolErrorCategory,
    ToolErrorCode,
)
from local_coding_assistant.tools_platform.exceptions import ToolPlatformError


def _safe_message(exc: BaseException, fallback: str) -> str:
    message = str(exc).strip()
    if not message:
        return fallback
    # Strip pydantic's "For further information visit ..." footer so the
    # message is concise when surfaced back to the LLM.
    lines = message.splitlines()
    while lines and lines[-1].strip().startswith(
        "For further information visit https://errors.pydantic.dev/"
    ):
        lines.pop()
    message = "\n".join(lines).strip()
    if not message:
        return fallback
    if message.startswith("[") and "] " in message:
        parts = message.split("] ", 1)
        if len(parts) == 2 and parts[0].startswith("["):
            return parts[1]
    return message


def _normalize_details(details: dict[str, Any] | None) -> dict[str, Any]:
    if details is None:
        return {}
    return dict(details)


def _with_context(
    *,
    tool_id: str | None,
    call_id: str | None,
    backend: str | None,
    workspace_id: str | None,
    details: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if tool_id is not None:
        payload["tool_id"] = tool_id
    if call_id is not None:
        payload["call_id"] = call_id
    if backend is not None:
        payload["backend"] = backend
    if workspace_id is not None:
        payload["workspace_id"] = workspace_id
    if details:
        payload.update(details)
    return _normalize_details(payload)


def map_exception(
    exc: Exception,
    *,
    tool_id: str | None = None,
    call_id: str | None = None,
    backend: str | None = None,
    workspace_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> ToolExecutionError:
    """Map a runtime exception to the canonical ToolExecutionError contract."""
    if isinstance(exc, ToolExecutionError):
        return exc

    if isinstance(exc, ToolPlatformError):
        tool_error = exc.to_tool_error()
        payload = _with_context(
            tool_id=tool_id,
            call_id=call_id,
            backend=backend,
            workspace_id=workspace_id,
            details=details,
        )
        tool_error.details.update(payload)
        return tool_error

    ctx = _with_context(
        tool_id=tool_id,
        call_id=call_id,
        backend=backend,
        workspace_id=workspace_id,
        details=details,
    )

    if isinstance(exc, (ValidationError, TypeError, ValueError)):
        return ToolExecutionError(
            code=ToolErrorCode.INVALID_ARGUMENTS,
            category=ToolErrorCategory.VALIDATION,
            message=_safe_message(exc, "Invalid arguments provided"),
            details=ctx,
            causal_error_type=type(exc).__name__,
        )

    if isinstance(exc, PermissionError):
        return ToolExecutionError(
            code=ToolErrorCode.DENIED,
            category=ToolErrorCategory.AUTHORIZATION,
            message=_safe_message(exc, "Permission denied"),
            details=ctx,
            causal_error_type=type(exc).__name__,
        )

    if isinstance(exc, FileNotFoundError):
        return ToolExecutionError(
            code=ToolErrorCode.INVALID_ARGUMENTS,
            category=ToolErrorCategory.VALIDATION,
            message=_safe_message(exc, "Requested file was not found"),
            details=ctx,
            causal_error_type=type(exc).__name__,
        )

    if isinstance(exc, TimeoutError):
        return ToolExecutionError(
            code=ToolErrorCode.TIMEOUT,
            category=ToolErrorCategory.RESOURCE,
            message=_safe_message(exc, "Operation timed out"),
            retryable=True,
            details=ctx,
            causal_error_type=type(exc).__name__,
        )

    if isinstance(exc, (ConnectionError, OSError)):
        return ToolExecutionError(
            code=ToolErrorCode.BACKEND_FAILURE,
            category=ToolErrorCategory.INFRASTRUCTURE,
            message=_safe_message(exc, "Backend operation failed"),
            retryable=True,
            details=ctx,
            causal_error_type=type(exc).__name__,
        )

    return ToolExecutionError(
        code=ToolErrorCode.INTERNAL_FAILURE,
        category=ToolErrorCategory.UNKNOWN,
        message=_safe_message(exc, "An internal tool error occurred"),
        details=ctx,
        causal_error_type=type(exc).__name__,
    )


__all__ = ["map_exception"]
