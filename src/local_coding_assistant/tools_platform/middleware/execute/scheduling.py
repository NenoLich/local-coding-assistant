"""Scheduling middleware for effect classification and lock acquisition.

This module implements the scheduling middleware that classifies
read/write/process/database effects and acquires resource/path locks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from local_coding_assistant.tools_platform.contracts import (
    ToolErrorCategory,
    ToolErrorCode,
    ToolExecutionError,
    ToolExecutionResult,
    ValidatedToolCall,
)
from local_coding_assistant.tools_platform.middleware.base import (
    ExecuteMiddleware,
    FinalizeMiddleware,
    MiddlewareContext,
)

if TYPE_CHECKING:
    from local_coding_assistant.tools_platform.scheduler import Scheduler


class SchedulingMiddleware(ExecuteMiddleware, FinalizeMiddleware):
    """Middleware for effect classification and lock acquisition.

    This middleware:
    - Classifies read/write/process/database effects
    - Acquires resource/path locks
    - Manages concurrency control
    """

    def __init__(self, scheduler: Scheduler) -> None:
        """Initialize scheduling middleware.

        Args:
            scheduler: Concurrency scheduler
        """
        self.scheduler = scheduler

    async def execute(self, context: MiddlewareContext) -> None:
        """Classify effects and acquire locks.

        Args:
            context: Middleware context

        Raises:
            ValueError: If validated call not set in context
        """
        validated_call = context.validated_call
        if validated_call is None:
            raise ValueError("Validated call not set in context")
        if context.decision is not None and context.decision.outcome != "allow":
            return

        # Acquire locks - this returns an async context manager
        lock_context = await self._acquire_locks(validated_call)
        # Enter the lock context for subsequent middleware
        await lock_context.__aenter__()
        # Store lock context for release in finalize
        context.metadata["lock_context"] = lock_context

    async def finalize(
        self, context: MiddlewareContext, result: ToolExecutionResult[Any]
    ) -> ToolExecutionResult[Any]:
        """Release locks on success path.

        Args:
            context: Middleware context
            result: Execution result

        Returns:
            Unmodified result
        """
        # Release the lock context on success
        lock_context = context.metadata.pop("lock_context", None)
        if lock_context is not None:
            await lock_context.__aexit__(None, None, None)
        return result

    async def on_error(
        self,
        context: MiddlewareContext,
        error: Exception,
    ) -> ToolExecutionError:
        """Ensure locks are released on error.

        Args:
            context: Middleware context
            error: Exception that occurred

        Returns:
            Canonical tool execution error
        """
        # Release the lock context on error
        lock_context = context.metadata.pop("lock_context", None)
        if lock_context is not None:
            await lock_context.__aexit__(type(error), error, error.__traceback__)
        # Return a generic error - will be converted by NormalizationMiddleware
        return ToolExecutionError(
            code=ToolErrorCode.INTERNAL_FAILURE,
            category=ToolErrorCategory.INFRASTRUCTURE,
            message=str(error),
        )

    async def _acquire_locks(self, validated_call: ValidatedToolCall):
        """Acquire locks based on tool effects using the scheduler.

        Args:
            validated_call: Validated tool call

        Returns:
            Async context manager for lock release
        """
        return self.scheduler.acquire_locks(validated_call)
