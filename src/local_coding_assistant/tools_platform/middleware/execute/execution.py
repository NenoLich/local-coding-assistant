"""Execution middleware for backend execution and supervision.

This module implements the execution middleware that selects the backend,
supervises streams and cancellation, and captures metrics.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from local_coding_assistant.tools_platform.contracts import (
    ToolOutputChunk,
    ValidatedToolCall,
)
from local_coding_assistant.tools_platform.exceptions import (
    ToolInterruptedError,
    ToolValidationError,
)
from local_coding_assistant.tools_platform.middleware.base import (
    ExecuteMiddleware,
    MiddlewareContext,
)
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.tools_platform.backends.base import ExecutionBackend


logger = get_logger("tools_platform.middleware.execute")


class ExecutionMiddleware(ExecuteMiddleware):
    """Middleware for backend execution and supervision.

    This middleware:
    - Selects the backend (fixed per session)
    - Supervises streams and cancellation
    - Captures metrics and timing
    """

    def __init__(self, backend: ExecutionBackend) -> None:
        """Initialize execution middleware.

        Args:
            backend: Execution backend
        """
        self.backend = backend

    async def execute(self, context: MiddlewareContext) -> None:
        """Execute tool on backend (non-streaming or streaming).

        Args:
            context: Middleware context

        Raises:
            ValueError: If validated call not set in context
            ToolValidationError: If tool implementation not found
        """
        validated_call = context.validated_call
        if validated_call is None:
            raise ValueError("Validated call not set in context")
        if context.decision is not None and context.decision.outcome != "allow":
            return

        if context.metadata.get("streaming"):
            context.metadata["execution_stream"] = self.execute_stream(context)
            return

        # Execute on backend and store result in context
        result = await self._execute_on_backend(validated_call, self.backend, context)
        context.metadata["execution_result"] = result

    async def _execute_on_backend(
        self,
        validated_call: ValidatedToolCall,
        backend: ExecutionBackend,
        context: MiddlewareContext,
    ) -> Any:
        """Execute tool on backend.

        Races the tool coroutine against the context interrupt event when one
        is present. On interrupt the execution task is cancelled (which makes
        supervised long-running processes terminate) and ToolInterruptedError
        is raised so the gateway can report a cancelled outcome.

        Args:
            validated_call: Validated tool call
            backend: Execution backend
            context: Middleware context

        Returns:
            Raw result from backend

        Raises:
            ToolInterruptedError: If the interrupt event fired during execution
        """
        # Get the tool implementation
        tool_impl = context.registry.get_implementation(validated_call.request.tool_id)
        if tool_impl is None:
            raise ToolValidationError(
                f"Tool implementation not found: {validated_call.request.tool_id}",
            )

        # Call the tool implementation with the backend and input model
        session_id = validated_call.request.session_id
        turn_id = validated_call.request.turn_id
        result = tool_impl.factory(
            backend, validated_call.input_model, session_id=session_id, turn_id=turn_id
        )
        if not inspect.isawaitable(result):
            return result

        timeout_seconds = self._resolve_timeout(validated_call)

        interrupt_event = context.interrupt_event
        if interrupt_event is None:
            return await self._await_with_timeout(result, timeout_seconds)

        return await self._race_with_interrupt(
            result,
            timeout_seconds,
            interrupt_event,
            validated_call,
        )

    def _resolve_timeout(self, validated_call: ValidatedToolCall) -> float | None:
        """Resolve the effective timeout in seconds for a tool call.

        Falls back to the timeout profile's hard timeout when no deadline
        policy is set.
        """
        timeout = validated_call.request.deadline_policy
        profile = validated_call.tool_definition.timeout_profile
        if timeout is None and profile is not None:
            timeout = profile.hard_timeout
        return timeout.total_seconds() if timeout is not None else None

    async def _await_with_timeout(
        self,
        result: Any,
        timeout_seconds: float | None,
    ) -> Any:
        """Await a coroutine, optionally bounded by a timeout."""
        if timeout_seconds is not None:
            return await asyncio.wait_for(result, timeout=timeout_seconds)
        return await result

    async def _race_with_interrupt(
        self,
        result: Any,
        timeout_seconds: float | None,
        interrupt_event: asyncio.Event,
        validated_call: ValidatedToolCall,
    ) -> Any:
        """Race execution against the context interrupt event.

        On interrupt the execution task is cancelled (which makes supervised
        long-running processes terminate) and ToolInterruptedError is raised
        so the gateway can report a cancelled outcome.
        """
        started = time.perf_counter()
        execution = asyncio.ensure_future(
            asyncio.wait_for(result, timeout=timeout_seconds)
            if timeout_seconds is not None
            else result
        )
        interrupt_waiter = asyncio.ensure_future(interrupt_event.wait())
        try:
            done, _ = await asyncio.wait(
                {execution, interrupt_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            interrupt_waiter.cancel()

        if execution in done:
            # Completed (or raised - errors propagate unchanged)
            return execution.result()

        # Interrupted: cancel the execution task and wait for it so backend
        # cleanup (process termination, lock release) completes before return
        execution.cancel()
        try:
            await asyncio.wait_for(execution, timeout=5.0)
        except TimeoutError:
            logger.debug(
                "Tool execution did not respond to cancellation within timeout",
                tool_id=validated_call.request.tool_id,
            )
        except asyncio.CancelledError:
            if not execution.cancelled():
                # Interrupt came from the outside, not our own cancel
                raise
        except Exception:
            # Secondary failures during teardown are not the interrupt cause
            logger.debug(
                "Tool execution raised during interrupt teardown",
                tool_id=validated_call.request.tool_id,
            )

        raise ToolInterruptedError(
            f"Tool '{validated_call.request.tool_id}' interrupted after "
            f"{time.perf_counter() - started:.2f}s of execution",
            details={
                "tool_id": validated_call.request.tool_id,
                "call_id": validated_call.request.call_id,
            },
        )

    async def execute_stream(
        self,
        context: MiddlewareContext,
    ) -> AsyncIterator[ToolOutputChunk]:
        """Execute tool on backend with streaming.

        Args:
            context: Middleware context

        Yields:
            ToolOutputChunk with streaming output
        """
        validated_call = context.validated_call
        if validated_call is None:
            raise ValueError("Validated call not set in context")

        result = await self._execute_on_backend(validated_call, self.backend, context)
        yield ToolOutputChunk(
            stream_id=validated_call.request.call_id,
            sequence=0,
            channel="status",
            payload={
                "status": "completed",
                "output": (
                    result.model_dump(mode="json")
                    if isinstance(result, BaseModel)
                    else result
                ),
            },
        )
