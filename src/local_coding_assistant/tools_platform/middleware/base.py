"""Middleware protocol and pipeline for tool execution.

This module defines the middleware interface and pipeline composition
for the tool gateway. Middleware provides composable hooks around tool
execution for exposure, validation, authorization, workspace admission,
scheduling, execution, normalization, and event emission.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from local_coding_assistant.tools_platform.contracts import (
    ToolCallRequest,
    ToolDecision,
    ToolEvent,
    ToolExecutionError,
    ToolExecutionResult,
    ValidatedToolCall,
)

if TYPE_CHECKING:
    from local_coding_assistant.tools_platform.approval import ApprovalService
    from local_coding_assistant.tools_platform.backends.base import ExecutionBackend
    from local_coding_assistant.tools_platform.registry import ToolRegistry
    from local_coding_assistant.tools_platform.scheduler import Scheduler
    from local_coding_assistant.tools_platform.workspace import WorkspaceManager


class MiddlewareStage(Enum):
    """Pipeline execution stages.

    Stages define logical groupings of middleware that can be executed
    independently. This enables patterns like parallel preparation
    followed by serialized execution with locks.
    """

    PREPARE = "prepare"
    EXECUTE = "execute"
    FINALIZE = "finalize"


class MiddlewareContext:
    """Context object passed through middleware chain.

    This context carries request, validated call, decision, and any
    additional metadata through the middleware pipeline.
    """

    def __init__(
        self,
        request: ToolCallRequest,
        registry: ToolRegistry,
        approval_service: ApprovalService,
        workspace_manager: WorkspaceManager,
        backend: ExecutionBackend,
        scheduler: Scheduler,
        event_callback: Callable[[ToolEvent], Awaitable[None]] | None = None,
        interrupt_event: asyncio.Event | None = None,
    ) -> None:
        """Initialize middleware context.

        Args:
            request: Tool call request
            registry: Tool definition registry
            approval_service: Approval policy engine
            workspace_manager: Workspace and transaction manager
            backend: Execution backend
            scheduler: Concurrency scheduler
            event_callback: Optional callback for emitting tool events
            interrupt_event: Optional cooperative interrupt signal (e.g. turn
                stop request). When set while a tool executes, the execution
                is cancelled and long-running managed processes are killed.
        """
        self.request = request
        self.registry = registry
        self.approval_service = approval_service
        self.workspace_manager = workspace_manager
        self.backend = backend
        self.scheduler = scheduler
        self._event_callback = event_callback
        self.interrupt_event = interrupt_event

        # Populated during pipeline execution
        self.validated_call: ValidatedToolCall | None = None
        self.decision: ToolDecision | None = None
        self.execution_result: ToolExecutionResult[Any] | None = None
        self.metadata: dict[str, Any] = {}

    async def emit_event(self, event: ToolEvent) -> None:
        """Emit a tool event if a callback is registered.

        Args:
            event: Tool event to emit
        """
        if self._event_callback:
            await self._event_callback(event)


class PrepareMiddleware:
    """Base class for middleware that runs in the PREPARE phase.

    PREPARE phase middleware:
    - Validates and normalizes the request
    - Checks authorization and approval
    - Verifies workspace admission
    - Must NOT acquire locks or execute tools
    """

    async def prepare(self, context: MiddlewareContext) -> None:
        """Run prepare-phase logic.

        Mutates context.validated_call, context.decision, context.metadata.
        Raises exception to short-circuit (handled by gateway).
        """
        raise NotImplementedError


class ExecuteMiddleware:
    """Base class for middleware that runs in the EXECUTE phase.

    EXECUTE phase middleware:
    - Acquires locks and classifies effects
    - Executes the tool on the backend
    - Produces execution_result or execution_stream in context.metadata
    """

    async def execute(self, context: MiddlewareContext) -> None:
        """Run execute-phase logic.

        Mutates context.metadata with execution_result or execution_stream.
        Raises exception on failure.
        """
        raise NotImplementedError


class FinalizeMiddleware:
    """Base class for middleware that runs in the FINALIZE phase.

    FINALIZE phase middleware:
    - Normalizes the execution result
    - Releases locks and cleans up resources
    - Converts exceptions to canonical ToolExecutionError
    """

    async def finalize(
        self, context: MiddlewareContext, result: ToolExecutionResult[Any]
    ) -> ToolExecutionResult[Any]:
        """Run finalize-phase logic on successful result.

        Args:
            context: Middleware context
            result: Execution result from EXECUTE phase

        Returns:
            Normalized execution result
        """
        raise NotImplementedError

    async def on_error(
        self, context: MiddlewareContext, error: Exception
    ) -> ToolExecutionError:
        """Convert exception to canonical ToolExecutionError.

        Args:
            context: Middleware context
            error: Exception that occurred

        Returns:
            Canonical tool execution error
        """
        raise error


@dataclass
class MiddlewarePipeline:
    """Explicit phase-based middleware pipeline.

    The pipeline composes middleware in order per phase and executes
    them at the appropriate stages of tool execution.
    """

    prepare: list[PrepareMiddleware] = field(default_factory=list)
    execute: list[ExecuteMiddleware] = field(default_factory=list)
    finalize: list[FinalizeMiddleware] = field(default_factory=list)

    @property
    def all(self) -> list[PrepareMiddleware | ExecuteMiddleware | FinalizeMiddleware]:
        """All middleware in phase order (for backward compat)."""
        return self.prepare + self.execute + self.finalize
