"""ToolGateway - central execution entry point.

The gateway is the only path used by RuntimeExecutor, PTC, CLI tool commands,
and future API callers. It handles:
- Exposure filtering
- Argument validation
- Policy/approval
- Workspace admission
- Scheduling
- Backend execution
- Timeout/output supervision
- Normalization
- Trace/events
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from local_coding_assistant.core.protocols import IConfigManager
    from local_coding_assistant.repository.service import RepositoryContextService
    from local_coding_assistant.storage import IExecutionStore

from local_coding_assistant.tools_platform.approval import ApprovalService
from local_coding_assistant.tools_platform.backends.base import ExecutionBackend
from local_coding_assistant.tools_platform.backends.host import HostBackend
from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ToolCallRequest,
    ToolDecision,
    ToolDecisionOutcome,
    ToolDefinition,
    ToolErrorCategory,
    ToolErrorCode,
    ToolEvent,
    ToolEventType,
    ToolExecutionError,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolOutputChunk,
    ValidatedToolCall,
)
from local_coding_assistant.tools_platform.error_mapping import map_exception
from local_coding_assistant.tools_platform.exceptions import (
    ToolBackendError,
    ToolInterruptedError,
    ToolPlatformError,
)
from local_coding_assistant.tools_platform.exposure_manager import ExposureManager
from local_coding_assistant.tools_platform.middleware import (
    MiddlewareContext,
    MiddlewarePipeline,
)
from local_coding_assistant.tools_platform.middleware.execute import (
    ExecutionMiddleware,
    SchedulingMiddleware,
)
from local_coding_assistant.tools_platform.middleware.finalize import (
    NormalizationMiddleware,
)
from local_coding_assistant.tools_platform.middleware.prepare import (
    AuthorizationMiddleware,
    ExposureMiddleware,
    ValidationMiddleware,
    WorkspaceAdmissionMiddleware,
)
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.scheduler import Scheduler
from local_coding_assistant.tools_platform.workspace import WorkspaceManager

T = TypeVar("T")


class ToolGateway:
    """Central gateway for tool execution.

    The gateway orchestrates the full execution pipeline:
    1. Exposure middleware - filter tools visible to LLM
    2. Decode/validation middleware - validate JSON shape, normalize paths
    3. Authorization middleware - map effects to permissions, evaluate approval
    4. Workspace admission middleware - check path scope, transaction state, conflicts
    5. Scheduling middleware - classify effects, acquire locks
    6. Execution middleware - select backend, supervise streams, capture metrics
    7. Normalization middleware - convert returns/exceptions to canonical result
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config_manager: IConfigManager,
        repository_context_service: RepositoryContextService | None = None,
    ):
        """Initialize the gateway with registry and config manager.

        All components are initialized internally with dynamic config resolution.
        Workspace identity persistence happens in :meth:`init_identity` once the
        store connection is open (called from the async initialize step).

        Args:
            registry: Tool definition registry
            config_manager: Config manager for dynamic configuration resolution
            repository_context_service: Optional repository context service for
                symbol search operations
        """
        self._registry = registry
        self._config_manager = config_manager
        self._repository_context_service = repository_context_service
        self._event_callback: Callable[[ToolEvent], Awaitable[None]] | None = None

        # Initialize all components with config_manager for dynamic config
        self._workspace_manager = WorkspaceManager.from_config_sync(
            config_manager=config_manager,
            repository_context_service=repository_context_service,
        )
        self._approval_service = ApprovalService(
            workspace_manager=self._workspace_manager,
            config_manager=config_manager,
        )
        self._backend = self._create_backend()
        self._scheduler = Scheduler(
            workspace_manager=self._workspace_manager,
            config_manager=config_manager,
        )

        # Initialize exposure manager - backend_type and exposure_policy resolved lazily
        self._exposure_manager = ExposureManager(
            config_manager=config_manager,
        )

        # Initialize middleware pipeline
        self._middleware_pipeline = self._create_middleware_pipeline(
            exposure_manager=self._exposure_manager,
        )
        # Validate per-tool policy config keys against registered tools.
        # Unknown tool IDs are config errors, not silent no-ops.
        self._validate_per_tool_policy_config()

    async def init_identity(self, store: IExecutionStore | None = None) -> str:
        """Resolve the persisted workspace identity after construction.

        The gateway is built synchronously by ``bootstrap`` (which is sync and
        patched as such by tests), so identity persistence happens here, from
        the async ``initialize`` step, once the store connection is open.

        Args:
            store: Optional execution store for identity persistence

        Returns:
            The resolved workspace_id
        """
        return await self._workspace_manager.init_identity(store=store)

    def _validate_per_tool_policy_config(self) -> None:
        """Validate per-tool policy config keys against registered tools.

        Raises:
            ValueError: If a tool ID in tools_platform.rules (excluding wildcard *)
                is not registered.
        """
        tool_config = self._config_manager.global_config.tools_platform
        per_tool = set()
        for rule in tool_config.rules:
            tid = rule.tool_id
            if isinstance(tid, list):
                per_tool.update(tid)
            elif tid != "*":  # Skip wildcard matching all tools
                per_tool.add(tid)
        registered_ids = {d.id for d in self._registry.list_tools()}
        unknown = sorted(set(per_tool) - registered_ids)
        if unknown:
            raise ValueError(
                "tools_platform.rules contains unknown tool IDs: "
                + ", ".join(unknown)
                + ". Registered tools: "
                + ", ".join(sorted(registered_ids))
            )

    def _create_backend(self) -> ExecutionBackend:
        """Create the execution backend based on config."""
        backend_type = self._get_backend_type()
        if backend_type == BackendType.DOCKER:
            from local_coding_assistant.tools_platform.backends.docker import (
                DockerBackend,
            )

            return DockerBackend(
                workspace_root=self._workspace_manager.workspace_root,
                workspace_manager=self._workspace_manager,
                config_manager=self._config_manager,
            )
        return HostBackend(
            workspace_root=self._workspace_manager.workspace_root,
            workspace_manager=self._workspace_manager,
            config_manager=self._config_manager,
        )

    def _get_backend_type(self) -> BackendType:
        """Get backend type from config."""
        backend_str = self._config_manager.global_config.tools_platform.backend
        try:
            return BackendType(backend_str)
        except ValueError:
            return BackendType.HOST

    def _get_exposure_policy(self) -> dict[str, Any] | None:
        """Get exposure policy from config manager if available."""
        exposed_tools = self._config_manager.global_config.tools_platform.exposed_tools
        if isinstance(exposed_tools, list):
            return {"allowed_tools": list(exposed_tools)}
        elif exposed_tools == "auto":
            return {}
        return None

    def _get_limits(self) -> dict[str, int]:
        """Get tool input limits from config manager if available."""
        tc = self._config_manager.global_config.tools_platform
        return {
            "max_output_bytes": tc.max_output_bytes,
            "max_read_lines": tc.max_read_lines,
            "max_search_matches": tc.max_search_matches,
        }

    def register_event_callback(
        self, callback: Callable[[ToolEvent], Awaitable[None]]
    ) -> None:
        """Register an async callback for emitting tool events.

        The callback is awaited by the gateway before execution proceeds
        so subscribers can durably record intent (TOOL_CALL_START) before
        side effects happen and outcome (TOOL_CALL_COMPLETE/ERROR) after.

        Args:
            callback: Async function to call when tool events are emitted
        """
        self._event_callback = callback

    def get_exposed_tool_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for tools exposed to the LLM.

        This uses the same exposure policy that the middleware applies
        at runtime, ensuring the prompt matches what's actually executable.

        Returns:
            List of tool schemas for LLM consumption
        """
        return self._exposure_manager.get_exposed_tool_schemas(self._registry)

    def get_exposed_tools(self) -> list[ToolDefinition]:
        """Get exposed tool definitions.

        Returns:
            List of exposed ToolDefinition objects
        """
        return self._exposure_manager.get_exposed_tools(self._registry)

    @property
    def workspace_manager(self) -> WorkspaceManager:
        """Get the workspace manager for direct access."""
        return self._workspace_manager

    def _create_middleware_pipeline(
        self,
        exposure_manager: ExposureManager,
    ) -> MiddlewarePipeline:
        """Create the middleware pipeline in execution order.

        Pipeline order:
        1. Exposure - filter tools visible to LLM
        2. Validation - validate JSON shape, normalize paths
        3. Authorization - map effects to permissions, evaluate approval
        4. Workspace admission - check path scope, transaction state, conflicts
        5. Scheduling - classify effects, acquire locks
        6. Execution - select backend, supervise streams, capture metrics
        7. Normalization - convert returns/exceptions to canonical result

        Args:
            exposure_manager: Exposure manager for the session

        Returns:
            Configured middleware pipeline
        """
        return MiddlewarePipeline(
            prepare=[
                ExposureMiddleware(exposure_manager=exposure_manager),
                ValidationMiddleware(
                    workspace_manager=self._workspace_manager,
                    limits=self._get_limits(),
                ),
                AuthorizationMiddleware(approval_service=self._approval_service),
                WorkspaceAdmissionMiddleware(workspace_manager=self._workspace_manager),
            ],
            execute=[
                # Scheduling runs before execution so resource/path locks are
                # held for the duration of the backend call
                SchedulingMiddleware(scheduler=self._scheduler),
                ExecutionMiddleware(backend=self._backend),
            ],
            finalize=[
                # Same instance: releases the locks acquired in the EXECUTE
                # phase (and on the error path via on_error)
                SchedulingMiddleware(scheduler=self._scheduler),
                NormalizationMiddleware(backend=self._backend),
            ],
        )

    async def execute(
        self,
        request: ToolCallRequest,
        interrupt_event: asyncio.Event | None = None,
    ) -> ToolExecutionResult[Any]:
        """Execute a tool call and return a single result.

        This is the non-streaming entry point. For streaming tools,
        the gateway collects the bounded stream and returns a single
        normalized result.

        Args:
            request: Tool call request
            interrupt_event: Optional cooperative interrupt signal (e.g. turn
                stop request). When set during execution, the execution is
                cancelled and long-running managed processes are killed.

        Returns:
            ToolExecutionResult with status, output, or error
        """
        context = MiddlewareContext(
            request=request,
            registry=self._registry,
            approval_service=self._approval_service,
            workspace_manager=self._workspace_manager,
            backend=self._backend,
            scheduler=self._scheduler,
            event_callback=self._event_callback,
            interrupt_event=interrupt_event,
        )

        # Phase 1: PREPARE
        short_circuit = await self._run_prepare_phase(context)
        if short_circuit is not None:
            return short_circuit

        if context.validated_call is None:
            raise ValueError("Validated call not set after PREPARE phase")

        # Emit tool_call_start event
        await self._emit_start_event(context.validated_call)

        # Phase 2: EXECUTE
        try:
            for mw in self._middleware_pipeline.execute:
                await mw.execute(context)
        except ToolInterruptedError as exc:
            # Phase 3: FINALIZE (interrupt path) - release locks, report cancelled
            await self._finalize_error(context, exc)
            result = ToolExecutionResult(
                status=ToolExecutionStatus.CANCELLED.value,
                error=exc.to_tool_error(),
                backend=self._backend_type(),
            )
            await self._emit_cancelled_event(context.validated_call, result)
            return result
        except Exception as exc:
            # Phase 3: FINALIZE (error path)
            result = await self._finalize_error(context, exc)
            await self._emit_error_event(context.validated_call, result)
            return result

        # Get execution result
        raw_result = context.metadata.get("execution_result")
        if raw_result is None:
            raise ValueError("Execution middleware did not produce a result")

        # Build initial result
        result = ToolExecutionResult(
            status="completed",
            output=raw_result,
            backend=self._backend_type(),
        )

        # Phase 3: FINALIZE (success path)
        try:
            for mw in self._middleware_pipeline.finalize:
                result = await mw.finalize(context, result)
        except Exception as exc:
            result = await self._finalize_error(context, exc)
            await self._emit_error_event(context.validated_call, result)
            return result

        # Add decision and timing
        if context.decision:
            result.decision = context.decision
        result.timing["elapsed_seconds"] = context.metadata.get("elapsed_seconds", 0)

        # Emit tool_call_complete event
        await self._emit_complete_event(context.validated_call, result)

        return result

    async def _run_prepare_phase(
        self, context: MiddlewareContext
    ) -> ToolExecutionResult[Any] | None:
        """Run the PREPARE phase of the middleware pipeline.

        Args:
            context: Middleware context to prepare

        Returns:
            A terminal result (denied or failed) when the call must not
            continue, or None when PREPARE completed successfully

        Note:
            Exceptions raised during PREPARE are converted once at this
            gateway boundary into canonical failed results.
        """
        try:
            for mw in self._middleware_pipeline.prepare:
                await mw.prepare(context)
                if context.decision and context.decision.outcome != "allow":
                    if context.validated_call is None:
                        raise ValueError("Validated call not set after PREPARE phase")
                    await self._emit_denied_event(
                        context.validated_call, context.decision
                    )
                    return self._build_denied_result(
                        context.validated_call, context.decision
                    )

            if context.validated_call is None:
                raise ValueError("Validated call not set after PREPARE phase")
        except Exception as exc:
            # Exceptions from the gateway boundary are converted once into
            # the canonical error contract (never leaked to the caller)
            return await self._handle_prepare_error(context, exc)
        return None

    async def _handle_prepare_error(
        self, context: MiddlewareContext, error: Exception
    ) -> ToolExecutionResult[Any]:
        """Convert a PREPARE-phase exception into a canonical failed result.

        Runs the finalize on_error hooks (lock release, error normalization)
        and emits the error event when the call was already identified.

        Args:
            context: Middleware context for the failed call
            error: Exception raised during the PREPARE phase

        Returns:
            Canonical failed ToolExecutionResult
        """
        result = await self._finalize_error(context, error)
        # Emit error event even when validated_call is not set (e.g.
        # exceptions raised before validation completes).
        if context.validated_call is not None:
            await self._emit_error_event(context.validated_call, result)
            return result
        fallback = ValidatedToolCall(
            request=context.request,
            tool_definition=ToolDefinition(
                id=context.request.tool_id,
                version="1.0.0",
                description="",
                input_schema={},
                output_schema={},
            ),
            input_model=type("_AnonymousInputModel", (BaseModel,), {})(),
        )
        await self._emit_error_event(fallback, result)
        return result

    async def _finalize_error(
        self, context: MiddlewareContext, error: Exception
    ) -> ToolExecutionResult[Any]:
        """Run FINALIZE phase on error path."""
        request = context.request
        backend_type = self._backend_type()

        if isinstance(error, ToolPlatformError):
            tool_error = error.to_tool_error()
        else:
            tool_error = map_exception(
                error,
                tool_id=request.tool_id,
                call_id=request.call_id,
                backend=backend_type.value,
                workspace_id=request.workspace_id,
            )

        # Run finalize middleware on_error hooks
        for mw in reversed(self._middleware_pipeline.finalize):
            try:
                tool_error = await mw.on_error(context, error)
                if tool_error is not None:
                    break
            except Exception:  # noqa: S112 - on_error failure must not abort remaining hooks
                continue

        # Also run execute middleware on_error for lock release
        for mw in reversed(self._middleware_pipeline.execute):
            on_error = getattr(mw, "on_error", None)
            if callable(on_error):
                try:
                    await on_error(context, error)
                except Exception:  # noqa: S110 - on_error failure during lock release must not mask the original error
                    pass

        return ToolExecutionResult(
            status="failed",
            error=tool_error,
            backend=backend_type,
        )

    async def _prepare_call(
        self,
        index: int,
        request: ToolCallRequest,
        prepared: list[tuple[int, MiddlewareContext]],
        results: list[ToolExecutionResult[Any] | None],
        interrupt_event: asyncio.Event | None,
    ) -> None:
        """Run the PREPARE phase for one call and record its outcome.

        Allowed calls are appended to `prepared` for later scheduling;
        denied or invalid calls get their canonical result in `results`.
        """
        context = MiddlewareContext(
            request=request,
            registry=self._registry,
            approval_service=self._approval_service,
            workspace_manager=self._workspace_manager,
            backend=self._backend,
            scheduler=self._scheduler,
            event_callback=self._event_callback,
            interrupt_event=interrupt_event,
        )
        try:
            # Run PREPARE phase
            for mw in self._middleware_pipeline.prepare:
                await mw.prepare(context)
                if context.decision and context.decision.outcome != "allow":
                    break

            if context.validated_call is None:
                raise ValueError("Middleware pipeline did not prepare the call")

            decision = context.decision
            if decision is None or decision.outcome != "allow":
                # DENY and ASK both short-circuit; DENY gets a terminal result,
                # ASK goes to `prepared` so the batch-pause logic can detect it
                denied_decision = decision or ToolDecision(
                    outcome=ToolDecisionOutcome.DENY,
                    reason="No decision set",
                    policy_rule="no_decision",
                )
                await self._emit_denied_event(context.validated_call, denied_decision)
                if decision is not None and decision.outcome == ToolDecisionOutcome.ASK:
                    prepared.append((index, context))
                else:
                    results[index] = self._build_denied_result(
                        context.validated_call, denied_decision
                    )
            else:
                prepared.append((index, context))
        except ToolPlatformError as exc:
            results[index] = self._build_error_result(
                request, exc.to_tool_error(), self._backend_type()
            )
        except Exception as exc:
            backend_type = self._backend_type()
            results[index] = self._build_error_result(
                request,
                map_exception(
                    exc,
                    tool_id=request.tool_id,
                    call_id=request.call_id,
                    backend=backend_type.value,
                    workspace_id=request.workspace_id,
                ),
                backend_type,
            )

    async def _pause_batch_for_approval(
        self,
        prepared: list[tuple[int, MiddlewareContext]],
        results: list[ToolExecutionResult[Any] | None],
    ) -> None:
        """Fill batch results when any call requires approval.

        ALLOWed calls are marked PENDING_APPROVAL (batch policy: nothing
        executes when any sibling call needs approval); other outcomes keep
        their canonical denied/pending results.
        """
        for index, context in prepared:
            if not (context.validated_call and context.decision):
                continue
            if context.decision.outcome == "allow":
                # Emit a denial event so subscribers learn the batch was paused
                denied = ToolDecision(
                    outcome=ToolDecisionOutcome.ASK,
                    reason="Batch paused: another tool requires approval",
                    policy_rule="batch_pause",
                )
                await self._emit_denied_event(context.validated_call, denied)
                results[index] = ToolExecutionResult(
                    status=ToolExecutionStatus.PENDING_APPROVAL.value,
                    backend=self._backend_type(),
                    decision=denied,
                    error=ToolExecutionError(
                        code=ToolErrorCode.APPROVAL_REQUIRED,
                        category=ToolErrorCategory.AUTHORIZATION,
                        message="Batch paused: another tool requires approval",
                    ),
                )
            else:
                results[index] = ToolExecutionResult(
                    status=ToolExecutionStatus.PENDING_APPROVAL.value,
                    backend=self._backend_type(),
                    decision=context.decision,
                    error=ToolExecutionError(
                        code=ToolErrorCode.APPROVAL_REQUIRED,
                        category=ToolErrorCategory.AUTHORIZATION,
                        message=context.decision.reason,
                    ),
                )

    async def execute_many(
        self,
        requests: Sequence[ToolCallRequest],
        interrupt_event: asyncio.Event | None = None,
    ) -> list[ToolExecutionResult[Any]]:
        """Execute calls in scheduler-defined parallel groups.

        Calls are prepared before execution locks are acquired. The scheduler
        groups calls with disjoint resources for concurrent execution and
        preserves the caller's request order in the returned results.

        Interruption: when interrupt_event is set, calls that have not started
        executing yet are reported as cancelled (TOOL_CALL_CANCELLED), and
        in-flight calls are cancelled cooperatively by the execution middleware
        (which kills supervised long-running processes). Groups that already
        started run to completion.
        """
        results: list[ToolExecutionResult[Any] | None] = [None] * len(requests)
        prepared: list[tuple[int, MiddlewareContext]] = []

        async def prepare(index: int, request: ToolCallRequest) -> None:
            await self._prepare_call(index, request, prepared, results, interrupt_event)

        await asyncio.gather(
            *(prepare(index, request) for index, request in enumerate(requests))
        )

        # Atomic batch: if ANY call needs approval (ASK), execute NONE
        # Return all results with their decisions; runtime re-submits after approval
        any_ask = any(
            context.decision and context.decision.outcome == ToolDecisionOutcome.ASK
            for _, context in prepared
            if context.validated_call and context.decision
        )

        if any_ask:
            await self._pause_batch_for_approval(prepared, results)
            return [r for r in results if r is not None]

        groups = await self._scheduler.classify_tool_calls(
            [
                context.validated_call
                for _, context in prepared
                if context.validated_call
            ]
        )
        prepared_by_call_id = {
            context.validated_call.request.call_id: (index, context)
            for index, context in prepared
            if context.validated_call is not None
        }

        group_items = list(groups.items())
        for group_number, (_group_key, group) in enumerate(group_items):
            if interrupt_event is not None and interrupt_event.is_set():
                # Cooperative interrupt between scheduler groups: groups that
                # already started run to completion, but nothing new starts
                await self._cancel_remaining_groups(
                    group_items,
                    group_number,
                    results,
                    prepared_by_call_id,
                )
                break

            await self._run_group(group, prepared_by_call_id, results)

        return [result for result in results if result is not None]

    async def _run_group(
        self,
        group: list[ValidatedToolCall],
        prepared_by_call_id: dict[str, tuple[int, MiddlewareContext]],
        results: list[ToolExecutionResult[Any] | None],
    ) -> None:
        """Execute one scheduler group in parallel and collect results."""
        group_results = await asyncio.gather(
            *(
                self._execute_prepared_context(
                    prepared_by_call_id[validated.request.call_id][1],
                )
                for validated in group
            ),
            return_exceptions=True,
        )
        for validated, group_result in zip(group, group_results, strict=True):
            index, _context = prepared_by_call_id[validated.request.call_id]
            if isinstance(group_result, BaseException):
                if not isinstance(group_result, Exception):
                    # KeyboardInterrupt, SystemExit, etc. - re-raise
                    raise group_result
                backend_type = self._backend_type()
                results[index] = self._build_error_result(
                    validated.request,
                    map_exception(
                        group_result,
                        tool_id=validated.request.tool_id,
                        call_id=validated.request.call_id,
                        backend=backend_type.value,
                        workspace_id=validated.request.workspace_id,
                    ),
                    backend_type,
                )
            else:
                results[index] = group_result

    async def _cancel_remaining_groups(
        self,
        group_items: list[tuple[str, list[ValidatedToolCall]]],
        start_group: int,
        results: list[ToolExecutionResult[Any] | None],
        prepared_by_call_id: dict[str, tuple[int, MiddlewareContext]],
    ) -> None:
        """Report not-yet-executed calls as cancelled after an interrupt.

        Fills results for every remaining group's calls so the returned list
        stays aligned with the caller's request order.
        """
        for _group_key, remaining_group in group_items[start_group:]:
            for validated in remaining_group:
                index, _ctx = prepared_by_call_id[validated.request.call_id]
                if results[index] is None:
                    cancelled = self._build_cancelled_result(
                        validated.request,
                        reason="Interrupted before execution started",
                    )
                    results[index] = cancelled
                    await self._emit_cancelled_event(validated, cancelled)

    async def _execute_prepared_context(
        self,
        context: MiddlewareContext,
    ) -> ToolExecutionResult[Any]:
        """Run the remaining middleware phases for a prepared call."""
        validated_call = context.validated_call
        decision = context.decision
        if validated_call is None or decision is None:
            raise ValueError("Prepared context is missing call or decision")

        started = time.perf_counter()
        try:
            # Emit tool_call_start event
            await self._emit_start_event(validated_call)

            # Phase 2: EXECUTE
            for mw in self._middleware_pipeline.execute:
                await mw.execute(context)

            raw_result = context.metadata.get("execution_result")
            if raw_result is None:
                raise ValueError("Execution middleware did not produce a result")

            result = ToolExecutionResult(
                status="completed",
                output=raw_result,
                backend=self._backend_type(),
            )
            # Phase 3: FINALIZE
            for mw in self._middleware_pipeline.finalize:
                result = await mw.finalize(context, result)

            result.decision = decision
            result.timing["elapsed_seconds"] = time.perf_counter() - started

            # Emit tool_call_complete event
            await self._emit_complete_event(validated_call, result)

            return result
        except ToolInterruptedError as exc:
            # Cooperative interrupt: release locks (finalize on_error hooks),
            # then report the call as cancelled with a terminal event
            await self._finalize_error(context, exc)
            result = ToolExecutionResult(
                status=ToolExecutionStatus.CANCELLED.value,
                error=exc.to_tool_error(),
                backend=self._backend_type(),
            )
            result.timing["elapsed_seconds"] = time.perf_counter() - started
            await self._emit_cancelled_event(validated_call, result)
            return result
        except Exception as exc:
            # Run finalize on error
            result = await self._finalize_error(context, exc)
            await self._emit_error_event(validated_call, result)
            return result

    async def execute_stream(
        self,
        request: ToolCallRequest,
        interrupt_event: asyncio.Event | None = None,
    ) -> AsyncIterator[ToolOutputChunk]:
        """Execute a tool call and stream output.

        This is the streaming entry point. Each chunk is typed and
        the final result is emitted exactly once.

        Args:
            request: Tool call request
            interrupt_event: Optional cooperative interrupt signal (e.g. turn
                stop request). When set during execution, the execution is
                canceled and long-running managed processes are killed.

        Yields:
            ToolOutputChunk with streaming output
        """
        backend_type = None
        context: MiddlewareContext | None = None
        try:
            context = self._build_stream_context(request, interrupt_event)
            context.metadata["streaming"] = True

            # Phase 1: PREPARE
            validated_call, decision = await self._stream_prepare(context)
            backend_type = self._backend_type()

            if decision.outcome != "allow":
                # Emit denied event
                await self._emit_denied_event(validated_call, decision)
                yield ToolOutputChunk(
                    stream_id=request.call_id,
                    sequence=0,
                    channel="error",
                    payload={"error": "denied", "reason": decision.reason},
                )
                return

            # Emit tool_call_start event
            await self._emit_start_event(validated_call)

            # Phase 2: EXECUTE
            stream = await self._stream_execute(context)

            async for chunk in stream:
                yield chunk

            # Phase 3: FINALIZE
            await self._stream_finalize(context, backend_type)

            # Emit tool_call_complete event
            await self._emit_complete_event(
                validated_call,
                ToolExecutionResult(status="completed", backend=backend_type),
            )

        except Exception as exc:
            async for chunk in self._stream_handle_error(
                request, context, exc, backend_type
            ):
                yield chunk
        finally:
            if context is not None and context.metadata.get("lock_context") is not None:
                # Release locks if not already released by SchedulingMiddleware.finalize()
                lock_context = context.metadata.pop("lock_context")
                await lock_context.__aexit__(None, None, None)

    def _build_stream_context(
        self,
        request: ToolCallRequest,
        interrupt_event: asyncio.Event | None,
    ) -> MiddlewareContext:
        """Build the middleware context for a streamed tool call."""
        return MiddlewareContext(
            request=request,
            registry=self._registry,
            approval_service=self._approval_service,
            workspace_manager=self._workspace_manager,
            backend=self._backend,
            scheduler=self._scheduler,
            event_callback=self._event_callback,
            interrupt_event=interrupt_event,
        )

    async def _stream_prepare(
        self, context: MiddlewareContext
    ) -> tuple[ValidatedToolCall, ToolDecision]:
        """Run the PREPARE phase for a streamed call."""
        for mw in self._middleware_pipeline.prepare:
            await mw.prepare(context)
            if context.decision and context.decision.outcome != "allow":
                break
        if context.validated_call is None or context.decision is None:
            raise ValueError("Middleware pipeline did not prepare the call")
        return context.validated_call, context.decision

    async def _stream_execute(
        self, context: MiddlewareContext
    ) -> AsyncIterator[ToolOutputChunk]:
        """Run the EXECUTE phase and return the byte stream."""
        for mw in self._middleware_pipeline.execute:
            await mw.execute(context)
        stream = context.metadata.get("execution_stream")
        if stream is None:
            raise ValueError("Execution middleware did not produce a stream")
        return stream

    async def _stream_finalize(
        self, context: MiddlewareContext, backend_type: BackendType
    ) -> None:
        """Run the FINALIZE phase for a completed streamed call."""
        for mw in self._middleware_pipeline.finalize:
            await mw.finalize(
                context,
                ToolExecutionResult(status="completed", backend=backend_type),
            )

    async def _stream_handle_error(
        self,
        request: ToolCallRequest,
        context: MiddlewareContext | None,
        exc: Exception,
        backend_type: BackendType | None,
    ) -> AsyncIterator[ToolOutputChunk]:
        """Handle an error during streamed execution, yielding an error chunk."""
        if isinstance(exc, ToolInterruptedError):
            error = exc.to_tool_error()
            yield ToolOutputChunk(
                stream_id=request.call_id,
                sequence=0,
                channel="status",
                payload={
                    "status": ToolExecutionStatus.CANCELLED.value,
                    "error": error.model_dump(),
                },
            )
            if context and context.validated_call:
                await self._emit_cancelled_event(
                    context.validated_call,
                    ToolExecutionResult(
                        status=ToolExecutionStatus.CANCELLED.value,
                        error=error,
                        backend=backend_type,
                    ),
                )
            return

        if isinstance(exc, ToolPlatformError):
            error = exc.to_tool_error()
        else:
            error = map_exception(
                exc,
                tool_id=request.tool_id,
                call_id=request.call_id,
                backend=backend_type.value if backend_type is not None else None,
                workspace_id=request.workspace_id,
            )

        yield self._build_error_chunk(request, error, backend_type)
        if context and context.validated_call:
            await self._emit_error_event(
                context.validated_call,
                ToolExecutionResult(status="failed", error=error, backend=backend_type),
            )

    async def _emit_cancelled_event(
        self,
        validated_call: ValidatedToolCall,
        result: ToolExecutionResult[Any],
        reason: str | None = None,
    ) -> None:
        """Emit tool_call_cancelled event when execution is interrupted.

        Cancelled is terminal: the outcome does not depend on restarting the
        tool, so subscribers can close the action record as cancelled.

        Args:
            validated_call: Validated tool call
            result: Cancelled execution result carrying the canonical error
            reason: Optional human-readable reason override
        """
        if not self._event_callback:
            return

        trace_id = str(uuid.uuid4())
        event = ToolEvent(
            type=ToolEventType.TOOL_CALL_CANCELLED,
            call_id=validated_call.request.call_id,
            session_id=validated_call.request.session_id,
            frame_id=validated_call.request.frame_id,
            trace_id=trace_id,
            data={
                "tool_id": validated_call.tool_definition.id,
                "status": ToolExecutionStatus.CANCELLED.value,
                "output": None,
                "error": result.error.model_dump() if result.error else None,
                "reason": reason or (result.error.message if result.error else None),
                "timing": result.timing,
                "backend": result.backend.value if result.backend else None,
                "arguments": validated_call.request.raw_arguments,
            },
        )
        await self._event_callback(event)

    async def _emit_start_event(self, validated_call: ValidatedToolCall) -> None:
        """Emit tool_call_start event before execution.

        Payload carries everything a subscriber needs to open a durable
        action record: tool_id, arguments, backend, workspace/transaction.
        """
        if not self._event_callback:
            return

        trace_id = str(uuid.uuid4())
        event = ToolEvent(
            type=ToolEventType.TOOL_CALL_START,
            call_id=validated_call.request.call_id,
            session_id=validated_call.request.session_id,
            frame_id=validated_call.request.frame_id,
            trace_id=trace_id,
            data={
                "tool_id": validated_call.tool_definition.id,
                "tool_version": validated_call.tool_definition.version,
                "risk_class": validated_call.tool_definition.risk_class.value,
                "arguments": validated_call.request.raw_arguments,
                "backend": self._backend.get_backend_type(),
                "workspace_id": validated_call.request.workspace_id,
                "transaction_id": validated_call.request.transaction_id,
            },
        )
        await self._event_callback(event)

    async def _emit_complete_event(
        self, validated_call: ValidatedToolCall, result: ToolExecutionResult[Any]
    ) -> None:
        """Emit tool_call_complete event after successful execution.

        Payload carries the full outcome (output, error, decision, changeset)
        so subscribers can close a durable action record without needing
        the in-process return value.
        """
        if not self._event_callback:
            return

        trace_id = str(uuid.uuid4())
        event = ToolEvent(
            type=ToolEventType.TOOL_CALL_COMPLETE,
            call_id=validated_call.request.call_id,
            session_id=validated_call.request.session_id,
            frame_id=validated_call.request.frame_id,
            trace_id=trace_id,
            data={
                "tool_id": validated_call.tool_definition.id,
                "status": result.status,
                "output": result.output,
                "error": result.error.model_dump() if result.error else None,
                "timing": result.timing,
                "backend": result.backend.value if result.backend else None,
                "changeset": result.changeset.model_dump()
                if result.changeset
                else None,
                "warnings": result.warnings,
                "decision": result.decision.model_dump() if result.decision else None,
                "arguments": validated_call.request.raw_arguments,
            },
        )
        await self._event_callback(event)

    async def _emit_error_event(
        self, validated_call: ValidatedToolCall, result: ToolExecutionResult[Any]
    ) -> None:
        """Emit tool_call_error event on execution failure.

        Args:
            validated_call: Validated tool call
            result: Canonical failed result from the error path
        """
        if not self._event_callback:
            return

        trace_id = str(uuid.uuid4())
        event = ToolEvent(
            type=ToolEventType.TOOL_CALL_ERROR,
            call_id=validated_call.request.call_id,
            session_id=validated_call.request.session_id,
            frame_id=validated_call.request.frame_id,
            trace_id=trace_id,
            data={
                "tool_id": validated_call.tool_definition.id,
                "status": "failed",
                "output": None,
                "error": result.error.model_dump() if result.error else None,
                "timing": result.timing,
                "backend": result.backend.value if result.backend else None,
                "arguments": validated_call.request.raw_arguments,
            },
        )
        await self._event_callback(event)

    async def _emit_denied_event(
        self,
        validated_call: ValidatedToolCall,
        decision: ToolDecision,
    ) -> None:
        """Emit tool_call_denied event when authorization fails.

        The decision outcome is included so subscribers can distinguish a
        hard denial (terminal) from an approval request (not terminal).

        Args:
            validated_call: Validated tool call
            decision: Authorization decision
        """
        if not self._event_callback:
            return

        event = ToolEvent(
            type=ToolEventType.TOOL_CALL_DENIED,
            call_id=validated_call.request.call_id,
            session_id=validated_call.request.session_id,
            frame_id=validated_call.request.frame_id,
            trace_id="",
            data={
                "tool_id": validated_call.tool_definition.id,
                "outcome": decision.outcome.value,
                "status": "denied"
                if decision.outcome == ToolDecisionOutcome.DENY
                else None,
                "reason": decision.reason,
                "policy_rule": decision.policy_rule,
                "confirmation_token": decision.confirmation_token,
                "arguments": validated_call.request.raw_arguments,
            },
        )
        await self._event_callback(event)

    def _backend_type(self) -> BackendType:
        """Return the configured backend type as a canonical enum."""
        try:
            return BackendType(self._backend.get_backend_type())
        except (TypeError, ValueError) as exc:
            raise ToolBackendError(
                "Configured backend returned an invalid backend type",
                details={"backend_type": self._backend.get_backend_type()},
                causal_error_type=type(exc).__name__,
            ) from exc

    def _build_error_result(
        self,
        request: ToolCallRequest,
        error: ToolExecutionError,
        backend_type: BackendType | None,
    ) -> ToolExecutionResult[Any]:
        """Build a canonical failed result from a ToolExecutionError."""
        if error.details is not None and "tool_id" not in error.details:
            error.details["tool_id"] = request.tool_id
        if error.details is not None and "call_id" not in error.details:
            error.details["call_id"] = request.call_id
        if backend_type is not None and "backend" not in error.details:
            error.details["backend"] = backend_type.value
        return ToolExecutionResult(
            status="failed",
            error=error,
            backend=backend_type,
        )

    def _build_error_chunk(
        self,
        request: ToolCallRequest,
        error: ToolExecutionError,
        backend_type: BackendType | None,
    ) -> ToolOutputChunk:
        """Build a streaming error chunk from a ToolExecutionError."""
        details = dict(error.details)
        details.setdefault("tool_id", request.tool_id)
        details.setdefault("call_id", request.call_id)
        if backend_type is not None:
            details.setdefault("backend", backend_type.value)
        return ToolOutputChunk(
            stream_id=request.call_id,
            sequence=0,
            channel="error",
            payload={
                "error": error.code.value,
                "message": error.message,
                "details": details,
            },
        )

    def _build_cancelled_result(
        self,
        request: ToolCallRequest,
        reason: str = "Tool call interrupted",
        backend_type: BackendType | None = None,
    ) -> ToolExecutionResult[Any]:
        """Build a canonical cancelled result for an interrupted call."""
        error = ToolExecutionError(
            code=ToolErrorCode.INTERRUPTED,
            category=ToolErrorCategory.CONCURRENCY,
            message=reason,
        )
        if error.details is not None:
            error.details.setdefault("tool_id", request.tool_id)
            error.details.setdefault("call_id", request.call_id)
        return ToolExecutionResult(
            status=ToolExecutionStatus.CANCELLED.value,
            error=error,
            backend=backend_type,
        )

    def _build_denied_result(
        self,
        validated_call: ValidatedToolCall,
        decision: ToolDecision,
    ) -> ToolExecutionResult[Any]:
        """Build a denied result from a decision."""
        # Carry the confirmation token so callers can approve the same call
        # and resume it unchanged (approval loop entry point)
        details: dict[str, Any] = {}
        if decision.confirmation_token:
            details = {"confirmation_token": decision.confirmation_token}
        error = ToolExecutionError(
            code=ToolErrorCode.DENIED
            if decision.outcome == "deny"
            else ToolErrorCode.APPROVAL_REQUIRED,
            category=ToolErrorCategory.AUTHORIZATION,
            message=decision.reason,
            details=details,
        )
        return ToolExecutionResult(
            status="denied",
            error=error,
            decision=decision,
        )
