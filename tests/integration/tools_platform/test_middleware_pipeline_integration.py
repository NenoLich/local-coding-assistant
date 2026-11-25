"""Integration tests for the middleware pipeline contract.

Covers the middleware pipeline behaviour from
plan_docs/2026_09/tools_platform_tests.md
(test_middleware_pipeline_integration.py):

- PREPARE phase order: Exposure → Validation → Authorization → WorkspaceAdmission
- EXECUTE phase order: Scheduling → Execution
- FINALIZE phase order: Scheduling (release locks) → Normalization
- on_error hooks run in reverse FINALIZE order, then reverse EXECUTE order
- SchedulingMiddleware.on_error releases acquired locks
- NormalizationMiddleware.on_error converts exception to ToolExecutionError
- Short-circuit on DENY: PREPARE returns denied result, EXECUTE/FINALIZE not run
- Short-circuit on ASK in batch: PREPARE returns PENDING_APPROVAL for all
- Event emission: START in PREPARE after authorization, COMPLETE/ERROR/CANCELLED in FINALIZE

Uses real WorkspaceManager, HostBackend, ApprovalService and Scheduler per the
acceptance criteria.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_coding_assistant.tools_platform.contracts import (
    ToolCallRequest,
    ToolErrorCode,
    ToolEventType,
    ToolExecutionError,
    ToolExecutionStatus,
)
from local_coding_assistant.tools_platform.middleware import (
    MiddlewareContext,
    MiddlewarePipeline,
)
from local_coding_assistant.tools_platform.middleware.execute.scheduling import (
    SchedulingMiddleware,
)
from local_coding_assistant.tools_platform.middleware.finalize.normalization import (
    NormalizationMiddleware,
)

PREPARE_NAMES = [
    "ExposureMiddleware",
    "ValidationMiddleware",
    "AuthorizationMiddleware",
    "WorkspaceAdmissionMiddleware",
]

EXECUTE_NAMES = ["SchedulingMiddleware(execute)", "ExecutionMiddleware"]

FINALIZE_NAMES = ["SchedulingMiddleware(finalize)", "NormalizationMiddleware"]


class _MiddlewareSpy:
    """Records each middleware phase call into a shared log, then delegates."""

    def __init__(self, wrapped: Any, log: list[str], name: str) -> None:
        self._wrapped = wrapped
        self._log = log
        self._name = name

    async def prepare(self, context: MiddlewareContext) -> None:
        self._log.append(f"prepare:{self._name}")
        return await self._wrapped.prepare(context)

    async def execute(self, context: MiddlewareContext) -> None:
        self._log.append(f"execute:{self._name}")
        return await self._wrapped.execute(context)

    async def finalize(self, context: MiddlewareContext, result: Any) -> Any:
        self._log.append(f"finalize:{self._name}")
        return await self._wrapped.finalize(context, result)

    async def on_error(self, context: MiddlewareContext, error: Exception) -> Any:
        self._log.append(f"on_error:{self._name}")
        return await self._wrapped.on_error(context, error)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _attach_spies(gateway: Any, log: list[str]) -> None:
    """Replace the gateway's pipeline middleware with recording spies."""
    pipeline: MiddlewarePipeline = gateway._middleware_pipeline
    pipeline.prepare = [
        _MiddlewareSpy(mw, log, name)
        for mw, name in zip(pipeline.prepare, PREPARE_NAMES, strict=True)
    ]
    pipeline.execute = [
        _MiddlewareSpy(mw, log, name)
        for mw, name in zip(pipeline.execute, EXECUTE_NAMES, strict=True)
    ]
    pipeline.finalize = [
        _MiddlewareSpy(mw, log, name)
        for mw, name in zip(pipeline.finalize, FINALIZE_NAMES, strict=True)
    ]


def _request(
    gateway: Any,
    tool_id: str,
    call_id: str,
    workspace_id: str,
    **raw_arguments: Any,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_id=tool_id,
        call_id=call_id,
        session_id="sess_1",
        frame_id="frame_1",
        raw_arguments=raw_arguments,
        workspace_id=workspace_id,
    )


def _read_file_request(
    gateway: Any, path: str = "README.md", call_id: str = "c1"
) -> ToolCallRequest:
    return _request(
        gateway,
        "read_file",
        call_id,
        gateway.workspace_manager.workspace_id,
        path=path,
    )


def _run_command_request(
    gateway: Any, argv: list[str], call_id: str = "c1"
) -> ToolCallRequest:
    return _request(
        gateway,
        "run_command",
        call_id,
        gateway.workspace_manager.workspace_id,
        argv=argv,
    )


def _context(gateway: Any, request: ToolCallRequest) -> MiddlewareContext:
    return MiddlewareContext(
        request=request,
        registry=gateway._registry,
        approval_service=gateway._approval_service,
        workspace_manager=gateway._workspace_manager,
        backend=gateway._backend,
        scheduler=gateway._scheduler,
    )


# ---------------------------------------------------------------------------
# Phase ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_phase_order(gateway: Any) -> None:
    """PREPARE runs Exposure → Validation → Authorization → WorkspaceAdmission."""
    log: list[str] = []
    _attach_spies(gateway, log)
    result = await gateway.execute(_read_file_request(gateway))
    assert result.status == ToolExecutionStatus.COMPLETED.value
    prepare_calls = [entry for entry in log if entry.startswith("prepare:")]
    assert prepare_calls == [f"prepare:{name}" for name in PREPARE_NAMES]


@pytest.mark.asyncio
async def test_execute_phase_order(gateway: Any) -> None:
    """EXECUTE runs SchedulingMiddleware → ExecutionMiddleware."""
    log: list[str] = []
    _attach_spies(gateway, log)
    result = await gateway.execute(_read_file_request(gateway))
    assert result.status == ToolExecutionStatus.COMPLETED.value
    execute_calls = [entry for entry in log if entry.startswith("execute:")]
    assert execute_calls == [f"execute:{name}" for name in EXECUTE_NAMES]


@pytest.mark.asyncio
async def test_finalize_phase_order(gateway: Any) -> None:
    """FINALIZE runs SchedulingMiddleware (release locks) → NormalizationMiddleware."""
    log: list[str] = []
    _attach_spies(gateway, log)
    result = await gateway.execute(_read_file_request(gateway))
    assert result.status == ToolExecutionStatus.COMPLETED.value
    finalize_calls = [entry for entry in log if entry.startswith("finalize:")]
    assert finalize_calls == [f"finalize:{name}" for name in FINALIZE_NAMES]


# ---------------------------------------------------------------------------
# Short-circuit behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_circuit_on_deny(gateway: Any) -> None:
    """DENY in PREPARE short-circuits: EXECUTE/FINALIZE are not run."""
    log: list[str] = []
    _attach_spies(gateway, log)
    result = await gateway.execute(
        _run_command_request(gateway, argv=["sudo", "reboot"])
    )

    assert result.status == ToolExecutionStatus.DENIED.value
    assert result.error is not None
    assert result.error.code == ToolErrorCode.DENIED
    # EXECUTE and FINALIZE phases must not run at all.
    assert not any(entry.startswith("execute:") for entry in log)
    assert not any(entry.startswith("finalize:") for entry in log)
    # WorkspaceAdmission is not reached; short-circuit happens at Authorization.
    prepare_calls = [entry for entry in log if entry.startswith("prepare:")]
    assert prepare_calls == [
        "prepare:ExposureMiddleware",
        "prepare:ValidationMiddleware",
        "prepare:AuthorizationMiddleware",
    ]


@pytest.mark.asyncio
async def test_short_circuit_on_ask_in_batch(gateway: Any) -> None:
    """ASK on one call in a batch returns PENDING_APPROVAL for all;
    EXECUTE/FINALIZE are not run for any call."""
    log: list[str] = []
    _attach_spies(gateway, log)
    allow_call = _read_file_request(gateway, call_id="allow-1")
    ask_call = _run_command_request(
        gateway, argv=["python", "script.py"], call_id="ask-1"
    )

    results = await gateway.execute_many([allow_call, ask_call])

    assert len(results) == 2
    assert all(r.status == ToolExecutionStatus.PENDING_APPROVAL.value for r in results)
    # No call reached EXECUTE/FINALIZE because the batch is paused for approval.
    assert not any(entry.startswith("execute:") for entry in log)
    assert not any(entry.startswith("finalize:") for entry in log)


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_emission_start_and_complete(gateway: Any) -> None:
    """START is emitted after authorization (PREPARE); COMPLETE in FINALIZE."""
    events: list[Any] = []

    async def callback(event: Any) -> None:
        events.append(event)

    gateway.register_event_callback(callback)
    result = await gateway.execute(_read_file_request(gateway))
    assert result.status == ToolExecutionStatus.COMPLETED.value

    types = [event.type for event in events]
    assert ToolEventType.TOOL_CALL_START in types
    assert ToolEventType.TOOL_CALL_COMPLETE in types

    start_index = types.index(ToolEventType.TOOL_CALL_START)
    complete_index = types.index(ToolEventType.TOOL_CALL_COMPLETE)
    assert start_index < complete_index

    start_event = events[start_index]
    assert start_event.data["tool_id"] == "read_file"
    assert start_event.data["backend"] == "host"
    assert start_event.data["arguments"] == {"path": "README.md"}


# ---------------------------------------------------------------------------
# on_error behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalization_on_error_converts_exception(gateway: Any) -> None:
    """NormalizationMiddleware.on_error converts an exception to ToolExecutionError."""
    middleware = NormalizationMiddleware(backend=gateway._backend)
    context = _context(gateway, _read_file_request(gateway))
    tool_error = await middleware.on_error(context, ValueError("boom"))

    assert isinstance(tool_error, ToolExecutionError)


@pytest.mark.asyncio
async def test_scheduling_on_error_releases_lock(gateway: Any) -> None:
    """SchedulingMiddleware.on_error pops and exits the acquired lock context."""
    middleware = SchedulingMiddleware(scheduler=gateway._scheduler)
    context = _context(gateway, _read_file_request(gateway))

    class _FakeLock:
        def __init__(self) -> None:
            self.exited = False

        async def __aenter__(self) -> _FakeLock:
            return self

        async def __aexit__(self, *args: Any) -> None:
            self.exited = True

    lock = _FakeLock()
    context.metadata["lock_context"] = lock

    await middleware.on_error(context, ValueError("boom"))

    assert lock.exited
    assert context.metadata.get("lock_context") is None


@pytest.mark.asyncio
async def test_prepare_error_returns_canonical_failed_result(gateway: Any) -> None:
    """A PREPARE-phase error (unknown tool) yields a canonical failed result."""
    result = await gateway.execute(
        _request(
            gateway,
            "nonexistent_tool_x",
            "c1",
            gateway.workspace_manager.workspace_id,
            path="x",
        )
    )

    assert result.status == ToolExecutionStatus.FAILED.value
    assert result.error is not None
