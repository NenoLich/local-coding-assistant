"""Unit tests for cooperative interruption of tool execution.

Covers the interrupt machinery across layers:
- ProcessSupervisor kills the managed process when execution is cancelled
- ExecutionMiddleware races the tool coroutine against the interrupt event
  and raises ToolInterruptedError (long-running tools die, quick tools that
  already finished keep their results)
- ToolGateway checkpoints between scheduler groups: in-flight groups finish,
  not-yet-started calls are reported as cancelled
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ProcessHandle,
    ProcessState,
    ToolCallRequest,
    ToolDecision,
    ToolDefinition,
    ToolEvent,
    ToolEventType,
    ToolExecutionStatus,
    ToolRiskClass,
    ValidatedToolCall,
)
from local_coding_assistant.tools_platform.exceptions import ToolInterruptedError
from local_coding_assistant.tools_platform.gateway import ToolGateway
from local_coding_assistant.tools_platform.middleware import (
    MiddlewareContext,
    MiddlewarePipeline,
)
from local_coding_assistant.tools_platform.middleware.execute import ExecutionMiddleware
from local_coding_assistant.tools_platform.supervisor import (
    ExecutionConfig,
    OutputLimits,
    ProcessSupervisor,
    TimeoutPolicy,
)
from local_coding_assistant.tools_platform.supervisor.transport import (
    ProcessTransport,
    TransportConfig,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class BlockingTransport(ProcessTransport):
    """Transport whose stdout blocks, simulating a long-running process."""

    def __init__(self) -> None:
        self.terminate_called = False
        self._running = True

    async def spawn(
        self,
        config: TransportConfig,
    ) -> tuple[ProcessHandle, AsyncIterator[bytes], AsyncIterator[bytes]]:
        async def stdout_iter() -> AsyncIterator[bytes]:
            # Never yields: consumer stays suspended mid-execution
            await asyncio.sleep(3600)
            yield b""  # pragma: no cover - unreachable

        async def stderr_iter() -> AsyncIterator[bytes]:
            await asyncio.sleep(3600)
            yield b""  # pragma: no cover - unreachable

        handle = ProcessHandle(
            handle_id="blocking-1",
            backend=BackendType.HOST,
            state=ProcessState.RUNNING,
            os_pid=4242,
        )
        return handle, stdout_iter(), stderr_iter()

    async def terminate(
        self,
        handle: ProcessHandle,
        grace: timedelta,
    ) -> None:
        self.terminate_called = True
        self._running = False

    async def wait_for_completion(self, handle: ProcessHandle) -> int | None:
        return 0 if not self._running else None

    def is_running(self, handle: ProcessHandle) -> bool:
        return self._running


def _tool_definition(tool_id: str) -> ToolDefinition:
    return ToolDefinition(
        id=tool_id,
        version="1.0.0",
        description=f"{tool_id} tool",
        input_schema={},
        output_schema={},
        risk_class=ToolRiskClass.LOW,
    )


class _InputModel(BaseModel):
    """Installed as the validated input model for stubbed calls."""


def _validated_call(request: ToolCallRequest) -> ValidatedToolCall:
    return ValidatedToolCall(
        request=request,
        tool_definition=_tool_definition(request.tool_id),
        input_model=_InputModel(),
    )


class StubPrepareMiddleware:
    """Bypasses prepare validation: every call is allowed."""

    async def prepare(self, context: MiddlewareContext) -> None:
        context.validated_call = _validated_call(context.request)
        context.decision = ToolDecision(
            outcome="allow",
            reason="allowed by stub",
            policy_rule="stub",
        )


class RecordingExecuteMiddleware:
    """Records executions; optionally triggers the interrupt event."""

    def __init__(self, interrupt_event: asyncio.Event | None = None) -> None:
        self.interrupt_event = interrupt_event
        self.executed: list[str] = []

    async def execute(self, context: MiddlewareContext) -> None:
        self.executed.append(context.request.call_id)
        context.metadata["execution_result"] = f"ran:{context.request.call_id}"
        if self.interrupt_event is not None:
            self.interrupt_event.set()


class StubScheduler:
    """Puts each tool in its own group to make group order deterministic."""

    async def classify_tool_calls(
        self, validated_calls: list[ValidatedToolCall]
    ) -> dict[str, list[ValidatedToolCall]]:
        groups: dict[str, list[ValidatedToolCall]] = {}
        for validated in validated_calls:
            key = validated.tool_definition.id
            groups.setdefault(key, []).append(validated)
        return groups


def _build_gateway(
    execute_middleware: Any,
    captured_events: list[ToolEvent],
) -> ToolGateway:
    async def capture(event: ToolEvent) -> None:
        captured_events.append(event)

    config_manager = MagicMock()
    config_manager.global_config.tools_platform.concurrency.max_parallel_reads = 10
    config_manager.global_config.tools_platform.concurrency.max_parallel_processes = 10

    gateway = ToolGateway(
        registry=MagicMock(),
        config_manager=config_manager,
    )
    gateway._middleware_pipeline = MiddlewarePipeline(
        prepare=[StubPrepareMiddleware()],
        execute=[execute_middleware],
        finalize=[],
    )
    gateway.register_event_callback(capture)
    return gateway


def _request(tool_id: str, call_id: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool_id=tool_id,
        call_id=call_id,
        session_id="sess_1",
        frame_id="frame_1",
        raw_arguments={"p": call_id},
        workspace_id="sess_1",
    )


# ---------------------------------------------------------------------------
# Supervisor: process kill on cancellation
# ---------------------------------------------------------------------------


class TestSupervisorInterrupt:
    @pytest.mark.asyncio
    async def test_cancelling_execution_terminates_process(self):
        """Cancelling the consumer task kills the managed process."""
        transport = BlockingTransport()
        supervisor = ProcessSupervisor(transport=transport)
        config = ExecutionConfig(
            argv=["sleep", "3600"],
            timeout_policy=TimeoutPolicy(
                hard_timeout=timedelta(seconds=300),
                cancellation_grace=timedelta(seconds=1),
            ),
            output_limits=OutputLimits(),
            backend=BackendType.HOST,
            transport=transport,
        )

        consumer = asyncio.create_task(anext(supervisor.execute(config)))
        # Let the generator spawn and block on output
        await asyncio.sleep(0.1)

        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

        assert transport.terminate_called is True, (
            "supervisor must terminate the process when its execution is "
            "cancelled, otherwise the subprocess is orphaned"
        )


# ---------------------------------------------------------------------------
# Execution middleware: interrupt race
# ---------------------------------------------------------------------------


class TestExecutionMiddlewareInterrupt:
    @staticmethod
    def _context(
        backend: Any, interrupt_event: asyncio.Event | None
    ) -> MiddlewareContext:
        context = MiddlewareContext(
            request=_request("tool_a", "c1"),
            registry=MagicMock(),
            approval_service=MagicMock(),
            workspace_manager=MagicMock(),
            backend=backend,
            scheduler=MagicMock(),
            interrupt_event=interrupt_event,
        )
        context.validated_call = _validated_call(context.request)
        return context

    @pytest.mark.asyncio
    async def test_interrupt_cancels_execution_and_raises(self):
        """A long-running tool is cancelled and ToolInterruptedError raised."""
        interrupt_event = asyncio.Event()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def long_running_tool():
            started.set()
            try:
                # Use a shorter sleep that can be cancelled
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        class OneShotRegistry:
            def get_implementation(self, tool_id: str):
                impl = MagicMock()
                impl.factory = (
                    lambda backend,
                    input_model,
                    session_id=None,
                    turn_id=None,
                    **kwargs: long_running_tool()
                )
                return impl

        context = self._context(MagicMock(), interrupt_event)
        context.registry = OneShotRegistry()
        middleware = ExecutionMiddleware(backend=MagicMock())

        task = asyncio.create_task(middleware.execute(context))
        await started.wait()
        # Small delay to ensure middleware sets up the race
        await asyncio.sleep(0.05)
        interrupt_event.set()

        with pytest.raises(ToolInterruptedError):
            await task

        # The tool coroutine itself observed the cancellation
        assert cancelled.is_set()
        assert context.metadata.get("execution_result") is None

    @pytest.mark.asyncio
    async def test_quick_tool_completes_before_interrupt(self):
        """A tool that finishes before the interrupt keeps its result."""
        event = asyncio.Event()

        async def quick_tool():
            await asyncio.sleep(0.05)
            return "done"

        class OneShotRegistry:
            def get_implementation(self, tool_id: str):
                impl = MagicMock()
                impl.factory = (
                    lambda backend,
                    input_model,
                    session_id=None,
                    turn_id=None,
                    **kwargs: quick_tool()
                )
                return impl

        context = self._context(MagicMock(), event)
        context.registry = OneShotRegistry()
        middleware = ExecutionMiddleware(backend=MagicMock())

        await middleware.execute(context)

        assert context.metadata["execution_result"] == "done"
        assert event.is_set() is False


# ---------------------------------------------------------------------------
# Gateway: scheduler-group checkpoints
# ---------------------------------------------------------------------------


class TestGatewayGroupCheckpoints:
    @pytest.mark.asyncio
    async def test_interrupt_during_first_group_skips_second_group(self):
        """In-flight group finishes; not-yet-started calls are cancelled."""
        interrupt_event = asyncio.Event()
        events: list[ToolEvent] = []
        execute_mw = RecordingExecuteMiddleware(interrupt_event=interrupt_event)
        gateway = _build_gateway(execute_mw, events)

        requests = [_request("tool_a", "c1"), _request("tool_b", "c2")]
        results = await gateway.execute_many(requests, interrupt_event=interrupt_event)

        # Results stay aligned with request order
        assert [r.status for r in results] == [
            "completed",
            ToolExecutionStatus.CANCELLED.value,
        ]
        assert execute_mw.executed == ["c1"]

        cancelled_events = [
            e for e in events if e.type == ToolEventType.TOOL_CALL_CANCELLED
        ]
        assert len(cancelled_events) == 1
        assert cancelled_events[0].call_id == "c2"
        assert (
            cancelled_events[0].data["reason"] == "Interrupted before execution started"
        )

    @pytest.mark.asyncio
    async def test_pre_set_interrupt_cancels_everything(self):
        """No call executes when the interrupt is already set."""
        interrupt_event = asyncio.Event()
        interrupt_event.set()
        events: list[ToolEvent] = []
        execute_mw = RecordingExecuteMiddleware(interrupt_event=None)
        gateway = _build_gateway(execute_mw, events)

        requests = [_request("tool_a", "c1"), _request("tool_b", "c2")]
        results = await gateway.execute_many(requests, interrupt_event=interrupt_event)

        assert execute_mw.executed == []
        assert all(r.status == ToolExecutionStatus.CANCELLED.value for r in results)

    @pytest.mark.asyncio
    async def test_no_interrupt_event_runs_all_groups(self):
        """Baseline: without an interrupt event everything executes."""
        events: list[ToolEvent] = []
        execute_mw = RecordingExecuteMiddleware(interrupt_event=None)
        gateway = _build_gateway(execute_mw, events)

        requests = [_request("tool_a", "c1"), _request("tool_b", "c2")]
        results = await gateway.execute_many(requests)

        assert execute_mw.executed == ["c1", "c2"]
        assert all(r.status == "completed" for r in results)
