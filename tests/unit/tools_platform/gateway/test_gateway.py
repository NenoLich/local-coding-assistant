"""Unit tests for ToolGateway.

Covers the gateway orchestration: single execute, execute_many, execute_stream,
event emission, and interruption checkpoints.
"""

import asyncio
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    TimeoutProfile,
    ToolCallRequest,
    ToolDecision,
    ToolDecisionOutcome,
    ToolDefinition,
    ToolErrorCode,
    ToolEvent,
    ToolEventType,
    ToolExecutionStatus,
    ToolOutputChunk,
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
from local_coding_assistant.tools_platform.middleware.finalize import (
    NormalizationMiddleware,
)


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
    pass


def _validated_call(request: ToolCallRequest) -> ValidatedToolCall:
    return ValidatedToolCall(
        request=request,
        tool_definition=_tool_definition(request.tool_id),
        input_model=_InputModel(),
    )


def _request(tool_id: str = "read_file", call_id: str = "c1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_id=tool_id,
        call_id=call_id,
        session_id="sess_1",
        frame_id="frame_1",
        raw_arguments={"path": "a.txt"},
        workspace_id="sess_1",
        transaction_id="txn_1",
    )


class StubPrepareMiddleware:
    def __init__(
        self, decision_outcome: str = "allow", raise_exc: Exception | None = None
    ):
        self.decision_outcome = decision_outcome
        self.raise_exc = raise_exc
        self.ask_token: str | None = "tok-1" if decision_outcome == "ask" else None

    async def prepare(self, context: MiddlewareContext) -> None:
        if self.raise_exc is not None:
            raise self.raise_exc
        context.validated_call = _validated_call(context.request)
        outcome = ToolDecisionOutcome(self.decision_outcome)
        context.decision = ToolDecision(
            outcome=outcome,
            reason=f"{self.decision_outcome} reason",
            policy_rule="stub",
            confirmation_token=self.ask_token,
        )


class StubExecuteMiddleware:
    def __init__(self, raise_exc: Exception | None = None, stream: bool = False):
        self.raise_exc = raise_exc
        self.stream = stream

    async def execute(self, context: MiddlewareContext) -> None:
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.stream:

            async def gen():
                yield ToolOutputChunk(
                    stream_id=context.request.call_id,
                    sequence=0,
                    channel="status",
                    payload={"status": "completed"},
                )

            context.metadata["execution_stream"] = gen()
        else:
            context.metadata["execution_result"] = "out"


class StubScheduler:
    async def classify_tool_calls(self, validated_calls):
        groups = {}
        for vc in validated_calls:
            groups.setdefault("group_0", []).append(vc)
        return groups


def _build_gateway(
    prepare: StubPrepareMiddleware,
    execute: StubExecuteMiddleware,
    events: list[ToolEvent],
) -> ToolGateway:
    backend = MagicMock()
    backend.get_backend_type.return_value = "host"

    async def capture(event):
        events.append(event)

    gateway = ToolGateway.__new__(ToolGateway)
    gateway._registry = MagicMock()
    gateway._config_manager = MagicMock()
    gateway._event_callback = None
    gateway._backend = backend
    gateway._workspace_manager = MagicMock()
    gateway._approval_service = MagicMock()
    gateway._scheduler = StubScheduler()
    gateway._middleware_pipeline = MiddlewarePipeline(
        prepare=[prepare],
        execute=[execute],
        finalize=[],
    )
    gateway.register_event_callback(capture)
    # minimal helpers
    gateway._backend_type = lambda: BackendType.HOST
    return gateway


@pytest.mark.asyncio
async def test_execute_single_allowed():
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("allow"), StubExecuteMiddleware(), events)
    result = await gw.execute(_request())
    assert result.status == "completed"
    assert result.output == "out"
    assert result.backend == BackendType.HOST
    assert "elapsed_seconds" in result.timing
    types = [e.type for e in events]
    assert types == [ToolEventType.TOOL_CALL_START, ToolEventType.TOOL_CALL_COMPLETE]


@pytest.mark.asyncio
async def test_execute_single_denied():
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("deny"), StubExecuteMiddleware(), events)
    result = await gw.execute(_request())
    assert result.status == "denied"
    assert result.error.code.value == "denied"
    assert any(e.type == ToolEventType.TOOL_CALL_DENIED for e in events)


@pytest.mark.asyncio
async def test_execute_single_ask():
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("ask"), StubExecuteMiddleware(), events)
    result = await gw.execute(_request())
    assert result.status == "denied"
    assert result.error.code.value == "approval_required"
    assert result.error.details.get("confirmation_token") == "tok-1"
    assert any(e.type == ToolEventType.TOOL_CALL_DENIED for e in events)


@pytest.mark.asyncio
async def test_execute_single_validation_error():
    events: list[ToolEvent] = []
    from local_coding_assistant.tools_platform.exceptions import ToolValidationError

    gw = _build_gateway(
        StubPrepareMiddleware(raise_exc=ToolValidationError("bad tool")),
        StubExecuteMiddleware(),
        events,
    )
    result = await gw.execute(_request())
    assert result.status == "failed"
    assert result.error.code.value == "invalid_arguments"
    assert any(e.type == ToolEventType.TOOL_CALL_ERROR for e in events)


@pytest.mark.asyncio
async def test_execute_single_interrupted():
    events: list[ToolEvent] = []
    gw = _build_gateway(
        StubPrepareMiddleware("allow"),
        StubExecuteMiddleware(raise_exc=ToolInterruptedError("stop")),
        events,
    )
    result = await gw.execute(_request())
    assert result.status == ToolExecutionStatus.CANCELLED.value
    assert result.error.code.value == "interrupted"
    assert any(e.type == ToolEventType.TOOL_CALL_CANCELLED for e in events)


@pytest.mark.asyncio
async def test_execute_single_unexpected():
    events: list[ToolEvent] = []
    gw = _build_gateway(
        StubPrepareMiddleware("allow"),
        StubExecuteMiddleware(raise_exc=RuntimeError("boom")),
        events,
    )
    result = await gw.execute(_request())
    assert result.status == "failed"
    assert result.error is not None
    assert any(e.type == ToolEventType.TOOL_CALL_ERROR for e in events)


@pytest.mark.asyncio
async def test_execute_many_all_allowed():
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("allow"), StubExecuteMiddleware(), events)
    reqs = [_request(call_id="c1"), _request(call_id="c2")]
    results = await gw.execute_many(reqs)
    assert [r.status for r in results] == ["completed", "completed"]
    assert [r.output for r in results] == ["out", "out"]


@pytest.mark.asyncio
async def test_execute_many_one_ask_pauses_all():
    events: list[ToolEvent] = []

    # First call allow, second ask -> batch pauses, both results become
    # PENDING_APPROVAL (nothing executes), the ASK call keeps its decision
    # with the confirmation token for the approval loop,
    # TOOL_CALL_DENIED for each
    class PerCallPrepare(StubPrepareMiddleware):
        def __init__(self):
            self._outcomes = {"c1": "allow", "c2": "ask"}

        async def prepare(self, context):
            outcome = self._outcomes.get(context.request.call_id, "allow")
            context.validated_call = _validated_call(context.request)
            context.decision = ToolDecision(
                outcome=ToolDecisionOutcome(outcome),
                reason=f"{outcome} reason",
                policy_rule="stub",
                confirmation_token="tok-1" if outcome == "ask" else None,
            )

    gw = _build_gateway(PerCallPrepare(), StubExecuteMiddleware(), events)
    reqs = [_request(call_id="c1"), _request(call_id="c2")]
    results = await gw.execute_many(reqs)
    assert results[0].status == ToolExecutionStatus.PENDING_APPROVAL.value
    assert results[1].status == ToolExecutionStatus.PENDING_APPROVAL.value
    assert results[1].decision is not None
    assert results[1].decision.outcome == ToolDecisionOutcome.ASK
    assert results[1].decision.confirmation_token == "tok-1"
    denied_events = [e for e in events if e.type == ToolEventType.TOOL_CALL_DENIED]
    assert len(denied_events) == 2


@pytest.mark.asyncio
async def test_execute_stream_allowed():
    events: list[ToolEvent] = []
    gw = _build_gateway(
        StubPrepareMiddleware("allow"), StubExecuteMiddleware(stream=True), events
    )
    chunks = []
    async for ch in gw.execute_stream(_request()):
        chunks.append(ch)
    assert len(chunks) >= 1
    assert any(e.type == ToolEventType.TOOL_CALL_START for e in events)
    assert any(e.type == ToolEventType.TOOL_CALL_COMPLETE for e in events)


@pytest.mark.asyncio
async def test_execute_stream_ask():
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("ask"), StubExecuteMiddleware(), events)
    chunks = []
    async for ch in gw.execute_stream(_request()):
        chunks.append(ch)
    assert chunks[0].channel == "error"
    assert any(e.type == ToolEventType.TOOL_CALL_DENIED for e in events)


@pytest.mark.asyncio
async def test_tool_call_start_event_payload():
    """TOOL_CALL_START carries tool_id, arguments, backend, workspace, txn."""
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("allow"), StubExecuteMiddleware(), events)
    await gw.execute(_request())
    start_event = next(e for e in events if e.type == ToolEventType.TOOL_CALL_START)
    data = start_event.data
    assert data["tool_id"] == "read_file"
    assert data["arguments"] == {"path": "a.txt"}
    assert data["backend"] == "host"
    assert data["workspace_id"] == "sess_1"
    assert data["transaction_id"] == "txn_1"
    assert start_event.frame_id == "frame_1"
    assert data["risk_class"] == "low"


@pytest.mark.asyncio
async def test_tool_call_complete_event_payload():
    """TOOL_CALL_COMPLETE carries full outcome: status, output, error, etc."""
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("allow"), StubExecuteMiddleware(), events)
    result = await gw.execute(_request())
    complete_event = next(
        e for e in events if e.type == ToolEventType.TOOL_CALL_COMPLETE
    )
    data = complete_event.data
    assert data["status"] == "completed"
    assert data["output"] == "out"
    assert data["timing"]["elapsed_seconds"] >= 0
    assert data["backend"] == "host"
    assert data["changeset"] is None
    assert data["warnings"] == []
    assert data["decision"] is not None
    assert data["arguments"] == {"path": "a.txt"}


@pytest.mark.asyncio
async def test_tool_call_error_event_payload():
    """TOOL_CALL_ERROR carries status=failed, error details, timing, backend."""
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("allow"), StubExecuteMiddleware(), events)

    class FailExec(StubExecuteMiddleware):
        def __init__(self):
            super().__init__()
            self._fail = True

        async def execute(self, context):
            if self._fail:
                raise RuntimeError("backend boom")
            await super().execute(context)

    gw._middleware_pipeline = MiddlewarePipeline(
        prepare=[StubPrepareMiddleware()], execute=[FailExec()], finalize=[]
    )
    result = await gw.execute(_request())
    error_event = next(e for e in events if e.type == ToolEventType.TOOL_CALL_ERROR)
    data = error_event.data
    assert data["status"] == "failed"
    assert data["output"] is None
    err = data["error"]
    assert err["code"] == "internal_failure"
    assert err["message"] == "backend boom"
    assert "timing" in data
    assert data["backend"] == "host"
    assert data["arguments"] == {"path": "a.txt"}


@pytest.mark.asyncio
async def test_tool_call_denied_event_payload():
    """TOOL_CALL_DENIED carries outcome, reason, policy_rule, token, arguments."""
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("ask"), StubExecuteMiddleware(), events)
    result = await gw.execute(_request())
    denied_event = next(e for e in events if e.type == ToolEventType.TOOL_CALL_DENIED)
    data = denied_event.data
    assert data["outcome"] == "ask"
    assert data["reason"] == "ask reason"
    assert data["policy_rule"] == "stub"
    assert data["confirmation_token"] == "tok-1"
    assert data["arguments"] == {"path": "a.txt"}


@pytest.mark.asyncio
async def test_tool_call_cancelled_event_payload():
    """TOOL_CALL_CANCELLED carries reason, timing, backend, error, arguments."""
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("allow"), StubExecuteMiddleware(), events)

    class CancelExec(StubExecuteMiddleware):
        async def execute(self, context):
            raise ToolInterruptedError("stop execution")

    gw._middleware_pipeline = MiddlewarePipeline(
        prepare=[StubPrepareMiddleware()], execute=[CancelExec()], finalize=[]
    )
    result = await gw.execute(_request())
    cancelled_event = next(
        e for e in events if e.type == ToolEventType.TOOL_CALL_CANCELLED
    )
    data = cancelled_event.data
    assert data["tool_id"] == "read_file"
    assert data["status"] == "cancelled"
    assert data["reason"] == "stop execution"
    assert data["backend"] == "host"
    err = data["error"]
    assert err["code"] == "interrupted"
    assert data["arguments"] == {"path": "a.txt"}


class SplitScheduler:
    """Scheduler that splits calls into two groups by call index."""

    async def classify_tool_calls(self, validated_calls):
        groups = {"group_0": [], "group_1": []}
        for i, vc in enumerate(validated_calls):
            groups[f"group_{i % 2}"].append(vc)
        return groups


class SlowFirstExecuteMiddleware:
    """Executes the first call slowly so an interrupt can land mid-group."""

    def __init__(self, interrupt_event: asyncio.Event) -> None:
        self.interrupt_event = interrupt_event
        self.started = asyncio.Event()

    async def execute(self, context: MiddlewareContext) -> None:
        if context.request.call_id == "c1":
            self.started.set()
            await asyncio.sleep(0.2)
        context.metadata["execution_result"] = f"out:{context.request.call_id}"


@pytest.mark.asyncio
async def test_execute_many_interrupt_before_any_group():
    """Interrupt set before any group starts: all calls cancelled, none execute."""
    interrupt_event = asyncio.Event()
    interrupt_event.set()
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("allow"), StubExecuteMiddleware(), events)
    reqs = [_request(call_id="c1"), _request(call_id="c2")]
    results = await gw.execute_many(reqs, interrupt_event=interrupt_event)
    assert [r.status for r in results] == [
        ToolExecutionStatus.CANCELLED.value,
        ToolExecutionStatus.CANCELLED.value,
    ]
    cancelled = [e for e in events if e.type == ToolEventType.TOOL_CALL_CANCELLED]
    assert len(cancelled) == 2
    assert all(
        e.data["reason"] == "Interrupted before execution started" for e in cancelled
    )


@pytest.mark.asyncio
async def test_execute_many_interrupt_during_first_group_skips_second():
    """Interrupt during group 0: group 0 completes, group 1 cancelled."""
    interrupt_event = asyncio.Event()
    events: list[ToolEvent] = []
    execute_mw = SlowFirstExecuteMiddleware(interrupt_event)
    gw = _build_gateway(StubPrepareMiddleware("allow"), execute_mw, events)
    gw._scheduler = SplitScheduler()
    reqs = [_request(call_id="c1"), _request(call_id="c2")]

    async def set_interrupt() -> None:
        await execute_mw.started.wait()
        interrupt_event.set()

    interrupt_task = asyncio.create_task(set_interrupt())
    results = await gw.execute_many(reqs, interrupt_event=interrupt_event)
    await interrupt_task

    assert results[0].status == "completed"
    assert results[0].output == "out:c1"
    assert results[1].status == ToolExecutionStatus.CANCELLED.value
    cancelled = [e for e in events if e.type == ToolEventType.TOOL_CALL_CANCELLED]
    assert len(cancelled) == 1
    assert cancelled[0].call_id == "c2"


def _timeout_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        id="slow_tool",
        version="1.0.0",
        description="slow tool",
        input_schema={},
        output_schema={},
        risk_class=ToolRiskClass.LOW,
        timeout_profile=TimeoutProfile(
            soft_timeout=timedelta(seconds=1),
            hard_timeout=timedelta(seconds=0.1),
            idle_timeout=timedelta(seconds=30),
            heartbeat_interval=timedelta(seconds=5),
            cancellation_grace=timedelta(seconds=5),
        ),
    )


class SlowToolRegistry:
    """Registry whose tool impl sleeps longer than the configured timeout."""

    def get_implementation(self, tool_id: str):
        impl = MagicMock()
        impl.output_model = None

        async def slow_tool(backend, input_model, session_id=None, turn_id=None):
            await asyncio.sleep(3600)
            return "done"

        impl.factory = (
            lambda backend, input_model, session_id=None, turn_id=None: slow_tool(
                backend, input_model, session_id, turn_id
            )
        )
        return impl


class StreamSlowRegistry:
    """Registry whose streaming tool impl sleeps, simulating a long stream."""

    def get_implementation(self, tool_id: str):
        impl = MagicMock()
        impl.output_model = None

        async def slow_stream(backend, input_model, session_id=None, turn_id=None):
            await asyncio.sleep(3600)
            return "done"

        impl.factory = (
            lambda backend, input_model, session_id=None, turn_id=None: slow_stream(
                backend, input_model, session_id, turn_id
            )
        )
        return impl


class TimeoutPrepareMiddleware:
    def __init__(self, tool_definition: ToolDefinition) -> None:
        self.tool_definition = tool_definition

    async def prepare(self, context: MiddlewareContext) -> None:
        context.validated_call = ValidatedToolCall(
            request=context.request,
            tool_definition=self.tool_definition,
            input_model=_InputModel(),
        )
        context.decision = ToolDecision(
            outcome=ToolDecisionOutcome.ALLOW,
            reason="allow",
            policy_rule="stub",
        )


def _build_real_execute_gateway(
    registry,
    backend,
    prepare_mw,
    events: list[ToolEvent],
) -> ToolGateway:
    gw = ToolGateway.__new__(ToolGateway)
    gw._registry = registry
    gw._config_manager = MagicMock()
    gw._event_callback = None
    gw._backend = backend
    gw._workspace_manager = MagicMock()
    gw._approval_service = MagicMock()
    gw._scheduler = StubScheduler()
    gw._middleware_pipeline = MiddlewarePipeline(
        prepare=[prepare_mw],
        execute=[ExecutionMiddleware(backend=backend)],
        finalize=[NormalizationMiddleware(backend=backend)],
    )

    async def capture(event: ToolEvent) -> None:
        events.append(event)

    gw.register_event_callback(capture)
    gw._backend_type = lambda: BackendType.HOST
    return gw


@pytest.mark.asyncio
async def test_execute_stream_interrupt():
    """Interrupt during stream execution yields a cancelled status chunk + CANCELLED event."""
    interrupt_event = asyncio.Event()
    interrupt_event.set()
    events: list[ToolEvent] = []
    backend = MagicMock()
    backend.get_backend_type.return_value = "host"
    gw = _build_real_execute_gateway(
        StreamSlowRegistry(), backend, StubPrepareMiddleware("allow"), events
    )
    chunks = [
        c
        async for c in gw.execute_stream(
            _request(tool_id="stream_tool"), interrupt_event=interrupt_event
        )
    ]
    assert len(chunks) == 1
    assert chunks[0].channel == "status"
    assert chunks[0].payload["status"] == ToolExecutionStatus.CANCELLED.value
    assert "error" in chunks[0].payload
    assert any(e.type == ToolEventType.TOOL_CALL_CANCELLED for e in events)


@pytest.mark.asyncio
async def test_execute_respects_timeout_profile_hard_timeout():
    """Tool timeout_profile hard_timeout exceeded: execution fails with timeout error."""
    events: list[ToolEvent] = []
    backend = MagicMock()
    backend.get_backend_type.return_value = "host"
    gw = _build_real_execute_gateway(
        SlowToolRegistry(),
        backend,
        TimeoutPrepareMiddleware(_timeout_tool_definition()),
        events,
    )
    result = await gw.execute(_request(tool_id="slow_tool"))
    assert result.status == "failed"
    assert result.error.code == ToolErrorCode.TIMEOUT
    assert result.error.retryable is True
    assert any(e.type == ToolEventType.TOOL_CALL_ERROR for e in events)
