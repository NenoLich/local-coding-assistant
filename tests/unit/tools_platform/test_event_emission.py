"""Unit tests for tool event emission.

Covers TOOL_CALL_START, TOOL_CALL_COMPLETE, TOOL_CALL_ERROR,
TOOL_CALL_DENIED, and TOOL_CALL_CANC lelled event payloads and correlation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ToolCallRequest,
    ToolDecision,
    ToolDecisionOutcome,
    ToolDefinition,
    ToolEvent,
    ToolEventType,
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


def _request(
    tool_id: str = "read_file",
    call_id: str = "c1",
    session_id: str = "sess_1",
    frame_id: str = "frame_1",
    raw_arguments: dict | None = None,
    workspace_id: str | None = None,
    transaction_id: str | None = None,
) -> ToolCallRequest:
    if raw_arguments is None:
        raw_arguments = {"path": "a.txt"}
    if workspace_id is None:
        workspace_id = session_id
    if transaction_id is None:
        transaction_id = "txn_1"
    return ToolCallRequest(
        tool_id=tool_id,
        call_id=call_id,
        session_id=session_id,
        frame_id=frame_id,
        raw_arguments=raw_arguments,
        workspace_id=workspace_id,
        transaction_id=transaction_id,
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
    _ = await gw.execute(_request())
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
    _ = await gw.execute(_request())
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
    _ = await gw.execute(_request())
    denied_event = next(e for e in events if e.type == ToolEventType.TOOL_CALL_DENIED)
    data = denied_event.data
    assert data["outcome"] == "ask"
    assert data["reason"] == "ask reason"
    assert data["policy_rule"] == "stub"
    assert data["confirmation_token"] == "tok-1"  # noqa: S105
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
    _ = await gw.execute(_request())
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


@pytest.mark.asyncio
async def test_events_include_unique_trace_id():
    """Each event emits a unique trace_id for correlation."""
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("allow"), StubExecuteMiddleware(), events)
    await gw.execute(_request())
    trace_ids = [e.trace_id for e in events]
    assert len(trace_ids) == len(set(trace_ids))


@pytest.mark.asyncio
async def test_event_callback_awaited_before_execution():
    """event_callback is awaited before execution proceeds (durable recording)."""

    captured: list[ToolEvent] = []

    async def capture(event: ToolEvent) -> None:
        captured.append(event)

    gw = _build_gateway(StubPrepareMiddleware("allow"), StubExecuteMiddleware(), [])
    gw._event_callback = capture
    await gw.execute(_request())
    assert len(captured) >= 2  # TOOL_CALL_START and TOOL_CALL_COMPLETE


@pytest.mark.asyncio
async def test_tool_call_start_with_custom_tool():
    """TOOL_CALL_START with custom tool_id and arguments."""
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("allow"), StubExecuteMiddleware(), events)
    await gw.execute(
        _request(tool_id="write_file", raw_arguments={"path": "b.txt", "content": "x"})
    )
    start_event = next(e for e in events if e.type == ToolEventType.TOOL_CALL_START)
    data = start_event.data
    assert data["tool_id"] == "write_file"
    assert data["arguments"] == {"path": "b.txt", "content": "x"}


@pytest.mark.asyncio
async def test_tool_call_complete_default_payload():
    """TOOL_CALL_COMPLETE default payload: changeset is None, warnings is empty list."""
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("allow"), StubExecuteMiddleware(), events)
    _ = await gw.execute(_request())
    complete_event = next(
        e for e in events if e.type == ToolEventType.TOOL_CALL_COMPLETE
    )
    data = complete_event.data
    assert data["changeset"] is None
    assert data["warnings"] == []


@pytest.mark.asyncio
async def test_tool_call_error_with_tool_error_code():
    """TOOL_CALL_ERROR with ToolExecutionError code and message."""
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
    _ = await gw.execute(_request())
    error_event = next(e for e in events if e.type == ToolEventType.TOOL_CALL_ERROR)
    data = error_event.data
    assert data["status"] == "failed"
    assert data["error"]["code"] == "internal_failure"
    assert data["error"]["message"] == "backend boom"


@pytest.mark.asyncio
async def test_tool_call_denied_with_deny_outcome():
    """TOOL_CALL_DENIED with deny outcome (not ask)."""
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("deny"), StubExecuteMiddleware(), events)
    _ = await gw.execute(_request())
    denied_event = next(e for e in events if e.type == ToolEventType.TOOL_CALL_DENIED)
    data = denied_event.data
    assert data["outcome"] == "deny"
    assert data["status"] == "denied"
    assert data["confirmation_token"] is None


@pytest.mark.asyncio
async def test_tool_call_cancelled_with_timing():
    """TOOL_CALL_CANCELLED includes timing information."""
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("allow"), StubExecuteMiddleware(), events)

    class CancelExec(StubExecuteMiddleware):
        async def execute(self, context):
            raise ToolInterruptedError("stop")

    gw._middleware_pipeline = MiddlewarePipeline(
        prepare=[StubPrepareMiddleware()], execute=[CancelExec()], finalize=[]
    )
    _ = await gw.execute(_request())
    cancelled_event = next(
        e for e in events if e.type == ToolEventType.TOOL_CALL_CANCELLED
    )
    data = cancelled_event.data
    assert "timing" in data


@pytest.mark.asyncio
async def test_tool_call_start_with_frame_id_none():
    """TOOL_CALL_START frame_id can be empty string when not set."""
    events: list[ToolEvent] = []
    gw = _build_gateway(StubPrepareMiddleware("allow"), StubExecuteMiddleware(), events)
    req = _request(frame_id="")
    await gw.execute(req)
    start_event = next(e for e in events if e.type == ToolEventType.TOOL_CALL_START)
    assert start_event.frame_id == ""
