"""Unit tests for ExecutionMiddleware.

Covers backend execution, timeout handling, interrupt support, and streaming
described in plan_docs/2026_09/tools_platform_tests.md.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    TimeoutProfile,
    ToolCallRequest,
    ToolDefinition,
    ToolOutputChunk,
    ValidatedToolCall,
)
from local_coding_assistant.tools_platform.definitions import ToolImplementation
from local_coding_assistant.tools_platform.exceptions import ToolInterruptedError
from local_coding_assistant.tools_platform.middleware import MiddlewareContext
from local_coding_assistant.tools_platform.middleware.execute.execution import (
    ExecutionMiddleware,
)
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.workspace import WorkspaceManager
from local_coding_assistant.tools_platform.workspace.models import PathPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockInputModel(BaseModel):
    """Mock input model for testing."""

    value: str = "test"


class MockOutputModel(BaseModel):
    """Mock output model for testing."""

    result: str = "success"


def _mock_tool_factory(
    backend: Any,
    input_model: MockInputModel,
    session_id: str,
    turn_id: str | None = None,
) -> Any:
    """Mock tool factory that returns a coroutine or direct result."""
    return MockOutputModel(result=f"{input_model.value}-{session_id}")


def _async_tool_factory(
    backend: Any,
    input_model: MockInputModel,
    session_id: str,
    turn_id: str | None = None,
) -> Any:
    """Mock tool factory that returns an async coroutine."""

    async def _run() -> MockOutputModel:
        await asyncio.sleep(0.01)
        return MockOutputModel(result=f"{input_model.value}-{session_id}")

    return _run()


def _slow_tool_factory(
    backend: Any,
    input_model: MockInputModel,
    session_id: str,
    turn_id: str | None = None,
) -> Any:
    """Mock tool factory that returns a slow async coroutine for timeout tests."""

    async def _run() -> MockOutputModel:
        await asyncio.sleep(10)
        return MockOutputModel(result=f"{input_model.value}-{session_id}")

    return _run()


def _tool_definition(
    tool_id: str,
    timeout_profile: TimeoutProfile | None = None,
) -> ToolDefinition:
    return ToolDefinition(
        id=tool_id,
        version="1.0.0",
        description=f"{tool_id} tool",
        input_schema={},
        output_schema={},
        risk_class="low",
        timeout_profile=timeout_profile,
    )


def _validated_call(
    tool_id: str,
    call_id: str = "c1",
    session_id: str = "sess_1",
    deadline_policy: timedelta | None = None,
    timeout_profile: TimeoutProfile | None = None,
) -> ValidatedToolCall:
    request = ToolCallRequest(
        tool_id=tool_id,
        call_id=call_id,
        session_id=session_id,
        frame_id="frame_1",
        raw_arguments={"value": "test"},
        workspace_id="sess_1",
        deadline_policy=deadline_policy,
        turn_id="turn_1",
    )
    return ValidatedToolCall(
        request=request,
        tool_definition=_tool_definition(tool_id, timeout_profile),
        input_model=MockInputModel(value="test"),
    )


def _context(
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    backend: Any,
    interrupt_event: asyncio.Event | None = None,
) -> MiddlewareContext:
    ctx = MiddlewareContext(
        request=_validated_call("test_tool"),
        registry=registry,
        approval_service=MagicMock(),
        workspace_manager=workspace_manager,
        backend=backend,
        scheduler=MagicMock(),
        interrupt_event=interrupt_event,
    )
    return ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """A temporary workspace directory used as the tool workspace root."""
    return tmp_path


@pytest.fixture
def path_policy(tmp_workspace: Path) -> PathPolicy:
    """Permissive path policy allowing every path under the workspace."""
    return PathPolicy(
        workspace_root=tmp_workspace,
        allowed_patterns=["**"],
        denied_patterns=[],
        allow_symlinks=False,
        max_depth=None,
    )


@pytest.fixture
def workspace_manager(tmp_workspace: Path, path_policy: PathPolicy) -> WorkspaceManager:
    """WorkspaceManager backed by the temporary workspace."""
    return WorkspaceManager(
        workspace_id="test-ws",
        workspace_root=tmp_workspace,
        path_policy=path_policy,
        config_manager=None,
    )


@pytest.fixture
def registry() -> ToolRegistry:
    """ToolRegistry with a mock tool registered."""
    registry = ToolRegistry()
    tool_def = _tool_definition("test_tool")
    tool_impl = ToolImplementation(
        factory=_mock_tool_factory,
        input_model=MockInputModel,
        output_model=MockOutputModel,
    )
    registry.register(tool_def, tool_impl)
    return registry


@pytest.fixture
def mock_backend() -> MagicMock:
    """Mock execution backend."""
    backend = MagicMock()
    backend.get_backend_type.return_value = BackendType.HOST
    return backend


@pytest.fixture
def execution_middleware(mock_backend: MagicMock) -> ExecutionMiddleware:
    """ExecutionMiddleware instance for tests."""
    return ExecutionMiddleware(backend=mock_backend)


# ---------------------------------------------------------------------------
# execute calls tool factory with backend, input_model, session_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_calls_tool_factory_with_backend_input_session(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute calls tool factory with backend, input_model, session_id."""
    # Register a tool with a spy factory
    spy_factory = MagicMock(wraps=_mock_tool_factory)
    tool_def = _tool_definition("spy_tool")
    tool_impl = ToolImplementation(
        factory=spy_factory,
        input_model=MockInputModel,
        output_model=MockOutputModel,
    )
    registry.register(tool_def, tool_impl)

    ctx = _context(registry, workspace_manager, mock_backend)
    ctx.validated_call = _validated_call("spy_tool", session_id="test_session")

    await execution_middleware.execute(ctx)

    assert spy_factory.call_count == 1
    call_args = spy_factory.call_args.args
    call_kwargs = spy_factory.call_args.kwargs
    # backend and input_model are positional, session_id is keyword
    assert len(call_args) == 2
    assert call_args[0] is mock_backend
    assert isinstance(call_args[1], MockInputModel)
    assert "session_id" in call_kwargs
    assert call_kwargs["session_id"] == "test_session"


# ---------------------------------------------------------------------------
# execute awaits result with timeout from tool timeout_profile or request deadline_policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_uses_request_deadline_policy(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute awaits result with timeout from request deadline_policy."""
    # Register a slow tool
    tool_def = _tool_definition("slow_tool")
    tool_impl = ToolImplementation(
        factory=_slow_tool_factory,
        input_model=MockInputModel,
        output_model=MockOutputModel,
    )
    registry.register(tool_def, tool_impl)

    ctx = _context(registry, workspace_manager, mock_backend)
    ctx.validated_call = _validated_call(
        "slow_tool",
        deadline_policy=timedelta(seconds=0.1),
    )

    with pytest.raises(asyncio.TimeoutError):
        await execution_middleware.execute(ctx)


@pytest.mark.asyncio
async def test_execute_uses_tool_timeout_profile_hard_timeout(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute uses tool timeout_profile hard_timeout when no deadline_policy."""
    # Register a slow tool with a short timeout profile
    timeout_profile = TimeoutProfile(hard_timeout=timedelta(seconds=0.1))
    tool_def = _tool_definition("slow_tool", timeout_profile=timeout_profile)
    tool_impl = ToolImplementation(
        factory=_slow_tool_factory,
        input_model=MockInputModel,
        output_model=MockOutputModel,
    )
    registry.register(tool_def, tool_impl)

    ctx = _context(registry, workspace_manager, mock_backend)
    ctx.validated_call = _validated_call("slow_tool", timeout_profile=timeout_profile)

    with pytest.raises(asyncio.TimeoutError):
        await execution_middleware.execute(ctx)


@pytest.mark.asyncio
async def test_execute_without_timeout_awaits_indefinitely(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute with no timeout awaits result indefinitely (or until tool completes)."""
    # Register an async tool that completes quickly
    tool_def = _tool_definition("async_tool")
    tool_impl = ToolImplementation(
        factory=_async_tool_factory,
        input_model=MockInputModel,
        output_model=MockOutputModel,
    )
    registry.register(tool_def, tool_impl)

    ctx = _context(registry, workspace_manager, mock_backend)
    ctx.validated_call = _validated_call("async_tool")

    await execution_middleware.execute(ctx)

    result = ctx.metadata.get("execution_result")
    assert result is not None
    assert result.result == "test-sess_1"


# ---------------------------------------------------------------------------
# execute races execution against interrupt_event, raises ToolInterruptedError on interrupt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_races_against_interrupt_event(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute races execution against interrupt_event, raises ToolInterruptedError on interrupt."""
    # Register a slow tool
    tool_def = _tool_definition("slow_tool")
    tool_impl = ToolImplementation(
        factory=_slow_tool_factory,
        input_model=MockInputModel,
        output_model=MockOutputModel,
    )
    registry.register(tool_def, tool_impl)

    interrupt_event = asyncio.Event()
    ctx = _context(registry, workspace_manager, mock_backend, interrupt_event)
    ctx.validated_call = _validated_call("slow_tool")

    # Schedule interrupt after a short delay
    async def _trigger_interrupt():
        await asyncio.sleep(0.05)
        interrupt_event.set()

    asyncio.create_task(_trigger_interrupt())

    with pytest.raises(ToolInterruptedError) as exc_info:
        await execution_middleware.execute(ctx)

    assert "interrupted" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# execute on interrupt cancels execution task, waits for backend cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_on_interrupt_cancels_and_waits_cleanup(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute on interrupt cancels execution task, waits for backend cleanup."""
    # Track if the task was cancelled
    cleanup_called = False

    async def _cleanup_tool():
        nonlocal cleanup_called
        await asyncio.sleep(0.01)
        cleanup_called = True

    async def _slow_with_cleanup():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await _cleanup_tool()
            raise

    def _factory_with_cleanup(
        backend: Any,
        input_model: MockInputModel,
        session_id: str,
        turn_id: str | None = None,
    ):
        return _slow_with_cleanup()

    tool_def = _tool_definition("cleanup_tool")
    tool_impl = ToolImplementation(
        factory=_factory_with_cleanup,
        input_model=MockInputModel,
        output_model=MockOutputModel,
    )
    registry.register(tool_def, tool_impl)

    interrupt_event = asyncio.Event()
    ctx = _context(registry, workspace_manager, mock_backend, interrupt_event)
    ctx.validated_call = _validated_call("cleanup_tool")

    # Trigger interrupt
    async def _trigger_interrupt():
        await asyncio.sleep(0.05)
        interrupt_event.set()

    asyncio.create_task(_trigger_interrupt())

    with pytest.raises(ToolInterruptedError):
        await execution_middleware.execute(ctx)

    # Cleanup should have been called (the middleware waits for cancelled task)
    assert cleanup_called


@pytest.mark.asyncio
async def test_execute_handles_cleanup_timeout_gracefully(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute on interrupt waits up to 5 seconds for cleanup, then proceeds."""

    async def _slow_cleanup():
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            # Ignore cancellation during cleanup
            await asyncio.sleep(100)

    def _factory_slow_cleanup(
        backend: Any,
        input_model: MockInputModel,
        session_id: str,
        turn_id: str | None = None,
    ):
        return _slow_cleanup()

    tool_def = _tool_definition("slow_cleanup_tool")
    tool_impl = ToolImplementation(
        factory=_factory_slow_cleanup,
        input_model=MockInputModel,
        output_model=MockOutputModel,
    )
    registry.register(tool_def, tool_impl)

    interrupt_event = asyncio.Event()
    ctx = _context(registry, workspace_manager, mock_backend, interrupt_event)
    ctx.validated_call = _validated_call("slow_cleanup_tool")

    # Trigger interrupt immediately
    interrupt_event.set()

    with pytest.raises(ToolInterruptedError):
        await execution_middleware.execute(ctx)

    # Should complete within ~5 seconds (cleanup timeout)
    # If it didn't handle timeout, this would hang


# ---------------------------------------------------------------------------
# execute_stream yields execution_stream from context.metadata for streaming tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_stream_yields_chunks_for_streaming_tools(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute_stream yields ToolOutputChunk sequence for streaming tools."""
    # Register a tool
    tool_def = _tool_definition("stream_tool")
    tool_impl = ToolImplementation(
        factory=_async_tool_factory,
        input_model=MockInputModel,
        output_model=MockOutputModel,
    )
    registry.register(tool_def, tool_impl)

    ctx = _context(registry, workspace_manager, mock_backend)
    ctx.validated_call = _validated_call("stream_tool")
    ctx.metadata["streaming"] = True

    chunks = []
    async for chunk in execution_middleware.execute_stream(ctx):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert isinstance(chunks[0], ToolOutputChunk)
    assert chunks[0].channel == "status"
    assert chunks[0].payload["status"] == "completed"
    assert "output" in chunks[0].payload


@pytest.mark.asyncio
async def test_execute_stream_raises_without_validated_call(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute_stream raises ValueError when validated_call is not set."""
    ctx = _context(registry, workspace_manager, mock_backend)
    ctx.validated_call = None
    ctx.metadata["streaming"] = True

    with pytest.raises(ValueError, match="Validated call not set"):
        async for _ in execution_middleware.execute_stream(ctx):
            pass


# ---------------------------------------------------------------------------
# execute respects tool timeout_profile: soft_timeout, hard_timeout, idle_timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_respects_timeout_profile_fields(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute respects tool timeout_profile: soft_timeout, hard_timeout, idle_timeout."""
    # The middleware uses hard_timeout for asyncio.wait_for
    # soft_timeout and idle_timeout are used by the supervisor (not tested here)
    timeout_profile = TimeoutProfile(
        soft_timeout=timedelta(seconds=5),
        hard_timeout=timedelta(seconds=0.1),
        idle_timeout=timedelta(seconds=30),
    )
    tool_def = _tool_definition("timeout_tool", timeout_profile=timeout_profile)
    tool_impl = ToolImplementation(
        factory=_slow_tool_factory,
        input_model=MockInputModel,
        output_model=MockOutputModel,
    )
    registry.register(tool_def, tool_impl)

    ctx = _context(registry, workspace_manager, mock_backend)
    ctx.validated_call = _validated_call(
        "timeout_tool", timeout_profile=timeout_profile
    )

    # Should timeout based on hard_timeout (0.1s)
    with pytest.raises(asyncio.TimeoutError):
        await execution_middleware.execute(ctx)


@pytest.mark.asyncio
async def test_execute_with_sync_factory_returns_directly(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute with non-awaitable factory result returns directly without await."""
    tool_def = _tool_definition("sync_tool")
    tool_impl = ToolImplementation(
        factory=_mock_tool_factory,
        input_model=MockInputModel,
        output_model=MockOutputModel,
    )
    registry.register(tool_def, tool_impl)

    ctx = _context(registry, workspace_manager, mock_backend)
    ctx.validated_call = _validated_call("sync_tool")

    await execution_middleware.execute(ctx)

    result = ctx.metadata.get("execution_result")
    assert result is not None
    assert result.result == "test-sess_1"


@pytest.mark.asyncio
async def test_execute_raises_without_validated_call(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute raises ValueError when validated_call is not set."""
    ctx = _context(registry, workspace_manager, mock_backend)
    ctx.validated_call = None

    with pytest.raises(ValueError, match="Validated call not set"):
        await execution_middleware.execute(ctx)


@pytest.mark.asyncio
async def test_execute_short_circuits_on_non_allow_decision(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute returns early when decision is not ALLOW."""
    from local_coding_assistant.tools_platform.contracts import (
        ToolDecision,
        ToolDecisionOutcome,
    )

    ctx = _context(registry, workspace_manager, mock_backend)
    ctx.validated_call = _validated_call("test_tool")
    ctx.decision = ToolDecision(
        outcome=ToolDecisionOutcome.DENY,
        reason="Test deny",
    )

    await execution_middleware.execute(ctx)

    # Should not execute the tool
    assert "execution_result" not in ctx.metadata


@pytest.mark.asyncio
async def test_execute_raises_for_missing_tool_implementation(
    execution_middleware: ExecutionMiddleware,
    registry: ToolRegistry,
    workspace_manager: WorkspaceManager,
    mock_backend: MagicMock,
) -> None:
    """execute raises ToolValidationError when tool implementation not found."""
    ctx = _context(registry, workspace_manager, mock_backend)
    ctx.validated_call = _validated_call("nonexistent_tool")

    with pytest.raises(Exception):  # ToolValidationError
        await execution_middleware.execute(ctx)
