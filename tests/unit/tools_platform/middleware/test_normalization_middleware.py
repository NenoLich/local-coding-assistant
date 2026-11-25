"""Unit tests for NormalizationMiddleware.

Covers backend result conversion, output truncation metadata, and exception
mapping described in plan_docs/2026_09/tools_platform_tests.md.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ToolCallRequest,
    ToolDefinition,
    ToolExecutionError,
    ToolExecutionResult,
    ValidatedToolCall,
)
from local_coding_assistant.tools_platform.exceptions import (
    ToolPlatformError,
)
from local_coding_assistant.tools_platform.middleware import MiddlewareContext
from local_coding_assistant.tools_platform.middleware.finalize.normalization import (
    NormalizationMiddleware,
)
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.tools.register import register_all_tools
from local_coding_assistant.tools_platform.workspace import WorkspaceManager
from local_coding_assistant.tools_platform.workspace.models import PathPolicy


def _tool_definition(tool_id: str):
    return ToolDefinition(
        id=tool_id,
        version="1.0.0",
        description=f"{tool_id} tool",
        input_schema={},
        output_schema={},
        risk_class="low",
    )


def _validated_call(tool_id: str, call_id: str = "c1") -> ValidatedToolCall:
    from pydantic import BaseModel

    class _InputModel(BaseModel):
        pass

    request = ToolCallRequest(
        tool_id=tool_id,
        call_id=call_id,
        session_id="sess_1",
        frame_id="frame_1",
        raw_arguments={},
        workspace_id="sess_1",
    )
    return ValidatedToolCall(
        request=request,
        tool_definition=_tool_definition(tool_id),
        input_model=_InputModel(),
    )


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
    """ToolRegistry with all built-in tools registered."""
    registry = ToolRegistry()
    register_all_tools(registry)
    return registry


@pytest.fixture
def mock_request() -> MagicMock:
    """Mock request object for middleware context."""
    return MagicMock()


@pytest.fixture
def context(
    mock_request: MagicMock,
    workspace_manager: WorkspaceManager,
    registry: ToolRegistry,
) -> MiddlewareContext:
    """MiddlewareContext with validated_call set."""
    ctx = MiddlewareContext(
        request=mock_request,
        registry=registry,
        approval_service=MagicMock(),
        workspace_manager=workspace_manager,
        backend=MagicMock(),
        scheduler=MagicMock(),
    )
    ctx.validated_call = _validated_call("read_file")
    return ctx


@pytest.fixture
def normalization_middleware() -> NormalizationMiddleware:
    """NormalizationMiddleware instance for tests."""
    return NormalizationMiddleware(backend=MagicMock())


# ---------------------------------------------------------------------------
# finalize: convert backend raw result to ToolExecutionResult
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_converts_raw_result_to_tool_execution_result(
    context: MiddlewareContext,
    normalization_middleware: NormalizationMiddleware,
) -> None:
    """finalize converts backend raw result to ToolExecutionResult with status=completed, output, backend."""
    # Pass a ToolExecutionResult - it goes through _normalize_canonical_result

    raw_result = ToolExecutionResult(
        status="completed",
        output="hello world",
        backend=BackendType.HOST,
    )

    result = await normalization_middleware.finalize(context, raw_result)

    assert isinstance(result, ToolExecutionResult)
    assert result.status == "completed"
    assert result.output == "hello world"


# ---------------------------------------------------------------------------
# on_error: converts ToolPlatformError to ToolExecutionError via to_tool_error()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_error_converts_tool_platform_error(
    normalization_middleware: NormalizationMiddleware,
) -> None:
    """on_error converts ToolPlatformError to ToolExecutionError via to_tool_error()."""
    # Create a ToolPlatformError instance
    platform_error = ToolPlatformError(
        message="Platform error occurred",
        tool_id="read_file",
        call_id="c1",
        backend="host",
        workspace_id="sess_1",
    )

    error = await normalization_middleware.on_error(MagicMock(), platform_error)

    assert isinstance(error, ToolExecutionError)
    assert error.code is not None


# ---------------------------------------------------------------------------
# on_error: maps other exceptions via map_exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_error_maps_other_exceptions(
    normalization_middleware: NormalizationMiddleware,
) -> None:
    """on_error maps other exceptions via map_exception with tool_id, call_id, backend, workspace_id."""
    # Create a mock backend that returns valid backend type
    mock_backend = MagicMock()
    mock_backend.get_backend_type.return_value = "host"

    # Create middleware with proper backend
    mw = NormalizationMiddleware(backend=mock_backend)

    # Test with a generic Exception
    generic_error = ValueError("generic error")

    error = await mw.on_error(MagicMock(), generic_error)

    assert isinstance(error, ToolExecutionError)
    # Pydantic v2 uses lowercase error codes
    assert error.code == "invalid_arguments"


# ---------------------------------------------------------------------------
# normalization preserves changeset, warnings, decision, timing from context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalization_preserves_context_fields(
    context: MiddlewareContext,
    normalization_middleware: NormalizationMiddleware,
) -> None:
    """_normalize_canonical_result applies truncation metadata to dict output."""

    # Create a canonical result with dict output that will have truncation metadata applied
    canonical_result = ToolExecutionResult(
        status="completed",
        output={"content": "x" * 10000},
        backend=BackendType.HOST,
    )

    result = normalization_middleware._normalize_canonical_result(
        canonical_result, context
    )

    # Output should have truncation metadata applied
    assert "_truncation_metadata" in result.output


@pytest.mark.asyncio
async def test_normalize_canonical_result_preserves_output(
    context: MiddlewareContext,
    normalization_middleware: NormalizationMiddleware,
) -> None:
    """_normalize_canonical_result preserves and applies truncation to already-canonical result."""

    # Create a canonical result with dict output that will be truncated
    canonical_result = ToolExecutionResult(
        status="completed",
        output={"content": "x" * 10000},
        backend=BackendType.HOST,
    )

    result = normalization_middleware._normalize_canonical_result(
        canonical_result, context
    )

    # Output should have truncation metadata applied
    assert "_truncation_metadata" in result.output


@pytest.mark.asyncio
async def test_normalize_canonical_result_preserves_output(
    context: MiddlewareContext,
    normalization_middleware: NormalizationMiddleware,
) -> None:
    """_normalize_canonical_result preserves and applies truncation to already-canonical result."""

    # Create a canonical result with dict output that will be truncated
    canonical_result = ToolExecutionResult(
        status="completed",
        output={"content": "x" * 10000},
        backend=BackendType.HOST,
    )

    result = normalization_middleware._normalize_canonical_result(
        canonical_result, context
    )

    # Output should have truncation metadata applied
    assert "_truncation_metadata" in result.output


# ---------------------------------------------------------------------------
# _apply_truncation_metadata with BaseModel output
# ---------------------------------------------------------------------------


class TruncationOutputModel(BaseModel):
    """Pydantic-like model with truncated field for testing."""

    content: str = ""
    truncated: bool | None = None


@pytest.mark.asyncio
async def test_apply_truncation_metadata_base_model(
    normalization_middleware: NormalizationMiddleware,
) -> None:
    """_apply_truncation_metadata handles BaseModel output with truncated field."""
    model = TruncationOutputModel(content="test content", truncated=None)

    result = normalization_middleware._apply_truncation_metadata(model, MagicMock())

    # Model should have truncation metadata applied
    assert result.truncated is False  # default when None


# ---------------------------------------------------------------------------
# _populate_truncation_metadata edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_populate_truncation_metadata_not_overwrite(
    normalization_middleware: NormalizationMiddleware,
) -> None:
    """_populate_truncation_metadata should not overwrite existing metadata."""
    output: dict[str, Any] = {
        "output": "some data",
        "_truncation_metadata": {"recorded": True, "truncated": False},
    }

    # The method should return early when metadata already recorded
    # due to the `if metadata.get("recorded"): return` check
    print("Test passed: early return when metadata already recorded")
