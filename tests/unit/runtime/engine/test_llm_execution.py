"""Unit tests for RuntimeEngine LLM phase execution.

Covers the plan's section 6.2 LLM Phase scenarios:

- Streaming content + reasoning accumulation
- Tool calls extraction from chunks
- Finish reason: stop -> COMPLETED (no tools) or RUNNING_TOOLS (with tools)
- Finish reason: length -> PARTIAL with truncation error
- Finish reason: content_filter -> BLOCKED
- Transient error (503, 429): frame stays RUNNING_LLM
- Permanent error (401): frame -> FAILED
- Token budget exceeded during streaming -> FAILED
- Token budget exceeded after LLM complete -> FAILED
- Metadata capture (model, provider, tokens, latency)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.agent.llm import LLMToolCall
from local_coding_assistant.config.schemas import TokenBudgetConfig
from local_coding_assistant.providers.exceptions import (
    ProviderAuthError,
    ProviderRateLimitError,
)
from local_coding_assistant.runtime.context.profile import (
    ContextSnapshot,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.engine.engine import RuntimeEngine
from local_coding_assistant.runtime.engine.frame import (
    ActionKind,
    ExecutionStatus,
    RuntimeFrame,
)
from local_coding_assistant.runtime.engine.limits import TokenBudget
from local_coding_assistant.runtime.session.turn import RuntimeTurn
from local_coding_assistant.runtime.types import RenderedPrompt
from local_coding_assistant.tools_platform.contracts import (
    ToolExecutionResult,
    ToolExecutionStatus,
)


@pytest.fixture
def mock_llm_service():
    """Mock LLM service whose stream() returns an async generator directly."""
    service = MagicMock()
    service.stream = MagicMock()
    return service


@pytest.fixture
def mock_tool_gateway():
    """Mock tool gateway."""
    gateway = MagicMock()
    gateway.execute_many = AsyncMock()
    return gateway


@pytest.fixture
def mock_event_pipeline():
    """Mock event pipeline."""
    pipeline = MagicMock()
    pipeline.emit = AsyncMock()
    return pipeline


@pytest.fixture
def sample_rendered():
    """Sample rendered prompt."""
    return RenderedPrompt(
        messages=[{"role": "user", "content": "test input"}],
        system_messages=["You are a helpful assistant"],
        user_messages=["test input"],
        history=[],
        tool_schemas=[],
    )


@pytest.fixture
def sample_resolved():
    """Sample resolved context."""
    return SimpleNamespace(
        metadata={"model": "test_model", "temperature": 0.7, "streaming": True},
        streaming=True,
    )


@pytest.fixture
def sample_runtime_turn():
    """Sample runtime turn."""
    turn = RuntimeTurn(
        turn_id="turn_123",
        session_id="session_123",
        trace_id="trace_123",
        turn_index=1,
        user_input="test input",
        context_snapshot=ContextSnapshot(
            id="ctx_123",
            turn_id="turn_123",
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
            repo_snapshot_ref=None,
            memory_snapshot_ref=None,
            message_boundary=None,
            content_hash="hash123",
            created_at_ms=1000,
        ),
        context_snapshot_id="ctx_123",
        context_config=RuntimeContextConfig(),
        limits_config=RuntimeLimitsConfig(),
    )
    return turn


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str = "",
    reasoning: str = "",
    tool_calls: list[LLMToolCall] | None = None,
    model: str = "mock-model",
    provider: str = "mock-provider",
    finish_reason: str = "stop",
    total_tokens: int | None = 100,
    prompt_tokens: int | None = 50,
    completion_tokens: int | None = 50,
    reasoning_tokens: int | None = 0,
    latency_ms: float | None = 10.0,
) -> SimpleNamespace:
    """Build a mock LLM stream chunk readable by RuntimeEngine._update_chunk_metadata."""
    return SimpleNamespace(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls or [],
        model=model,
        provider=provider,
        finish_reason=finish_reason,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        latency_ms=latency_ms,
        usage=None,
        metadata={},
    )


async def _stream(chunks: list[SimpleNamespace]):
    """Async generator yielding mock chunks."""
    for chunk in chunks:
        yield chunk


def _make_engine(
    llm_service: MagicMock | None = None,
    tool_gateway: MagicMock | None = None,
    event_pipeline: MagicMock | None = None,
) -> RuntimeEngine:
    return RuntimeEngine(
        llm_service=llm_service or MagicMock(),
        tool_gateway=tool_gateway or MagicMock(),
        event_pipeline=event_pipeline or MagicMock(),
    )


def _make_turn(
    turn_id: str = "turn_123",
    session_id: str = "session_123",
    trace_id: str = "trace_123",
    limits_config: RuntimeLimitsConfig | None = None,
    interrupted: bool = False,
) -> RuntimeTurn:
    turn = RuntimeTurn(
        turn_id=turn_id,
        session_id=session_id,
        trace_id=trace_id,
        turn_index=1,
        user_input="test input",
        context_snapshot=ContextSnapshot(
            id="ctx_123",
            turn_id=turn_id,
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
            repo_snapshot_ref=None,
            memory_snapshot_ref=None,
            message_boundary=None,
            content_hash="hash123",
            created_at_ms=1000,
        ),
        context_snapshot_id="ctx_123",
        context_config=RuntimeContextConfig(),
        limits_config=limits_config or RuntimeLimitsConfig(),
    )
    if interrupted:
        turn.interrupt_event.set()
    return turn


class TestLLMExecution:
    """LLM phase: stream processing, token budget, finish_reason handling,
    reasoning accumulation, tool_calls extraction, metadata capture,
    and LLM failure classification."""

    # -- finish_reason transitions -------------------------------------------

    @pytest.mark.asyncio
    async def test_llm_stop_without_tools_transitions_to_completed(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [_make_chunk(content="Hello, world!", finish_reason="stop")]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert result.result is not None
        assert result.result.final_answer == "Hello, world!"
        assert result.model_response_raw == "Hello, world!"
        assert mock_event_pipeline.emit.await_count > 0

    @pytest.mark.asyncio
    async def test_llm_stop_with_tool_calls_transitions_tools_then_completed(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        tool_calls = [
            LLMToolCall(
                id="call_1", name="read_file", arguments={"path": "/fake/a.txt"}
            )
        ]
        mock_llm_service.stream.return_value = _stream(
            [
                _make_chunk(
                    content="reading",
                    tool_calls=tool_calls,
                    finish_reason="stop",
                )
            ]
        )
        mock_tool_gateway.execute_many.return_value = [
            ToolExecutionResult(
                status=ToolExecutionStatus.COMPLETED.value,
                output="ok",
                call_id="call_1",
                tool_id="read_file",
            ),
        ]
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        # LLM phase transitions to RUNNING_TOOLS, then tools phase completes
        # within the same execute_frame call.
        assert result.status == ExecutionStatus.COMPLETED
        assert result.model_response_raw == "reading"
        mock_tool_gateway.execute_many.assert_awaited_once()
        llm_action = next(a for a in result.actions if a.kind == ActionKind.LLM_MESSAGE)
        assert len(llm_action.tool_calls) == 1
        assert llm_action.tool_calls[0].id == "call_1"
        assert llm_action.tool_calls[0].name == "read_file"

    @pytest.mark.asyncio
    async def test_llm_length_finish_reason_transitions_to_partial(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [_make_chunk(content="partial", finish_reason="length")]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.PARTIAL
        assert result.result is not None
        assert result.result.final_answer == "partial"
        assert "truncated" in (result.result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_llm_content_filter_transitions_to_blocked(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [_make_chunk(content="", finish_reason="content_filter")]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.BLOCKED
        assert result.result is not None
        assert "blocked" in (result.result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_llm_safety_finish_reason_transitions_to_blocked(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [_make_chunk(content="", finish_reason="safety")]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.BLOCKED

    # -- streaming accumulation ---------------------------------------------

    @pytest.mark.asyncio
    async def test_streaming_content_accumulates_across_chunks(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [
                _make_chunk(content="Hello", finish_reason=None),
                _make_chunk(content=", ", finish_reason=None),
                _make_chunk(content="world!", finish_reason="stop"),
            ]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert result.model_response_raw == "Hello, world!"

    @pytest.mark.asyncio
    async def test_streaming_reasoning_accumulates_across_chunks(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [
                _make_chunk(reasoning="step one ", finish_reason=None),
                _make_chunk(reasoning="step two", finish_reason="stop"),
            ]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.COMPLETED
        llm_action = next(a for a in result.actions if a.kind == ActionKind.LLM_MESSAGE)
        assert llm_action.metadata["reasoning"] == "step one step two"

    # -- tool call extraction ------------------------------------------------

    @pytest.mark.asyncio
    async def test_tool_calls_extracted_from_chunks(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        tool_calls = [
            LLMToolCall(id="call_1", name="read_file", arguments={"path": "/a.txt"}),
            LLMToolCall(id="call_2", name="write_file", arguments={"path": "/b.txt"}),
        ]
        mock_llm_service.stream.return_value = _stream(
            [
                _make_chunk(
                    content="calling tools", tool_calls=tool_calls, finish_reason="stop"
                )
            ]
        )
        mock_tool_gateway.execute_many.return_value = [
            ToolExecutionResult(
                status=ToolExecutionStatus.COMPLETED.value,
                output="ok",
                call_id="call_1",
                tool_id="read_file",
            ),
            ToolExecutionResult(
                status=ToolExecutionStatus.COMPLETED.value,
                output="ok",
                call_id="call_2",
                tool_id="write_file",
            ),
        ]
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.COMPLETED
        mock_tool_gateway.execute_many.assert_awaited_once()
        llm_action = next(a for a in result.actions if a.kind == ActionKind.LLM_MESSAGE)
        assert len(llm_action.tool_calls) == 2
        assert {tc.id for tc in llm_action.tool_calls} == {"call_1", "call_2"}
        assert {tc.name for tc in llm_action.tool_calls} == {"read_file", "write_file"}

    # -- token budget --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_token_budget_exceeded_during_streaming_transitions_to_failed(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        turn.limits.token_budget = TokenBudget(
            config=TokenBudgetConfig(output_max_tokens=10)
        )
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [_make_chunk(content="too long", completion_tokens=50, total_tokens=100)]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.FAILED
        assert result.result is not None
        assert "budget" in (result.result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_token_budget_exceeded_after_llm_complete_transitions_to_failed(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        turn.limits.token_budget = TokenBudget(
            config=TokenBudgetConfig(output_max_tokens=5)
        )
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [_make_chunk(content="ok", completion_tokens=50, total_tokens=100)]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.FAILED
        assert result.result is not None
        assert "budget" in (result.result.error_message or "").lower()

    # -- error classification ------------------------------------------------

    @pytest.mark.asyncio
    async def test_llm_transient_error_keeps_frame_running_llm(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.side_effect = ProviderRateLimitError("rate limited")
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.RUNNING_LLM
        assert result.result is not None
        assert "rate limited" in (result.result.error_message or "")
        assert result.result.metadata.get("transient_error") is True

    @pytest.mark.asyncio
    async def test_llm_permanent_error_transitions_to_failed(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.side_effect = ProviderAuthError("bad key")
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.FAILED
        assert result.result is not None
        assert "bad key" in (result.result.error_message or "")

    # -- metadata capture ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_llm_metadata_captured_in_action(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )
        mock_llm_service.stream.return_value = _stream(
            [
                _make_chunk(
                    content="Hi",
                    finish_reason="stop",
                    total_tokens=100,
                    prompt_tokens=50,
                    completion_tokens=50,
                    reasoning_tokens=0,
                    latency_ms=42.5,
                )
            ]
        )
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.COMPLETED
        llm_action = next(a for a in result.actions if a.kind == ActionKind.LLM_MESSAGE)
        metrics = llm_action.llm_metrics
        assert metrics is not None
        assert metrics.model == "mock-model"
        assert metrics.prompt_tokens == 50
        assert metrics.completion_tokens == 50
        assert metrics.latency_ms == 42.5

    # -- interrupt -----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_interrupt_during_llm_preserves_partial_content(
        self,
        mock_llm_service: MagicMock,
        mock_tool_gateway: MagicMock,
        mock_event_pipeline: MagicMock,
        sample_rendered: RenderedPrompt,
        sample_resolved: Any,
    ) -> None:
        engine = _make_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline)
        turn = _make_turn()
        frame = RuntimeFrame(
            id="frame_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            status=ExecutionStatus.RUNNING_LLM,
        )

        async def _interrupting_stream():
            yield _make_chunk(content="partial", finish_reason="stop")
            turn.interrupt_event.set()
            yield _make_chunk(content=" more", finish_reason="stop")

        mock_llm_service.stream.return_value = _interrupting_stream()
        result = await engine.execute_frame(
            frame, turn, sample_resolved, sample_rendered
        )

        assert result.status == ExecutionStatus.RUNNING_LLM
        assert result.model_response_raw == "partial more"
