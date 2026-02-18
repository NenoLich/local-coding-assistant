import pytest

from unittest.mock import MagicMock, AsyncMock

from local_coding_assistant.runtime.execution_types import (
    ExecutionFrame,
    ActionKind,
    ExecutionStatus,
    LLMMetrics,
)
from local_coding_assistant.runtime.executor import RuntimeExecutor
from local_coding_assistant.agent.llm.models import LLMResult, LLMToolCall
from local_coding_assistant.tools.types import ToolExecutionResponse


@pytest.fixture
def mock_llm_service():
    manager = MagicMock()
    manager.generate = AsyncMock()
    return manager


@pytest.fixture
def mock_tool_manager():
    manager = MagicMock()
    manager.execute_async = AsyncMock()
    return manager


@pytest.mark.asyncio
async def test_executor_basic_flow(mock_llm_service, mock_tool_manager):
    """Test basic executor flow with new data structures."""
    # Setup
    executor = RuntimeExecutor(mock_llm_service, mock_tool_manager)

    # Create mock prompt context and rendered prompt
    from local_coding_assistant.runtime.runtime_types import (
        PromptContext,
        RenderedPrompt,
    )

    prompt_context = PromptContext(
        session_id="test-session",
        execution_mode="reasoning_only",
        tool_call_mode="classic",
        user_input="Hello",
        tools=[],
        history=[],
    )

    rendered_prompt = RenderedPrompt(
        system_messages=["System message"],
        user_messages=["User: Hello"],
        tool_schemas=[],
    )

    frame = ExecutionFrame(
        session_id="test-session",
        prompt_context=prompt_context,
        rendered_prompt=rendered_prompt,
    )

    # Mock LLM Response
    mock_llm_service.generate.return_value = LLMResult(
        content="Hello back!",
        model="test-model",
        provider="test-provider",
        total_tokens=10,
        prompt_tokens=5,
        completion_tokens=5,
        reasoning_tokens=0,
    )

    # Execute
    result_frame = await executor.execute(frame, use_streaming=False)

    # Verify
    assert result_frame.model_response_raw == "Hello back!"
    assert result_frame.result.status == ExecutionStatus.SUCCESS
    assert result_frame.result.final_answer == "Hello back!"
    assert len(result_frame.actions) == 1
    assert result_frame.actions[0].kind == ActionKind.LLM_MESSAGE
    assert result_frame.actions[0].name == "generate"

    # Test on-demand LLM metrics access
    llm_metrics = result_frame.get_llm_metrics()
    assert llm_metrics is not None
    assert llm_metrics.model == "test-model"
    assert llm_metrics.total_tokens == 10


@pytest.mark.asyncio
async def test_executor_with_tool_call(mock_llm_service, mock_tool_manager):
    """Test executor with tool call using new data structures."""
    # Setup
    executor = RuntimeExecutor(mock_llm_service, mock_tool_manager)

    # Create mock prompt context and rendered prompt
    from local_coding_assistant.runtime.runtime_types import (
        PromptContext,
        RenderedPrompt,
        ToolSpec,
    )

    tool_spec = ToolSpec(name="get_time", description="Get current time")

    prompt_context = PromptContext(
        session_id="test-session",
        execution_mode="classic_tools",
        tool_call_mode="classic",
        user_input="What time is it?",
        tools=[tool_spec],
        history=[],
    )

    rendered_prompt = RenderedPrompt(
        system_messages=["System message"],
        user_messages=["User: What time is it?"],
        tool_schemas=[tool_spec.to_openai_function()],
    )

    frame = ExecutionFrame(
        session_id="test-session",
        prompt_context=prompt_context,
        rendered_prompt=rendered_prompt,
    )

    # Mock LLM Response with Tool Call
    tool_call = LLMToolCall(name="get_time", arguments={})
    mock_llm_service.generate.return_value = LLMResult(
        content="Let me check.",
        model="test-model",
        provider="test-provider",
        total_tokens=20,
        prompt_tokens=10,
        completion_tokens=10,
        reasoning_tokens=0,
        tool_calls=[tool_call],
    )

    # Mock Tool Execution
    mock_tool_manager.execute_async.return_value = ToolExecutionResponse(
        success=True,
        tool_name="get_time",
        tool_args={},
        result="2026-01-20T12:00:00",
        execution_time_ms=5.0,
    )

    # Execute
    result_frame = await executor.execute(frame, use_streaming=False)

    # Verify
    assert len(result_frame.actions) == 2  # 1 LLM + 1 Tool
    assert result_frame.actions[1].kind == ActionKind.TOOL_CALL
    assert result_frame.actions[1].name == "get_time"

    # Test tool trace (new data structure)
    tool_action = result_frame.actions[1]
    assert tool_action.tool_trace is not None
    assert tool_action.tool_trace.tool_name == "get_time"
    assert tool_action.tool_trace.success is True

    # Test legacy output property
    assert tool_action.output == "2026-01-20T12:00:00"

    # Test on-demand tool results access
    tool_results = result_frame.get_tool_results()
    assert len(tool_results) == 1
    assert tool_results[0]["tool_name"] == "get_time"
    assert tool_results[0]["result"] == "2026-01-20T12:00:00"
    assert tool_results[0]["success"] is True

    assert result_frame.result.status == ExecutionStatus.SUCCESS
