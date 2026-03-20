import pytest

from unittest.mock import MagicMock, AsyncMock

from local_coding_assistant.runtime.execution_types import (
    ExecutionFrame,
    ActionKind,
    ExecutionStatus,
)
from local_coding_assistant.runtime.executor import RuntimeExecutor
from local_coding_assistant.agent.llm.models import LLMToolCall
from local_coding_assistant.tools.types import ToolExecutionResponse


@pytest.fixture
def mock_llm_service():
    manager = MagicMock()
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
        execution_mode=ExecutionMode.REASONING_ONLY,
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

    # Mock LLM stream
    async def mock_stream(task, options=None):
        chunk = MagicMock()
        chunk.content = "Hello back!"
        chunk.tool_calls = None
        chunk.reasoning = None
        chunk.usage = {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}
        chunk.metadata = {}
        chunk.model = "test-model"
        chunk.provider = "test-provider"
        chunk.finish_reason = "stop"
        yield chunk

    mock_llm_service.stream = mock_stream

    # Execute
    events = []
    async for event in executor.execute(frame):
        events.append(event)

    # Find the frame complete event
    frame_complete_events = [e for e in events if e.type == EventType.FRAME_COMPLETE]
    assert len(frame_complete_events) == 1
    result_frame = frame_complete_events[0].data["frame"]

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
        execution_mode=ExecutionMode.CLASSIC_TOOLS,
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

    # Mock LLM stream with tool call
    tool_call = LLMToolCall(name="get_time", arguments={}, id="call_1")

    async def mock_stream(task, options=None):
        chunk = MagicMock()
        chunk.content = "Let me check."
        chunk.tool_calls = [tool_call]
        chunk.reasoning = None
        chunk.usage = {"total_tokens": 20, "prompt_tokens": 10, "completion_tokens": 10}
        chunk.metadata = {}
        chunk.model = "test-model"
        chunk.provider = "test-provider"
        chunk.finish_reason = "tool_calls"
        yield chunk

    mock_llm_service.stream = mock_stream

    # Mock Tool Execution
    mock_tool_manager.execute_async.return_value = ToolExecutionResponse(
        success=True,
        tool_name="get_time",
        tool_args={},
        result="2026-01-20T12:00:00",
        execution_time_ms=5.0,
    )

    # Execute
    events = []
    async for event in executor.execute(frame):
        events.append(event)

    # Find the frame complete event
    frame_complete_events = [e for e in events if e.type == EventType.FRAME_COMPLETE]
    assert len(frame_complete_events) == 1
    result_frame = frame_complete_events[0].data["frame"]

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


# Streaming tests
from local_coding_assistant.runtime.events import EventType
from local_coding_assistant.agent.llm import LLMOptions, LLMTask


@pytest.fixture
def mock_llm_service_streaming():
    """Mock LLM service for streaming testing."""
    service = MagicMock()
    return service


@pytest.fixture
def mock_config_manager_streaming():
    """Mock config manager for streaming testing."""
    return MagicMock()


@pytest.fixture
def mock_context_manager_streaming():
    """Mock context manager for streaming testing."""
    return MagicMock()


@pytest.fixture
def executor_streaming(
    mock_llm_service_streaming,
    mock_tool_manager,
    mock_config_manager_streaming,
    mock_context_manager_streaming,
):
    """Create RuntimeExecutor instance with mocked dependencies for streaming."""
    executor = RuntimeExecutor(
        llm_service=mock_llm_service_streaming,
        tool_manager=mock_tool_manager,
        context_manager=mock_context_manager_streaming,
        config_manager=None,  # Set to None to avoid mocking issues
    )
    executor._max_tokens = 10000
    return executor


from local_coding_assistant.runtime.runtime_types import (
    PromptContext,
    RenderedPrompt,
    ExecutionMode,
)


@pytest.fixture
def sample_frame_streaming():
    """Create a sample execution frame for streaming testing."""
    prompt_context = PromptContext(
        session_id="test_session",
        agent_profile=None,
        execution_mode=ExecutionMode.CLASSIC_TOOLS,
        tool_call_mode="classic",
        user_input="Test message",
        tools=[],
        history=[],
    )
    rendered_prompt = RenderedPrompt(
        user_messages=["Test message"],
        system_messages=[],
        tool_schemas=[],
    )

    frame = ExecutionFrame(
        session_id="test_session",
        prompt_context=prompt_context,
        rendered_prompt=rendered_prompt,
    )
    return frame


class TestRuntimeExecutorStreaming:
    """Test RuntimeExecutor event emission in streaming mode."""

    @pytest.mark.asyncio
    async def test_execute_emits_frame_start_and_complete(
        self, executor_streaming, sample_frame_streaming, mock_llm_service_streaming
    ):
        """Test that execute emits FRAME_START and FRAME_COMPLETE events."""

        # Mock LLM stream to return a simple response
        async def mock_stream(task, options=None):
            chunk = MagicMock()
            chunk.content = "Test response"
            chunk.tool_calls = None
            chunk.reasoning = None
            chunk.usage = {
                "total_tokens": 10,
                "prompt_tokens": 5,
                "completion_tokens": 5,
            }
            chunk.metadata = {}
            chunk.model = "test-model"
            chunk.provider = "test-provider"
            chunk.finish_reason = "stop"
            yield chunk

        mock_llm_service_streaming.stream = mock_stream

        events = []
        async for event in executor_streaming.execute(sample_frame_streaming):
            events.append(event)

        assert len(events) >= 2  # At least FRAME_START and FRAME_COMPLETE
        assert events[0].type == EventType.FRAME_START
        assert events[0].session_id == "test_session"
        assert events[0].frame_id == sample_frame_streaming.id

        # Find FRAME_COMPLETE event
        frame_complete_events = [
            e for e in events if e.type == EventType.FRAME_COMPLETE
        ]
        assert len(frame_complete_events) == 1
        assert frame_complete_events[0].frame_id == sample_frame_streaming.id

    @pytest.mark.asyncio
    async def test_execute_emits_llm_events(
        self, executor_streaming, sample_frame_streaming, mock_llm_service_streaming
    ):
        """Test that execute emits LLM_START, LLM_CHUNK, and LLM_COMPLETE events."""

        async def mock_stream(task, options=None):
            # First chunk
            chunk1 = MagicMock()
            chunk1.content = "Hello"
            chunk1.tool_calls = None
            chunk1.reasoning = None
            chunk1.usage = None
            chunk1.metadata = {}
            chunk1.model = "test-model"
            chunk1.provider = "test-provider"
            chunk1.finish_reason = None
            yield chunk1

            # Second chunk
            chunk2 = MagicMock()
            chunk2.content = " world"
            chunk2.tool_calls = None
            chunk2.reasoning = None
            chunk2.usage = None
            chunk2.metadata = {}
            chunk2.model = "test-model"
            chunk2.provider = "test-provider"
            chunk2.finish_reason = None
            yield chunk2

            # Final chunk
            chunk3 = MagicMock()
            chunk3.content = ""
            chunk3.tool_calls = None
            chunk3.reasoning = None
            chunk3.usage = {
                "total_tokens": 15,
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }
            chunk3.metadata = {}
            chunk3.model = "test-model"
            chunk3.provider = "test-provider"
            chunk3.finish_reason = "stop"
            yield chunk3

        mock_llm_service_streaming.stream = mock_stream

        events = []
        async for event in executor_streaming.execute(sample_frame_streaming):
            events.append(event)

        llm_events = [e for e in events if e.type.value.startswith("llm_")]
        assert len(llm_events) >= 4  # START, at least 2 CHUNK, COMPLETE

        # Check sequence
        event_types = [e.type for e in llm_events]
        assert EventType.LLM_START in event_types
        assert EventType.LLM_CHUNK in event_types
        assert EventType.LLM_COMPLETE in event_types

        # Check LLM_COMPLETE has result
        complete_events = [e for e in llm_events if e.type == EventType.LLM_COMPLETE]
        assert len(complete_events) == 1
        assert "result" in complete_events[0].data
        result = complete_events[0].data["result"]
        assert result.content == "Hello world"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_execute_with_tool_calls_emits_tool_events(
        self,
        executor_streaming,
        sample_frame_streaming,
        mock_llm_service_streaming,
        mock_tool_manager,
    ):
        """Test that execute emits TOOL_START and TOOL_RESULT events for tool calls."""
        from local_coding_assistant.agent.llm import LLMToolCall
        from local_coding_assistant.tools.types import ToolExecutionResponse
        from local_coding_assistant.runtime.runtime_types import ToolSpec

        # Add tool to frame
        tool_spec = ToolSpec(name="test_tool", description="Test tool")
        sample_frame_streaming.prompt_context.tools = [tool_spec]
        sample_frame_streaming.rendered_prompt.tool_schemas = [
            tool_spec.to_openai_function()
        ]

        # Mock tool call
        tool_call = LLMToolCall(
            id="call_1", type="function", name="test_tool", arguments={"arg": "value"}
        )

        async def mock_stream(task, options=None):
            chunk = MagicMock()
            chunk.content = ""
            chunk.tool_calls = [tool_call]
            chunk.reasoning = None
            chunk.usage = {"total_tokens": 20}
            chunk.metadata = {}
            chunk.model = "test-model"
            chunk.provider = "test-provider"
            chunk.finish_reason = "tool_calls"
            yield chunk

        mock_llm_service_streaming.stream = mock_stream

        # Mock tool execution
        mock_tool_manager.execute_async.return_value = ToolExecutionResponse(
            success=True,
            tool_name="test_tool",
            result="Tool result",
            execution_time_ms=100.0,
        )

        events = []
        async for event in executor_streaming.execute(sample_frame_streaming):
            events.append(event)

        tool_events = [e for e in events if e.type.value.startswith("tool_")]
        assert len(tool_events) == 2  # START and RESULT

        start_events = [e for e in tool_events if e.type == EventType.TOOL_START]
        result_events = [e for e in tool_events if e.type == EventType.TOOL_RESULT]

        assert len(start_events) == 1
        assert len(result_events) == 1

        assert start_events[0].data["tool_call"] == tool_call
        assert result_events[0].data["tool_call"] == tool_call
        assert result_events[0].data["response"].success is True

    @pytest.mark.asyncio
    async def test_execute_handles_llm_error(
        self, executor_streaming, sample_frame_streaming, mock_llm_service_streaming
    ):
        """Test that execute handles LLM errors and emits ERROR event."""
        mock_llm_service_streaming.stream.side_effect = Exception("LLM service error")

        events = []
        async for event in executor_streaming.execute(sample_frame_streaming):
            events.append(event)

        error_events = [e for e in events if e.type == EventType.ERROR]
        assert len(error_events) == 1
        assert "LLM service error" in error_events[0].data["error"]

        # Should still emit FRAME_COMPLETE
        frame_complete_events = [
            e for e in events if e.type == EventType.FRAME_COMPLETE
        ]
        assert len(frame_complete_events) == 1

    @pytest.mark.asyncio
    async def test_prepare_llm_request(
        self, executor_streaming, sample_frame_streaming
    ):
        """Test _prepare_llm_request method."""
        task, options = executor_streaming._prepare_llm_request(sample_frame_streaming)

        assert isinstance(task, LLMTask)
        assert task.prompt == "Test message"
        assert task.system_prompt is None
        assert task.tools == []
        assert task.context == []

        assert isinstance(options, LLMOptions)

    @pytest.mark.asyncio
    async def test_generate_llm_events_yields_chunks(
        self, executor_streaming, mock_llm_service_streaming
    ):
        """Test _generate_llm_events yields correct events."""
        # Mock config manager to enable reasoning capture
        mock_config = MagicMock()
        mock_llm_config = MagicMock()
        mock_llm_config.capture_reasoning = True
        mock_llm_config.reasoning_max_chars = 0
        mock_config.global_config.llm = mock_llm_config
        executor_streaming._config_manager = mock_config

        task = LLMTask(prompt="Test")
        options = LLMOptions()

        async def mock_stream(task, options=None):
            # Content chunk
            chunk1 = MagicMock()
            chunk1.configure_mock(
                content="Chunk 1",
                tool_calls=None,
                reasoning=None,
                usage=None,
                metadata={},
                model="test-model",
                provider="test-provider",
                finish_reason=None,
            )
            yield chunk1

            # Reasoning chunk
            chunk2 = MagicMock()
            chunk2.configure_mock(
                content=None,
                tool_calls=None,
                reasoning="Thinking...",
                usage=None,
                metadata={},
                model="test-model",
                provider="test-provider",
                finish_reason=None,
            )
            yield chunk2

            # Final chunk
            chunk3 = MagicMock()
            chunk3.configure_mock(
                content="",
                tool_calls=None,
                reasoning=None,
                usage={"total_tokens": 10},
                metadata={},
                model="test-model",
                provider="test-provider",
                finish_reason="stop",
            )
            yield chunk3

        mock_llm_service_streaming.stream = mock_stream

        events = []
        async for event in executor_streaming._generate_llm_events(
            task, options=options, session_id="test", frame_id="frame_1"
        ):
            events.append(event)

        assert len(events) >= 3  # content chunk + reasoning chunk + complete

        chunk_events = [e for e in events if e.type == EventType.LLM_CHUNK]
        assert len(chunk_events) >= 2  # content chunk + reasoning chunk

        complete_events = [e for e in events if e.type == EventType.LLM_COMPLETE]
        assert len(complete_events) == 1
