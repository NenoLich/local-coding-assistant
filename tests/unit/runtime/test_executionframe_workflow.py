"""Unit tests for the new ExecutionFrame workflow implementation."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock


from local_coding_assistant.runtime.execution_types import (
    ActionKind,
    ExecutionFrame,
    ExecutionStatus,
)
from local_coding_assistant.runtime.executor import RuntimeExecutor
from local_coding_assistant.runtime.runtime_types import (
    PromptContext,
    RenderedPrompt,
    AgentProfile,
    ExecutionMode,
    ToolSpec,
)
from local_coding_assistant.agent.llm import LLMResult, LLMToolCall
from local_coding_assistant.tools.types import ToolExecutionResponse

from local_coding_assistant.core.telemetry_types import (
    ToolCallTrace,
    ResourceMetric,
    ResourceType,
    ExecutionEnvelope,
)


@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    service = AsyncMock()
    return service


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager."""
    manager = AsyncMock()
    manager.execute_async = AsyncMock()
    return manager


@pytest.fixture
def mock_context_manager():
    """Mock context manager."""
    manager = AsyncMock()
    return manager


@pytest.fixture
def runtime_executor(mock_llm_service, mock_tool_manager, mock_context_manager):
    """Runtime executor fixture."""
    return RuntimeExecutor(
        llm_service=mock_llm_service,
        tool_manager=mock_tool_manager,
        context_manager=mock_context_manager,
    )


@pytest.fixture
def sample_prompt_context():
    """Sample prompt context for testing."""
    return PromptContext(
        session_id="test_session",
        execution_mode=ExecutionMode.CLASSIC_TOOLS,
        tool_call_mode="classic",
        user_input="Test input",
        agent_profile=AgentProfile.default(),
        tools=[ToolSpec(name="test_tool", description="Test tool")],
        history=[],
    )


@pytest.fixture
def sample_rendered_prompt():
    """Sample rendered prompt for testing."""
    return RenderedPrompt(
        system_messages=["System message"],
        user_messages=["User message"],
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "Test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool1",
                    "description": "Tool 1",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool2",
                    "description": "Tool 2",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "failing_tool",
                    "description": "Failing tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "stats_tool",
                    "description": "Stats tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        history=[],
    )


@pytest.fixture
def sample_execution_frame(sample_prompt_context, sample_rendered_prompt):
    """Sample execution frame for testing."""
    return ExecutionFrame(
        session_id="test_session",
        iteration=1,
        prompt_context=sample_prompt_context,
        rendered_prompt=sample_rendered_prompt,
    )


class TestRuntimeExecutor:
    """Test cases for RuntimeExecutor."""

    @pytest.mark.asyncio
    async def test_execute_llm_only_success(
        self, runtime_executor, sample_execution_frame, mock_llm_service
    ):
        """Test successful execution with LLM only (no tool calls)."""
        # Setup mock LLM response
        mock_response = LLMResult(
            content="Test response",
            model="test-model",
            provider="test-provider",
            total_tokens=100,
            tool_calls=None,
            metadata={
                "provider_metadata": {
                    "latency_ms": 1.7
                }
            }
        )
        mock_llm_service.generate.return_value = mock_response

        # Setup mock stream
        del mock_llm_service.stream
        async def mock_stream(task, options=None):
            from unittest.mock import MagicMock
            chunk = MagicMock()
            chunk.content = mock_response.content
            chunk.tool_calls = mock_response.tool_calls
            chunk.reasoning = mock_response.reasoning
            chunk.usage = {"total_tokens": mock_response.total_tokens}
            chunk.metadata = mock_response.metadata.get("provider_metadata", {})
            chunk.model = mock_response.model
            chunk.provider = mock_response.provider
            chunk.finish_reason = mock_response.finish_reason
            yield chunk
        mock_llm_service.stream = mock_stream

        # Execute frame
        events = []
        async for event in runtime_executor.execute(sample_execution_frame):
            events.append(event)
        result_frame = events[-1].data['frame']

        # Verify frame structure
        assert result_frame.id == sample_execution_frame.id
        assert result_frame.session_id == "test_session"
        assert result_frame.iteration == 1
        assert result_frame.model_response_raw == "Test response"
        assert result_frame.result is not None
        assert result_frame.result.status == ExecutionStatus.SUCCESS
        assert result_frame.result.final_answer == "Test response"
        assert len(result_frame.actions) == 1
        assert result_frame.actions[0].kind == ActionKind.LLM_MESSAGE

        # Verify metrics
        metrics = result_frame.get_llm_metrics()
        tool_calls = result_frame.get_tool_results()
        assert metrics is not None
        assert metrics.total_tokens == 100
        assert metrics.latency_ms > 0
        assert result_frame.result.total_latency_ms > 0
        assert len(tool_calls) == 0

    @pytest.mark.asyncio
    async def test_execute_with_tool_call_success(
        self,
        runtime_executor,
        sample_execution_frame,
        mock_llm_service,
        mock_tool_manager,
    ):
        """Test successful execution with tool calls."""
        # Setup mock LLM response with tool call
        tool_call = LLMToolCall(
            name="test_tool",
            arguments={"param": "value"},
            type="function",
        )
        mock_response = LLMResult(
            content="Response with tool call",
            model="test-model",
            provider="test-provider",
            total_tokens=150,
            tool_calls=[tool_call],
            metadata={
                "provider_metadata": {
                    "latency_ms": 1.7
                }
            }
        )
        mock_llm_service.generate.return_value = mock_response

        # Setup mock stream
        async def mock_stream(task, options=None):
            from unittest.mock import MagicMock
            chunk = MagicMock()
            chunk.content = mock_response.content
            chunk.tool_calls = mock_response.tool_calls
            chunk.reasoning = mock_response.reasoning
            chunk.usage = mock_response.metadata.get("usage", {})
            chunk.metadata = mock_response.metadata.get("provider_metadata", {})
            chunk.model = mock_response.model
            chunk.provider = mock_response.provider
            chunk.finish_reason = mock_response.finish_reason
            yield chunk
        mock_llm_service.stream = mock_stream

        # Setup mock tool response
        tool_response = ToolExecutionResponse(
            success=True,
            tool_name="test_tool",
            tool_args={"param": "value"},
            result="Tool executed successfully",
            execution_time_ms=50.0,
        )
        mock_tool_manager.execute_async.return_value = tool_response

        # Execute frame
        events = []
        async for event in runtime_executor.execute(sample_execution_frame):
            events.append(event)
        result_frame = events[-1].data['frame']

        # Verify frame structure
        assert result_frame.result.status == ExecutionStatus.SUCCESS
        assert len(result_frame.actions) == 2  # LLM + tool call
        assert result_frame.actions[1].kind == ActionKind.TOOL_CALL
        assert result_frame.actions[1].name == "test_tool"

        # Verify tool metrics
        tool_calls = result_frame.get_tool_results()
        assert len(tool_calls) == 1
        tool_metrics = tool_calls[0]
        assert tool_metrics['execution_time_ms'] > 0

    @pytest.mark.asyncio
    async def test_execute_with_tool_call_failure(
        self,
        runtime_executor,
        sample_execution_frame,
        mock_llm_service,
        mock_tool_manager,
    ):
        """Test execution with failed tool call."""
        # Setup mock LLM response with tool call
        tool_call = LLMToolCall(
            name="test_tool",
            arguments={"param": "value"},
            type="function",
        )
        mock_response = LLMResult(
            content="Response with tool call",
            model="test-model",
            provider="test-provider",
            total_tokens=150,
            tool_calls=[tool_call],
            metadata={
                "provider_metadata": {
                    "latency_ms": 1.7
                }
            }
        )
        mock_llm_service.generate.return_value = mock_response

        # Setup mock stream
        async def mock_stream(task, options=None):
            from unittest.mock import MagicMock
            chunk = MagicMock()
            chunk.content = mock_response.content
            chunk.tool_calls = mock_response.tool_calls
            chunk.reasoning = mock_response.reasoning
            chunk.usage = mock_response.metadata.get("usage", {})
            chunk.metadata = mock_response.metadata.get("provider_metadata", {})
            chunk.model = mock_response.model
            chunk.provider = mock_response.provider
            chunk.finish_reason = mock_response.finish_reason
            yield chunk
        mock_llm_service.stream = mock_stream

        # Setup mock tool failure
        tool_response = ToolExecutionResponse(
            success=False,
            tool_name="test_tool",
            tool_args={"param": "value"},
            error_message="Tool execution failed",
            execution_time_ms=30.0,
        )
        mock_tool_manager.execute_async.return_value = tool_response

        # Execute frame
        events = []
        async for event in runtime_executor.execute(sample_execution_frame):
            events.append(event)
        result_frame = events[-1].data['frame']

        # Verify status is PARTIAL (LLM succeeded but tool failed)
        assert result_frame.result.status == ExecutionStatus.PARTIAL

        # Verify tool metrics still recorded
        tool_calls = result_frame.get_tool_results()
        assert len(tool_calls) == 1
        tool_metrics = tool_calls[0]
        assert tool_metrics['execution_time_ms'] > 0

    @pytest.mark.asyncio
    async def test_execute_llm_failure_blocked(
        self, runtime_executor, sample_execution_frame, mock_llm_service
    ):
        """Test execution with LLM failure due to rate limiting (BLOCKED status)."""
        # Setup mock LLM to raise rate limit error
        mock_llm_service.generate.side_effect = Exception("Rate limit exceeded")

        # Setup mock stream
        async def mock_stream(task, options=None):
            raise Exception("Rate limit exceeded")
            yield
        mock_llm_service.stream = mock_stream

        # Execute frame
        events = []
        async for event in runtime_executor.execute(sample_execution_frame):
            events.append(event)
        result_frame = events[-1].data['frame']

        # Verify BLOCKED status
        assert result_frame.result.status == ExecutionStatus.BLOCKED
        assert "Rate limit exceeded" in result_frame.result.error_message
        assert len(result_frame.actions) == 1
        assert result_frame.actions[0].kind == ActionKind.LLM_MESSAGE

    @pytest.mark.asyncio
    async def test_execute_llm_failure_failed(
        self, runtime_executor, sample_execution_frame, mock_llm_service
    ):
        """Test execution with LLM failure (FAILED status)."""
        # Setup mock LLM to raise general error
        mock_llm_service.generate.side_effect = Exception("General LLM error")

        # Setup mock stream
        del mock_llm_service.stream
        async def mock_stream(task, options=None):
            raise Exception("General LLM error")
            yield
        mock_llm_service.stream = mock_stream

        # Execute frame
        events = []
        async for event in runtime_executor.execute(sample_execution_frame):
            events.append(event)
        result_frame = events[-1].data['frame']

        # Verify FAILED status
        assert result_frame.result.status == ExecutionStatus.FAILED
        assert "General LLM error" in result_frame.result.error_message

    @pytest.mark.asyncio
    async def test_execute_tool_not_exposed(
        self, runtime_executor, sample_execution_frame, mock_llm_service
    ):
        """Test execution when tool is not exposed (should fail)."""
        # Setup mock LLM response with tool call that's not in exposed tools
        tool_call = LLMToolCall(
            name="unexposed_tool",
            arguments={"param": "value"},
            type="function",
        )
        mock_response = LLMResult(
            content="Response with unexposed tool call",
            model="test-model",
            provider="test-provider",
            total_tokens=150,
            tool_calls=[tool_call],
            metadata={
                "provider_metadata": {
                    "latency_ms": 1.7
                }
            }
        )
        mock_llm_service.generate.return_value = mock_response

        # Setup mock stream
        async def mock_stream(task, options=None):
            from unittest.mock import MagicMock
            chunk = MagicMock()
            chunk.content = mock_response.content
            chunk.tool_calls = mock_response.tool_calls
            chunk.reasoning = mock_response.reasoning
            chunk.usage = mock_response.metadata.get("usage", {})
            chunk.metadata = mock_response.metadata.get("provider_metadata", {})
            chunk.model = mock_response.model
            chunk.provider = mock_response.provider
            chunk.finish_reason = mock_response.finish_reason
            yield chunk
        mock_llm_service.stream = mock_stream

        # Execute frame
        events = []
        async for event in runtime_executor.execute(sample_execution_frame):
            events.append(event)
        result_frame = events[-1].data['frame']

        # Verify BLOCKED status due to unexposed tool
        assert result_frame.result.status == ExecutionStatus.PARTIAL
        # The error message should be in the handler_context
        assert result_frame.result.handler_context is not None
        assert "not exposed to LLM" in result_frame.result.handler_context.get('message', '')

    @pytest.mark.asyncio
    async def test_execute_with_final_answer_tool(
        self,
        runtime_executor,
        sample_execution_frame,
        mock_llm_service,
        mock_tool_manager,
    ):
        """Test execution where tool provides final answer."""
        # Setup mock LLM response with tool call
        tool_call = LLMToolCall(
            name="test_tool",
            arguments={"param": "value"},
            type="function",
        )
        mock_response = LLMResult(
            content="Response with tool call",
            model="test-model",
            provider="test-provider",
            total_tokens=150,
            tool_calls=[tool_call],
            metadata={
                "provider_metadata": {
                    "latency_ms": 1.7
                }
            }
        )
        mock_llm_service.generate.return_value = mock_response

        # Setup mock stream
        async def mock_stream(task, options=None):
            from unittest.mock import MagicMock
            chunk = MagicMock()
            chunk.content = mock_response.content
            chunk.tool_calls = mock_response.tool_calls
            chunk.reasoning = mock_response.reasoning
            chunk.usage = mock_response.metadata.get("usage", {})
            chunk.metadata = mock_response.metadata.get("provider_metadata", {})
            chunk.model = mock_response.model
            chunk.provider = mock_response.provider
            chunk.finish_reason = mock_response.finish_reason
            yield chunk
        mock_llm_service.stream = mock_stream

        # Setup mock tool response with final answer
        tool_response = ToolExecutionResponse(
            success=True,
            tool_name="test_tool",
            tool_args={"param": "value"},
            result="Final answer from tool",
            execution_time_ms=50.0,
            is_final=True,
        )
        mock_tool_manager.execute_async.return_value = tool_response

        # Execute frame
        events = []
        async for event in runtime_executor.execute(sample_execution_frame):
            events.append(event)
        result_frame = events[-1].data['frame']

        # Verify final answer from tool
        assert result_frame.result.final_answer == "Final answer from tool"
        assert result_frame.result.status == ExecutionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_mixed_success(
        self,
        runtime_executor,
        sample_execution_frame,
        mock_llm_service,
        mock_tool_manager,
    ):
        """Test execution with multiple tool calls with mixed success."""
        # Setup mock LLM response with multiple tool calls
        tool_calls = [
            LLMToolCall(name="tool1", arguments={}, type="function"),
            LLMToolCall(name="tool2", arguments={}, type="function"),
        ]
        mock_response = LLMResult(
            content="Response with multiple tool calls",
            model="test-model",
            provider="test-provider",
            total_tokens=200,
            tool_calls=tool_calls,
            metadata={
                "provider_metadata": {
                    "latency_ms": 1.7
                }
            }
        )
        mock_llm_service.generate.return_value = mock_response

        # Setup mock stream
        async def mock_stream(task, options=None):
            from unittest.mock import MagicMock
            chunk = MagicMock()
            chunk.content = mock_response.content
            chunk.tool_calls = mock_response.tool_calls
            chunk.reasoning = mock_response.reasoning
            chunk.usage = mock_response.metadata.get("usage", {})
            chunk.metadata = mock_response.metadata.get("provider_metadata", {})
            chunk.model = mock_response.model
            chunk.provider = mock_response.provider
            chunk.finish_reason = mock_response.finish_reason
            yield chunk
        mock_llm_service.stream = mock_stream

        # Setup mock tool responses - one success, one failure
        tool_responses = [
            ToolExecutionResponse(
                success=True,
                tool_name="tool1",
                tool_args={"param": "value"},
                result="Tool 1 success",
                execution_time_ms=30.0,
            ),
            ToolExecutionResponse(
                success=False,
                tool_name="tool2",
                tool_args={"param": "value"},
                error_message="Tool 2 failed",
                execution_time_ms=40.0,
            ),
        ]
        mock_tool_manager.execute_async.side_effect = tool_responses

        # Execute frame
        events = []
        async for event in runtime_executor.execute(sample_execution_frame):
            events.append(event)
        result_frame = events[-1].data['frame']

        # Verify PARTIAL status (mixed success)
        assert result_frame.result.status == ExecutionStatus.PARTIAL
        assert len(result_frame.actions) == 3  # LLM + 2 tool calls
        tool_calls = result_frame.get_tool_results()
        assert len(tool_calls) == 2

    @pytest.mark.asyncio
    async def test_execute_with_sandbox_tool_calls_expands_actions(
        self,
        runtime_executor,
        sample_prompt_context,
        mock_llm_service,
        mock_tool_manager,
    ):
        """Test sandbox tool call expansion into child actions and metrics."""
        sandbox_prompt = RenderedPrompt(
            system_messages=["System message"],
            user_messages=["User message"],
            tool_schemas=[
                {
                    "type": "function",
                    "function": {
                        "name": "execute_python_code",
                        "description": "Execute Python in sandbox",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            history=[],
        )
        frame = ExecutionFrame(
            session_id="test_session",
            iteration=1,
            prompt_context=sample_prompt_context,
            rendered_prompt=sandbox_prompt,
        )

        tool_call = LLMToolCall(
            name="execute_python_code",
            arguments={"code": "print('hi')"},
            type="function",
        )
        mock_llm_service.generate.return_value = LLMResult(
            content="Running in sandbox",
            model="test-model",
            provider="test-provider",
            total_tokens=150,
            tool_calls=[tool_call],
        )

        # Setup mock stream
        async def mock_stream(task, options=None):
            from unittest.mock import MagicMock
            mock_response = mock_llm_service.generate.return_value
            chunk = MagicMock()
            chunk.content = mock_response.content
            chunk.tool_calls = mock_response.tool_calls
            chunk.reasoning = mock_response.reasoning
            chunk.usage = mock_response.metadata.get("usage", {})
            chunk.metadata = mock_response.metadata.get("provider_metadata", {})
            chunk.model = mock_response.model
            chunk.provider = mock_response.provider
            chunk.finish_reason = mock_response.finish_reason
            yield chunk
        mock_llm_service.stream = mock_stream

        tool_calls = [
            ToolCallTrace(
                call_id="call-1",
                tool_name="tool1",
                success=True,
                input={"args": [1]},
                output="ok",
                duration_ms=12.0,
                resource_metrics=[
                    ResourceMetric(
                        type=ResourceType.CPU,
                        name="cpu_usage",
                        value=1.0,
                        unit="percent",
                    )
                ],
            ),
            ToolCallTrace(
                call_id="call-2",
                tool_name="tool2",
                success=False,
                input={"kwargs": {"q": "x"}},
                error="failed",
                duration_ms=8.0,
            ),
        ]
        envelope = ExecutionEnvelope(
            tool_name="execute_python_code",
            session_id="sandbox-session",
            success=True,
            system_metrics=[
                ResourceMetric(
                    type=ResourceType.MEMORY,
                    name="memory_rss_mb",
                    value=10.0,
                    unit="mb",
                )
            ],
        )
        tool_response = ToolExecutionResponse(
            success=True,
            tool_name="execute_python_code",
            tool_args={"code": "print('hi')"},
            execution_time_ms=50.0,
            tool_calls=tool_calls,
            envelope=envelope,
        )
        mock_tool_manager.execute_async.return_value = tool_response

        events = []
        async for event in runtime_executor.execute(frame):
            events.append(event)
        result_frame = events[-1].data['frame']

        assert result_frame.result.status == ExecutionStatus.SUCCESS
        assert len(result_frame.actions) == 2  # LLM + tool call
        child_names = [
            action.name
            for action in result_frame.actions
            if action.kind == ActionKind.TOOL_CALL
        ]
        assert "execute_python_code" in child_names
        tool_calls = result_frame.get_tool_results()
        assert len(tool_calls) == 1  # Only the main tool call

    def test_executor_initialization(
        self, mock_llm_service, mock_tool_manager, mock_context_manager
    ):
        """Test RuntimeExecutor initialization."""
        # Test with all parameters
        executor = RuntimeExecutor(
            llm_service=mock_llm_service,
            tool_manager=mock_tool_manager,
            context_manager=mock_context_manager,
        )
        assert executor._llm_service == mock_llm_service
        assert executor._tool_manager == mock_tool_manager
        assert executor._context_manager == mock_context_manager

        # Test with minimal parameters
        executor_default = RuntimeExecutor(
            llm_service=mock_llm_service,
            tool_manager=mock_tool_manager,
        )
        assert executor_default._llm_service == mock_llm_service
        assert executor_default._tool_manager == mock_tool_manager
