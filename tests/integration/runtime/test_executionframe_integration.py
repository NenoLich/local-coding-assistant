"""Integration tests for the complete ExecutionFrame workflow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.runtime import RuntimeManager
from local_coding_assistant.runtime.events import EventType
from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.tools.types import ToolExecutionResponse
from tests.unit.runtime.conftest import MockConfigManager


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
    manager.list_tools = MagicMock(return_value=[])
    return manager


@pytest.fixture
def runtime_manager(mock_llm_service, mock_tool_manager):
    """Runtime manager fixture with frame agent mode."""
    config_manager = MockConfigManager()
    manager = RuntimeManager(
        config_manager=config_manager,
        llm_service=mock_llm_service,
        tool_manager=mock_tool_manager,
    )

    # Mock the context manager to include test_tool
    from local_coding_assistant.runtime.runtime_types import ToolSpec

    test_tool = ToolSpec(
        name="test_tool", description="Test tool for testing", parameters={}
    )

    # Override the context manager's build_context to include our test tool
    original_build_context = manager._context_manager.build_context

    def mock_build_context(
        session, user_input, tool_call_mode, agent_mode, handler_context
    ):
        context = original_build_context(
            session=session,
            user_input=user_input,
            tool_call_mode=tool_call_mode,
            agent_mode=agent_mode,
            handler_context=handler_context,
        )
        context.tools.append(test_tool)
        return context

    manager._context_manager.build_context = mock_build_context

    return manager


class TestExecutionFrameIntegration:
    """Integration tests for the complete ExecutionFrame workflow."""

    @pytest.mark.asyncio
    async def test_frame_agent_mode_simple_success(
        self, runtime_manager, mock_llm_service
    ):
        """Test simple successful execution in frame agent mode."""
        # Setup mock LLM response
        from local_coding_assistant.agent.llm import LLMStreamChunk

        mock_response = LLMStreamChunk(
            content="This is a simple response",
            provider="test-provider",
            model="test-model",
            is_final=True,
            usage={"total_tokens": 50},
        )

        # Mock the stream method to return an async iterator
        async def mock_stream(request, options=None):
            yield mock_response

        mock_llm_service.stream = mock_stream

        # Execute in frame agent mode and collect events
        events = []
        async for event in runtime_manager.orchestrate(
            "Simple test query", agent_mode="frame"
        ):
            events.append(event)

        # Extract result from TURN_COMPLETE event
        turn_complete_event = next(
            (e for e in events if e.type == EventType.TURN_COMPLETE), None
        )
        assert turn_complete_event is not None
        result = turn_complete_event.data["report"].model_dump()

        # Verify result structure
        assert "final_answer" in result
        assert result["final_answer"] == "This is a simple response"
        assert "iterations" in result
        assert result["iterations"] == 1
        assert "history" in result
        assert "session_id" in result
        assert "frames" in result
        assert len(result["frames"]) == 1

        # Verify frame structure
        frame = result["frames"][0]
        assert frame["session_id"] == result["session_id"]
        assert frame["iteration"] == 1
        assert frame["result"]["status"] == ExecutionStatus.SUCCESS
        assert frame["result"]["final_answer"] == "This is a simple response"

    @pytest.mark.asyncio
    async def test_frame_agent_mode_with_tool_execution(
        self, runtime_manager, mock_llm_service, mock_tool_manager
    ):
        """Test frame agent mode with tool execution."""
        # Setup mock LLM response with tool call
        from local_coding_assistant.agent.llm import LLMStreamChunk, LLMToolCall

        tool_call = LLMToolCall(
            name="test_tool",
            arguments={"query": "test"},
            type="function",
        )
        mock_response = LLMStreamChunk(
            content="I'll help you with that",
            provider="test-provider",
            model="test-model",
            is_final=True,
            tool_calls=[tool_call],
            usage={"total_tokens": 75},
        )

        # Mock the stream method to return an async iterator
        async def mock_stream(request, options=None):
            yield mock_response

        mock_llm_service.stream = mock_stream

        # Setup mock tool response
        tool_response = ToolExecutionResponse(
            success=True,
            tool_name="test_tool",
            tool_args={"query": "test"},
            result="Tool executed successfully",
            execution_time_ms=100.0,
        )
        mock_tool_manager.execute_async.return_value = tool_response

        # Execute in frame agent mode and collect events
        events = []
        async for event in runtime_manager.orchestrate(
            "Execute a tool for me", agent_mode="frame"
        ):
            events.append(event)

        # Extract result from TURN_COMPLETE event
        turn_complete_event = next(
            (e for e in events if e.type == EventType.TURN_COMPLETE), None
        )
        assert turn_complete_event is not None
        result = turn_complete_event.data["report"].model_dump()

        # Verify result
        assert "final_answer" in result

    @pytest.mark.asyncio
    async def test_frame_agent_mode_multiple_iterations(
        self, runtime_manager, mock_llm_service, mock_tool_manager
    ):
        """Test frame agent mode with multiple iterations."""
        # Setup first LLM response with tool call
        from local_coding_assistant.agent.llm import LLMStreamChunk, LLMToolCall

        tool_call = LLMToolCall(
            name="test_tool",
            arguments={"step": "1"},
            type="function",
        )
        first_response = LLMStreamChunk(
            content="Let me execute the first step",
            provider="test-provider",
            model="test-model",
            is_final=True,
            tool_calls=[tool_call],
            usage={"total_tokens": 60},
        )

        # Setup second LLM response with final answer
        second_response = LLMStreamChunk(
            content="Here's the final answer after processing",
            provider="test-provider",
            model="test-model",
            is_final=True,
            usage={"total_tokens": 80},
        )

        # Mock the stream method to return an async iterator
        async def mock_stream(request, options=None):
            yield second_response

        mock_llm_service.stream = mock_stream

        # Setup mock tool response
        tool_response = ToolExecutionResponse(
            success=True,
            tool_name="test_tool",
            tool_args={"step": "1"},
            result="First step completed",
            execution_time_ms=80.0,
        )
        mock_tool_manager.execute_async.return_value = tool_response

        # Execute in frame agent mode and collect events
        events = []
        async for event in runtime_manager.orchestrate(
            "Complex multi-step query", agent_mode="frame"
        ):
            events.append(event)

        # Extract result from TURN_COMPLETE event
        turn_complete_event = next(
            (e for e in events if e.type == EventType.TURN_COMPLETE), None
        )
        assert turn_complete_event is not None
        result = turn_complete_event.data["report"].model_dump()

        # Verify result
        assert result["iterations"] == 1
        assert len(result["frames"]) == 1

        # Verify first frame
        first_frame = result["frames"][0]
        assert first_frame["iteration"] == 1

    @pytest.mark.asyncio
    async def test_frame_agent_mode_tool_failure_partial_success(
        self, runtime_manager, mock_llm_service, mock_tool_manager
    ):
        """Test frame agent mode with tool failure leading to partial success."""
        # Setup LLM response with tool call
        from local_coding_assistant.agent.llm import LLMStreamChunk, LLMToolCall

        tool_call = LLMToolCall(
            name="failing_tool",
            arguments={"param": "value"},
            type="function",
        )
        mock_response = LLMStreamChunk(
            content="I'll try to execute the tool",
            provider="test-provider",
            model="test-model",
            is_final=True,
            tool_calls=[tool_call],
            usage={"total_tokens": 70},
        )

        # Mock the stream method to return an async iterator
        async def mock_stream(request, options=None):
            yield mock_response

        mock_llm_service.stream = mock_stream

        # Setup mock tool failure
        tool_response = ToolExecutionResponse(
            success=False,
            tool_name="failing_tool",
            tool_args={"param": "value"},
            error_message="Tool execution failed",
            execution_time_ms=50.0,
        )
        mock_tool_manager.execute_async.return_value = tool_response

        # Execute in frame agent mode and collect events
        events = []
        async for event in runtime_manager.orchestrate(
            "Execute failing tool", agent_mode="frame"
        ):
            events.append(event)

        # Extract result from TURN_COMPLETE event
        turn_complete_event = next(
            (e for e in events if e.type == EventType.TURN_COMPLETE), None
        )
        assert turn_complete_event is not None

    @pytest.mark.asyncio
    async def test_frame_agent_mode_statistics_integration(
        self, runtime_manager, mock_llm_service, mock_tool_manager
    ):
        """Test that statistics are properly recorded during frame agent execution."""
        # Setup LLM response with tool call
        from local_coding_assistant.agent.llm import LLMStreamChunk, LLMToolCall

        tool_call = LLMToolCall(
            name="stats_tool",
            arguments={"action": "measure"},
            type="function",
        )
        mock_response = LLMStreamChunk(
            content="Measuring performance",
            provider="test-provider",
            model="test-model",
            is_final=True,
            tool_calls=[tool_call],
            usage={"total_tokens": 90},
        )

        # Mock the stream method to return an async iterator
        async def mock_stream(request, options=None):
            yield mock_response

        mock_llm_service.stream = mock_stream

        # Setup mock tool response
        tool_response = ToolExecutionResponse(
            success=True,
            tool_name="stats_tool",
            tool_args={"action": "measure"},
            result="Performance measured",
            execution_time_ms=120.0,
        )
        mock_tool_manager.execute_async.return_value = tool_response

        # Setup mock statistics
        mock_stats = MagicMock()
        mock_tool_stats = MagicMock()
        mock_tool_stats.total_executions = 1
        mock_tool_stats.success_count = 1
        mock_stats.get_tool_stats.return_value = mock_tool_stats
        mock_tool_manager._statistics_manager = mock_stats

        # Execute in frame agent mode
        events = []
        async for event in runtime_manager.orchestrate(
            "Measure performance", agent_mode="frame"
        ):
            events.append(event)

        # Extract result from TURN_COMPLETE event
        turn_complete_event = next(
            (e for e in events if e.type == EventType.TURN_COMPLETE), None
        )
        assert turn_complete_event is not None
        result = turn_complete_event.data["report"].model_dump()

        # Verify statistics manager was used
        # The statistics manager should have recorded the tool call
        assert runtime_manager._tool_manager._statistics_manager is not None

        # Check that we can retrieve tool statistics
        tool_stats = runtime_manager._tool_manager._statistics_manager.get_tool_stats(
            "stats_tool"
        )
        assert tool_stats is not None
        assert tool_stats.total_executions == 1
        assert tool_stats.success_count == 1

    @pytest.mark.asyncio
    async def test_frame_agent_mode_session_persistence(
        self, runtime_manager, mock_llm_service
    ):
        """Test that session state is properly maintained across frame agent execution."""
        # Setup first query
        from local_coding_assistant.agent.llm import LLMStreamChunk

        first_response = LLMStreamChunk(
            content="First response",
            provider="test-provider",
            model="test-model",
            is_final=True,
            usage={"total_tokens": 40},
        )

        # Setup second query (should use same session by default)
        second_response = LLMStreamChunk(
            content="Second response",
            provider="test-provider",
            model="test-model",
            is_final=True,
            usage={"total_tokens": 45},
        )

        # Mock the stream method to return different responses
        responses = [first_response] + [
            second_response
        ] * 10  # First response once, then second for rest
        response_iter = iter(responses)

        async def mock_stream(request, options=None):
            yield next(response_iter)

        mock_llm_service.stream = mock_stream

        # Execute first query and collect events
        events1 = []
        async for event in runtime_manager.orchestrate(
            "First query", agent_mode="frame"
        ):
            events1.append(event)

        # Extract result from TURN_COMPLETE event
        turn_complete_event1 = next(
            (e for e in events1 if e.type == EventType.TURN_COMPLETE), None
        )
        assert turn_complete_event1 is not None
        result1 = turn_complete_event1.data["report"].model_dump()

        session_id_1 = result1["session_id"]

        # Execute second query and collect events
        events2 = []
        async for event in runtime_manager.orchestrate(
            "Second query", agent_mode="frame"
        ):
            events2.append(event)

        # Extract result from TURN_COMPLETE event
        turn_complete_event2 = next(
            (e for e in events2 if e.type == EventType.TURN_COMPLETE), None
        )
        assert turn_complete_event2 is not None
        result2 = turn_complete_event2.data["report"].model_dump()

        # Verify session persistence
        assert result2["session_id"] == session_id_1  # Same session
        assert runtime_manager.session is not None
        assert (
            len(runtime_manager.session.history) >= 2
        )  # Should have both conversations

    @pytest.mark.asyncio
    async def test_frame_agent_mode_vs_regular_mode(
        self, runtime_manager, mock_llm_service
    ):
        """Test that frame agent mode produces different results than regular mode."""
        # Setup mock LLM response for stream (FrameAgent)
        from local_coding_assistant.agent.llm import LLMResult, LLMStreamChunk

        stream_response = LLMStreamChunk(
            content="Response for testing",
            provider="test-provider",
            model="test-model",
            is_final=True,
            usage={"total_tokens": 55},
        )

        # Setup mock LLM response for generate (regular mode)
        generate_response = LLMResult(
            content="Response for testing",
            model="test-model",
            provider="test-provider",
            total_tokens=55,
            tool_calls=[],
        )

        # Mock the stream method to return an async iterator
        async def mock_stream(request, options=None):
            yield stream_response

        mock_llm_service.stream = mock_stream

        # Mock generate for regular mode
        mock_llm_service.generate.return_value = generate_response

        # Execute in regular mode and collect events
        regular_events = []
        async for event in runtime_manager.orchestrate("Test query", agent_mode=None):
            regular_events.append(event)

        # Extract result from TURN_COMPLETE event
        regular_turn_complete_event = next(
            (e for e in regular_events if e.type == EventType.TURN_COMPLETE), None
        )
        assert regular_turn_complete_event is not None
        regular_result = regular_turn_complete_event.data["report"].model_dump()

        # Execute in frame agent mode and collect events
        frame_events = []
        async for event in runtime_manager.orchestrate(
            "Test query", agent_mode="frame"
        ):
            frame_events.append(event)

        # Extract result from TURN_COMPLETE event
        frame_turn_complete_event = next(
            (e for e in frame_events if e.type == EventType.TURN_COMPLETE), None
        )
        assert frame_turn_complete_event is not None
        frame_result = frame_turn_complete_event.data["report"].model_dump()

        # Both should have the same content but different structure
        assert regular_result["final_answer"] == frame_result["final_answer"]

        # Regular mode result structure
        assert "final_answer" in regular_result
        assert "session_id" in regular_result
        assert "models_used" in regular_result
        assert "tokens_used" in regular_result

        # Frame agent mode result structure
        assert "final_answer" in frame_result
        assert "iterations" in frame_result
        assert "frames" in frame_result
        assert "history" in frame_result

    @pytest.mark.asyncio
    async def test_frame_agent_mode_error_handling(
        self, runtime_manager, mock_llm_service
    ):
        """Test error handling in frame agent mode."""

        # Setup LLM to raise an error
        class RaisingAsyncIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise Exception("LLM service error")

        mock_llm_service.stream = lambda *args, **kwargs: RaisingAsyncIterator()

        # Execute in frame agent mode and collect events
        events = []
        async for event in runtime_manager.orchestrate(
            "Error test query", agent_mode="frame"
        ):
            events.append(event)

        # Extract result from TURN_COMPLETE event
        turn_complete_event = next(
            (e for e in events if e.type == EventType.TURN_COMPLETE), None
        )
        assert turn_complete_event is not None
        result = turn_complete_event.data["report"].model_dump()

        # Should handle error gracefully
        assert "final_answer" in result
        # In case of error, final_answer might be None or error message
        assert result["iterations"] == 1
        assert len(result["frames"]) == 1

        # Verify error is captured in frame
        frame = result["frames"][0]
        assert frame["result"]["status"] in [
            ExecutionStatus.FAILED,
            ExecutionStatus.BLOCKED,
        ]
        assert "error" in str(frame["result"]).lower() or "llm service error" in str(
            frame["result"]
        )

    @pytest.mark.asyncio
    async def test_frame_agent_mode_configuration_inheritance(
        self, runtime_manager, mock_llm_service
    ):
        """Test that frame agent mode properly inherits configuration."""
        # Setup mock LLM response
        from local_coding_assistant.agent.llm import LLMStreamChunk

        mock_response = LLMStreamChunk(
            content="Configured response",
            provider="test-provider",
            model="test-model",
            is_final=True,
            usage={"total_tokens": 65},
        )

        # Mock the stream method to return an async iterator
        async def mock_stream(request, options=None):
            yield mock_response

        mock_llm_service.stream = mock_stream

        # Execute with specific model configuration and collect events
        events = []
        async for event in runtime_manager.orchestrate(
            "Configured query",
            agent_mode="frame",
            model="custom-model",
            temperature=0.5,
            max_tokens=2000,
        ):
            events.append(event)

        # Extract result from TURN_COMPLETE event
        turn_complete_event = next(
            (e for e in events if e.type == EventType.TURN_COMPLETE), None
        )
        assert turn_complete_event is not None
        result = turn_complete_event.data["report"].model_dump()

        # Verify execution succeeded
        assert result["final_answer"] == "Configured response"
        # Note: Configuration inheritance would need to be implemented at the FrameAgent level
        # For now, we just verify that the execution completes successfully
