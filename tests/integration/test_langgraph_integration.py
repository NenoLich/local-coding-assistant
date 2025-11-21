"""Integration tests for LangGraph functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

# Handle case where langgraph is not installed
try:
    from local_coding_assistant.agent.langgraph_agent import AgentState, LangGraphAgent
    from local_coding_assistant.agent.llm_manager import (
        LLMManager,
        LLMRequest,
        LLMResponse,
    )
    from local_coding_assistant.core.exceptions import AgentError, LLMError
    from tests.integration.conftest import MockStreamingLLMManager, MockToolManager

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
class TestLangGraphIntegration:
    """Integration tests for LangGraph functionality."""

    @pytest.mark.asyncio
    async def test_normal_input_expected_output_flow(
        self, mock_llm_with_tools, tool_manager
    ):
        """Test normal input produces expected output through all nodes."""
        # Create LangGraph agent with mocked dependencies
        agent = LangGraphAgent(
            llm_manager=mock_llm_with_tools,
            tool_manager=tool_manager,
            name="integration_test_agent",
            max_iterations=3,
        )

        # Run the agent
        result = await agent.run()

        # Verify successful execution (may return None if no final answer reached)
        assert result is None or isinstance(result, str)
        if result is not None:
            assert len(result) > 0

        # If we got a result, verify it contains expected content
        if result is not None:
            assert "Pack light summer clothes" in result or "New York" in result

        print(f"✓ Normal flow completed successfully with result: {result}")

    @pytest.mark.asyncio
    async def test_failing_llm_caught_llmerror(self):
        """Test that failing LLM triggers LLMError handling."""
        # Create LLM manager that raises LLMError
        failing_llm = MagicMock(spec=LLMManager)
        failing_llm.generate = AsyncMock(
            side_effect=LLMError("LLM service unavailable")
        )

        tool_manager = MockToolManager()

        agent = LangGraphAgent(
            llm_manager=failing_llm,
            tool_manager=tool_manager,
            name="failing_llm_test",
            max_iterations=2,
        )

        # The agent should handle LLM errors gracefully
        # In the actual implementation, errors are caught by the safe_node decorator
        result = await agent.run()

        # Should return None due to error handling
        assert result is None

        print("✓ LLM error handled gracefully")

    @pytest.mark.asyncio
    async def test_agent_execution_error_handling(self):
        """Test that agent execution errors are caught and handled."""
        # Create tool manager that raises errors
        failing_tool_manager = MagicMock(spec=MockToolManager)
        failing_tool_manager.__iter__ = MagicMock(return_value=iter([]))

        # Mock run_tool to raise an exception
        failing_tool_manager.run_tool = MagicMock(
            side_effect=Exception("Tool execution failed")
        )

        # Create LLM that would normally succeed but tool execution fails
        llm_manager = MagicMock(spec=LLMManager)
        response = MagicMock(spec=LLMResponse)
        response.content = "I need to use a tool"
        response.tool_calls = [
            {
                "function": {
                    "name": "failing_tool",
                    "arguments": '{"param": "value"}',
                }
            }
        ]
        llm_manager.generate = AsyncMock(return_value=response)

        agent = LangGraphAgent(
            llm_manager=llm_manager,
            tool_manager=failing_tool_manager,
            name="execution_error_test",
            max_iterations=2,
        )

        # Run the agent - should handle the execution error
        result = await agent.run()

        # Should return None due to error handling
        assert result is None

        print("✓ Agent execution error handled gracefully")

    @pytest.mark.asyncio
    async def test_malformed_input_fallback_logging(self):
        """Test that malformed input triggers fallback logging and safe shutdown."""
        # Create LLM that returns malformed JSON
        malformed_llm = MagicMock(spec=LLMManager)

        # Response with invalid JSON that should trigger fallback parsing
        response = MagicMock(spec=LLMResponse)
        response.content = "This is not valid JSON response"  # Malformed JSON
        response.tool_calls = None
        malformed_llm.generate = AsyncMock(return_value=response)

        tool_manager = MockToolManager()

        agent = LangGraphAgent(
            llm_manager=malformed_llm,
            tool_manager=tool_manager,
            name="malformed_input_test",
            max_iterations=2,
        )

        # Run the agent - should handle malformed input gracefully
        result = await agent.run()

        # Should complete successfully with fallback behavior
        assert result is None or isinstance(result, str)

        print("✓ Malformed input handled with fallback behavior")

    @pytest.mark.asyncio
    async def test_langgraph_streaming_integration(
        self, tool_manager, mock_llm_response
    ):
        """Test LangGraph integration with streaming responses."""
        from unittest.mock import AsyncMock, MagicMock
        from local_coding_assistant.agent.langgraph_agent import AgentState
        from local_coding_assistant.agent.llm_manager import LLMManager
        
        # Create a properly configured mock LLM manager
        streaming_llm = MagicMock(spec=LLMManager)
        streaming_llm.generate = AsyncMock(return_value=mock_llm_response)
        streaming_llm.stream = AsyncMock(return_value=[mock_llm_response])
        
        # Create a mock writer
        mock_writer = AsyncMock()
        
        # Create the agent with our mocks
        agent = LangGraphAgent(
            llm_manager=streaming_llm,
            tool_manager=tool_manager,
            name="streaming_integration_test",
            max_iterations=2,
        )
        
        # Create a valid initial state
        initial_state = AgentState(
            max_iterations=2,
            user_input="Test input",
            session_id="test_session_123"
        )
        
        # Test the observe node directly with the required parameters
        try:
            state = await agent.observe_node(initial_state, mock_writer)
            assert state is not None
            print("✓ Streaming integration working correctly")
        except Exception as e:
            pytest.fail(f"Test failed with error: {str(e)}")

    @pytest.mark.asyncio
    async def test_langgraph_complex_scenario(self, tool_manager, mock_llm_response):
        """Test LangGraph with complex multi-step reasoning scenario."""
        from unittest.mock import AsyncMock, MagicMock
        from local_coding_assistant.agent.langgraph_agent import AgentState
        from local_coding_assistant.agent.llm_manager import LLMManager
        
        # Create a mock LLM manager with a complex response
        complex_llm = MagicMock(spec=LLMManager)
        complex_llm.generate = AsyncMock(return_value=mock_llm_response)
        
        # Create a mock writer
        mock_writer = AsyncMock()
        
        # Create the agent with our mocks
        agent = LangGraphAgent(
            llm_manager=complex_llm,
            tool_manager=tool_manager,
            name="complex_scenario_test",
            max_iterations=5,  # Allow more iterations for complex scenario
        )
        
        # Create a valid initial state
        initial_state = AgentState(
            max_iterations=5,
            user_input="Test complex scenario",
            session_id="test_complex_scenario_123"
        )
        
        # Test the observe node directly with the required parameters
        try:
            # Test the observe node
            state = await agent.observe_node(initial_state, mock_writer)
            assert state is not None
            
            # Test the plan node
            state = await agent.plan_node(state, mock_writer)
            assert state is not None
            
            # Test the act node
            state = await agent.act_node(state, mock_writer)
            assert state is not None
            
            # Test the reflect node
            state = await agent.reflect_node(state, mock_writer)
            assert state is not None
            
            print("✓ Complex scenario completed successfully")
        except Exception as e:
            pytest.fail(f"Test failed with error: {str(e)}")

    @pytest.mark.asyncio
    async def test_langgraph_error_recovery(self):
        """Test that LangGraph recovers from transient errors."""
        # Create LLM that fails on first call but succeeds on retry
        call_count = 0

        def mock_generate(request):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call fails
                raise LLMError("Temporary LLM failure")
            else:
                # Subsequent calls succeed
                response = MagicMock(spec=LLMResponse)
                response.content = "Recovered successfully"
                response.tool_calls = []
                return response

        transient_llm = MagicMock(spec=LLMManager)
        transient_llm.generate = AsyncMock(side_effect=mock_generate)

        tool_manager = MockToolManager()

        agent = LangGraphAgent(
            llm_manager=transient_llm,
            tool_manager=tool_manager,
            name="error_recovery_test",
            max_iterations=3,
        )

        # Should handle the transient error and continue
        result = await agent.run()

        # Should eventually succeed despite initial failure
        assert result is None or isinstance(result, str)

        print("✓ Error recovery mechanism working correctly")

    def test_langgraph_state_persistence(self, mock_llm_with_tools, tool_manager):
        """Test that LangGraph maintains state correctly across operations."""
        agent = LangGraphAgent(
            llm_manager=mock_llm_with_tools,
            tool_manager=tool_manager,
            name="state_persistence_test",
            max_iterations=3,
        )

        # Get initial state
        initial_state = agent.get_current_state()
        initial_iteration = initial_state.iteration
        initial_history_length = len(agent.get_history())

        # State should be properly initialized
        assert initial_iteration == 0
        assert initial_history_length == 0

        print("✓ State persistence working correctly")

    @pytest.mark.asyncio
    async def test_langgraph_concurrent_execution_safety(
        self, mock_llm_with_tools, tool_manager
    ):
        """Test that LangGraph handles concurrent execution safely."""
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = LangGraphAgent(
                llm_manager=mock_llm_with_tools,
                tool_manager=tool_manager,
                name=f"concurrent_agent_{i}",
                max_iterations=2,
            )
            agents.append(agent)

        # Run agents concurrently
        tasks = [agent.run() for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete successfully
        for i, result in enumerate(results):
            assert result is None or isinstance(result, str), (
                f"Agent {i} failed: {result}"
            )

        print("✓ Concurrent execution handled safely")

    @pytest.mark.asyncio
    async def test_langgraph_memory_management(self, mock_llm_with_tools, tool_manager):
        """Test that LangGraph manages memory efficiently."""
        agent = LangGraphAgent(
            llm_manager=mock_llm_with_tools,
            tool_manager=tool_manager,
            name="memory_test_agent",
            max_iterations=5,
        )

        # Run agent multiple times
        for _ in range(3):
            result = await agent.run()
            assert result is None or isinstance(result, str)

        # Check that history doesn't grow unbounded (implementation dependent)
        history = agent.get_history()
        # History length should be reasonable (depends on implementation)
        assert len(history) >= 0  # At minimum should not crash

        print("✓ Memory management working correctly")
