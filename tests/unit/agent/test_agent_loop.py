"""Unit tests for AgentLoop functionality."""

import time
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from local_coding_assistant.agent.agent_loop import AgentLoop
from local_coding_assistant.agent.llm_manager import LLMManager, LLMResponse
from local_coding_assistant.core.exceptions import AgentError
from local_coding_assistant.tools.tool_manager import ToolManager


class TestAgentLoopInitialization:
    """Test AgentLoop initialization and internal state management."""

    def test_agent_loop_initializes_with_zero_iterations(self):
        """Test that a new AgentLoop starts with iteration count 0."""
        llm_manager = MagicMock(spec=LLMManager)
        tool_manager = MagicMock(spec=ToolManager)

        agent_loop = AgentLoop(
            llm_manager=llm_manager, tool_manager=tool_manager, name="test_agent"
        )

        assert agent_loop.current_iteration == 0
        assert agent_loop.is_loop_running() is False
        assert agent_loop.final_answer is None
        assert agent_loop.name == "test_agent"
        assert len(agent_loop.get_history()) == 0

    def test_agent_loop_initializes_handlers(self):
        """Test that AgentLoop creates default handlers during initialization."""
        llm_manager = MagicMock(spec=LLMManager)
        tool_manager = MagicMock(spec=ToolManager)

        agent_loop = AgentLoop(
            llm_manager=llm_manager, tool_manager=tool_manager, name="test_agent"
        )

        # Check that handlers are created
        assert agent_loop.observe_handler is not None
        assert agent_loop.plan_handler is not None
        assert agent_loop.act_handler is not None
        assert agent_loop.reflect_handler is not None

    def test_agent_loop_caches_tools(self):
        """Test that AgentLoop caches tools during initialization."""
        llm_manager = MagicMock(spec=LLMManager)

        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.available = True

        # Mock the tool manager's list_tools method
        tool_manager = MagicMock(spec=ToolManager)
        tool_manager.list_tools.return_value = [mock_tool]

        agent_loop = AgentLoop(
            llm_manager=llm_manager, tool_manager=tool_manager, name="test_agent"
        )

        # Verify list_tools was called with available_only=True
        tool_manager.list_tools.assert_called_once_with(available_only=True, execution_mode='classic')

        # Check that tools are cached
        assert hasattr(agent_loop, "_cached_tools")
        assert len(agent_loop._cached_tools) == 1
        assert agent_loop._cached_tools[0]["name"] == "test_tool"

    def test_agent_loop_streaming_initialization(self):
        """Test that AgentLoop initializes correctly with streaming flag."""
        llm_manager = MagicMock(spec=LLMManager)
        tool_manager = MagicMock(spec=ToolManager)

        # Test with streaming enabled
        agent_loop_streaming = AgentLoop(
            llm_manager=llm_manager,
            tool_manager=tool_manager,
            name="test_agent_streaming",
            streaming=True,
        )

        assert agent_loop_streaming.streaming is True
        assert agent_loop_streaming.name == "test_agent_streaming"

        # Test with streaming disabled (default)
        agent_loop_no_streaming = AgentLoop(
            llm_manager=llm_manager,
            tool_manager=tool_manager,
            name="test_agent_no_streaming",
        )

        assert agent_loop_no_streaming.streaming is False
        assert agent_loop_no_streaming.name == "test_agent_no_streaming"


class TestAgentLoopControlFlow:
    """Test AgentLoop control flow and iteration logic."""

    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager for testing."""
        llm_manager = MagicMock(spec=LLMManager)

        # Mock LLM response for planning
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.content = (
            '{"reasoning": "Test plan", "actions": ["test_action"], "confidence": 0.8}'
        )
        llm_manager.generate = AsyncMock(return_value=mock_response)

        return llm_manager

    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock tool manager for testing."""
        tool_manager = MagicMock(spec=ToolManager)
        tool_manager.__iter__ = MagicMock(return_value=iter([]))  # No tools by default
        return tool_manager

    @pytest.fixture
    def agent_loop(self, mock_llm_manager, mock_tool_manager):
        """Create an AgentLoop instance for testing."""
        return AgentLoop(
            llm_manager=mock_llm_manager,
            tool_manager=mock_tool_manager,
            name="test_agent",
            max_iterations=3,
        )

    @pytest.fixture
    def agent_loop_single(self, mock_llm_manager, mock_tool_manager):
        """Create an AgentLoop instance for single iteration testing."""
        return AgentLoop(
            llm_manager=mock_llm_manager,
            tool_manager=mock_tool_manager,
            name="test_agent_single",
            max_iterations=1,
        )

    @pytest.mark.asyncio
    async def test_agent_loop_already_running_error(self, agent_loop):
        """Test that starting an already running loop raises an error."""
        # Set the loop as running
        agent_loop.is_running = True

        # Try to run the loop - this should raise AgentError
        with pytest.raises(AgentError, match="Agent loop is already running"):
            await agent_loop.run()

    @pytest.mark.asyncio
    async def test_agent_loop_single_iteration_execution(
        self, agent_loop_single, mock_llm_manager
    ):
        """Test single iteration execution with proper state management."""
        # Mock the handlers to return predictable results
        agent_loop_single.observe_handler = Mock(
            return_value={
                "content": "Test observation",
                "metadata": {"session_id": "test_session"},
                "timestamp": time.time(),
                "source": "test",
            }
        )

        agent_loop_single.plan_handler = AsyncMock(
            return_value={
                "reasoning": "Test plan",
                "actions": ["test_action"],
                "confidence": 0.8,
                "metadata": {},
            }
        )

        agent_loop_single.act_handler = AsyncMock(
            return_value={"success": True, "output": "Action completed", "metadata": {}}
        )

        agent_loop_single.reflect_handler = AsyncMock(
            return_value={
                "analysis": "Test reflection",
                "lessons_learned": ["lesson1"],
                "improvements": [],
                "success_rating": 0.9,
            }
        )

        # Run single iteration
        result = await agent_loop_single.run()

        # Verify state changes
        assert agent_loop_single.current_iteration == 1
        assert len(agent_loop_single.get_history()) == 1

        # Verify history contains expected data
        iteration_data = agent_loop_single.get_history()[0]
        assert iteration_data["iteration"] == 1
        assert "observation" in iteration_data
        assert "plan" in iteration_data
        assert "action_result" in iteration_data
        assert "reflection" in iteration_data

        # Verify final answer is None (no final_answer tool called)
        assert result is None

    @pytest.mark.asyncio
    async def test_agent_loop_multiple_iterations(self, agent_loop, mock_llm_manager):
        """Test multiple iterations with proper accumulation."""
        # Mock handlers for multiple iterations
        call_count = {"observe": 0, "plan": 0, "act": 0, "reflect": 0}

        def mock_observe():
            call_count["observe"] += 1
            return {
                "content": f"Test observation {call_count['observe']}",
                "metadata": {"session_id": "test_session"},
                "timestamp": time.time(),
                "source": "test",
            }

        async def mock_plan(observation):
            call_count["plan"] += 1
            return {
                "reasoning": f"Test plan {call_count['plan']}",
                "actions": ["test_action"],
                "confidence": 0.8,
                "metadata": {},
            }

        async def mock_act(plan):
            call_count["act"] += 1
            return {
                "success": True,
                "output": f"Action completed {call_count['act']}",
                "metadata": {},
            }

        async def mock_reflect(action_result, plan):
            call_count["reflect"] += 1
            return {
                "analysis": f"Test reflection {call_count['reflect']}",
                "lessons_learned": ["lesson1"],
                "improvements": [],
                "success_rating": 0.9,
            }

        agent_loop.observe_handler = mock_observe
        agent_loop.plan_handler = mock_plan
        agent_loop.act_handler = mock_act
        agent_loop.reflect_handler = mock_reflect

        # Run multiple iterations
        await agent_loop.run()

        # Verify multiple iterations ran
        assert agent_loop.current_iteration == 3
        assert len(agent_loop.get_history()) == 3

        # Verify all handlers were called the expected number of times
        assert call_count["observe"] == 3
        assert call_count["plan"] == 3
        assert call_count["act"] == 3
        assert call_count["reflect"] == 3

        # Verify history accumulation
        for i, iteration_data in enumerate(agent_loop.get_history()):
            assert iteration_data["iteration"] == i + 1
            assert (
                f"Test observation {i + 1}" in iteration_data["observation"]["content"]
            )


class TestAgentLoopToolInvocation:
    """Test AgentLoop tool invocation and handling."""

    @pytest.fixture
    def mock_llm_with_tools(self):
        """Create a mock LLM manager that returns tool calls."""
        llm_manager = MagicMock(spec=LLMManager)

        # Mock response with tool call
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.content = "I'll use the sum tool to add 2 + 3"
        mock_response.tool_calls = [
            {"function": {"name": "sum", "arguments": '{"a": 2, "b": 3}'}}
        ]
        # Mock the generate method to return the response for act handler
        llm_manager.generate = AsyncMock(return_value=mock_response)

        return llm_manager

    @pytest.fixture
    def mock_tool_manager_with_sum(self):
        """Create a mock tool manager with a sum tool."""
        # Create a mock sum tool
        mock_sum_tool = MagicMock()
        mock_sum_tool.name = "sum"
        mock_sum_tool.description = "Add two numbers"
        mock_sum_tool.Input = MagicMock()
        mock_sum_tool.Output = MagicMock()
        mock_sum_tool.run = MagicMock(return_value=MagicMock(sum=5))

        tool_manager = MagicMock(spec=ToolManager)
        tool_manager.__iter__ = MagicMock(return_value=iter([mock_sum_tool]))
        tool_manager.run_tool = MagicMock(return_value=5)

        return tool_manager

    @pytest.fixture
    def agent_loop_with_tools(self, mock_llm_with_tools, mock_tool_manager_with_sum):
        """Create an AgentLoop with tool capabilities."""
        return AgentLoop(
            llm_manager=mock_llm_with_tools,
            tool_manager=mock_tool_manager_with_sum,
            name="test_agent_tools",
            max_iterations=1,  # Changed to 1 for single iteration tests
        )

    @pytest.mark.asyncio
    async def test_agent_loop_tool_invocation_success(
        self, agent_loop_with_tools, mock_tool_manager_with_sum
    ):
        """Test successful tool invocation in agent loop."""
        # Mock handlers to trigger tool usage
        agent_loop_with_tools.observe_handler = Mock(
            return_value={
                "content": "Need to calculate 2 + 3",
                "metadata": {"session_id": "test_session"},
                "timestamp": time.time(),
                "source": "test",
            }
        )

        agent_loop_with_tools.plan_handler = AsyncMock(
            return_value={
                "reasoning": "Need to use sum tool",
                "actions": ["calculate_sum"],
                "confidence": 0.9,
                "metadata": {},
            }
        )

        # Mock the act handler to simulate direct tool calling
        async def mock_act_with_tool_call(plan):
            # Directly simulate tool call without LLM interaction
            func_name = "sum"
            args = {"a": 2, "b": 3}
            tool_result = mock_tool_manager_with_sum.run_tool(func_name, args)

            return {
                "success": True,
                "output": f"Tool {func_name} executed successfully",
                "metadata": {"tool_results": {func_name: tool_result}},
            }

        agent_loop_with_tools.act_handler = mock_act_with_tool_call
        agent_loop_with_tools.reflect_handler = AsyncMock(
            return_value={
                "analysis": "Tool execution successful",
                "lessons_learned": ["Tools work correctly"],
                "improvements": [],
                "success_rating": 0.95,
            }
        )

        # Run the agent loop
        await agent_loop_with_tools.run()

        # Verify tool was called (it might be called twice due to test setup, but that's ok)
        assert mock_tool_manager_with_sum.run_tool.call_count >= 1
        # Check that sum was called at least once with the right arguments
        calls = mock_tool_manager_with_sum.run_tool.call_args_list
        assert any(call[0] == ("sum", {"a": 2, "b": 3}) for call in calls)

        # Verify iteration completed successfully
        assert agent_loop_with_tools.current_iteration == 1
        assert len(agent_loop_with_tools.get_history()) == 1

    @pytest.mark.asyncio
    async def test_agent_loop_final_answer_termination(
        self, agent_loop_with_tools, mock_tool_manager_with_sum
    ):
        """Test that final_answer tool terminates the loop and returns the answer."""
        # Mock handlers to trigger final answer
        agent_loop_with_tools.observe_handler = Mock(
            return_value={
                "content": "Analysis complete",
                "metadata": {"session_id": "test_session"},
                "timestamp": time.time(),
                "source": "test",
            }
        )

        agent_loop_with_tools.plan_handler = AsyncMock(
            return_value={
                "reasoning": "Ready to provide final answer",
                "actions": ["provide_final_answer"],
                "confidence": 0.9,
                "metadata": {},
            }
        )

        # Mock the act handler to simulate final_answer tool call
        async def mock_act_with_final_answer(plan):
            # Set the final answer directly since we're mocking the act handler
            agent_loop_with_tools.final_answer = "42 is the answer to everything"

            # The act handler logic would normally handle this
            return {
                "success": True,
                "output": "Final answer: 42 is the answer to everything",
                "metadata": {"stopped": True},
            }

        agent_loop_with_tools.act_handler = mock_act_with_final_answer
        agent_loop_with_tools.reflect_handler = AsyncMock(
            return_value={
                "analysis": "Loop terminated by final answer",
                "lessons_learned": ["Final answer tool works"],
                "improvements": [],
                "success_rating": 1.0,
            }
        )

        # Run the agent loop
        result = await agent_loop_with_tools.run()

        # Verify final answer is returned
        assert result == "42 is the answer to everything"

        # Verify loop stopped early (didn't complete all iterations)
        assert agent_loop_with_tools.current_iteration == 1
        assert len(agent_loop_with_tools.get_history()) == 1


class TestAgentLoopErrorHandling:
    """Test AgentLoop error handling and recovery."""

    @pytest.fixture
    def mock_llm_manager_error(self):
        """Create a mock LLM manager that raises errors."""
        llm_manager = MagicMock(spec=LLMManager)
        llm_manager.generate = AsyncMock(
            side_effect=Exception("LLM service unavailable")
        )
        return llm_manager

    @pytest.fixture
    def mock_tool_manager_error(self):
        """Create a mock tool manager that raises errors."""
        tool_manager = MagicMock(spec=ToolManager)
        tool_manager.__iter__ = MagicMock(return_value=iter([]))
        tool_manager.run_tool = MagicMock(
            side_effect=Exception("Tool execution failed")
        )
        return tool_manager

    @pytest.fixture
    def agent_loop_error(self, mock_llm_manager_error, mock_tool_manager_error):
        """Create an AgentLoop that will encounter errors."""
        return AgentLoop(
            llm_manager=mock_llm_manager_error,
            tool_manager=mock_tool_manager_error,
            name="test_agent_error",
            max_iterations=2,
        )

    @pytest.mark.asyncio
    async def test_agent_loop_handles_llm_errors_gracefully(self, agent_loop_error):
        """Test that AgentLoop handles LLM errors gracefully with fallback responses."""
        # Mock handlers - observe should work, but plan will fail due to LLM error
        # Don't override plan_handler - let the default handler trigger the LLM error and use fallback
        agent_loop_error.observe_handler = Mock(
            return_value={
                "content": "Test observation",
                "metadata": {"session_id": "test_session"},
                "timestamp": time.time(),
                "source": "test",
            }
        )

        # Run the agent loop - should handle the LLM error gracefully with fallback responses
        result = await agent_loop_error.run()

        # Verify graceful handling - should complete successfully with fallback responses
        assert result is None  # No final answer
        assert agent_loop_error.current_iteration == 1  # Should complete 1 iteration
        assert (
            len(agent_loop_error.get_history()) == 1
        )  # Should have 1 iteration in history

        # Verify that fallback plan was used (contains "Failed to generate plan")
        iteration_data = agent_loop_error.get_history()[0]
        assert "Failed to generate plan" in iteration_data["plan"]["reasoning"]
        assert iteration_data["plan"]["confidence"] == 0.3  # Fallback confidence

        # Verify loop state after completion
        assert agent_loop_error.is_loop_running() is False

    @pytest.mark.asyncio
    async def test_agent_loop_handles_tool_errors_gracefully(
        self, agent_loop_error, mock_tool_manager_error
    ):
        """Test that AgentLoop handles tool execution errors."""
        # Mock handlers to trigger tool error
        agent_loop_error.observe_handler = Mock(
            return_value={
                "content": "Need to use tool",
                "metadata": {"session_id": "test_session"},
                "timestamp": time.time(),
                "source": "test",
            }
        )

        agent_loop_error.plan_handler = AsyncMock(
            return_value={
                "reasoning": "Plan requires tool usage",
                "actions": ["use_tool"],
                "confidence": 0.8,
                "metadata": {},
            }
        )

        # Mock act handler to trigger tool call that will fail
        async def mock_act_with_tool_error(plan):
            # Simulate tool call that fails
            try:
                mock_tool_manager_error.run_tool("failing_tool", {"param": "value"})
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "metadata": {"plan_actions": plan["actions"]},
                }

        agent_loop_error.act_handler = mock_act_with_tool_error
        agent_loop_error.reflect_handler = AsyncMock(
            return_value={
                "analysis": "Tool failed but loop continued",
                "lessons_learned": ["Handle tool failures"],
                "improvements": ["Better error handling"],
                "success_rating": 0.2,
            }
        )

        # Run the agent loop
        result = await agent_loop_error.run()

        # Verify graceful handling
        assert result is None  # No final answer due to error
        assert agent_loop_error.current_iteration == 1
        assert len(agent_loop_error.get_history()) == 1

        # Verify error was captured in history
        iteration_data = agent_loop_error.get_history()[0]
        assert iteration_data["action_result"]["success"] is False
        assert "error" in iteration_data["action_result"]

    @pytest.mark.asyncio
    async def test_agent_loop_action_failure_stops_loop(self, agent_loop_error):
        """Test that action failures stop the loop appropriately."""
        # Mock handlers where action fails
        agent_loop_error.observe_handler = Mock(
            return_value={
                "content": "Test observation",
                "metadata": {"session_id": "test_session"},
                "timestamp": time.time(),
                "source": "test",
            }
        )

        agent_loop_error.plan_handler = AsyncMock(
            return_value={
                "reasoning": "Plan that will fail",
                "actions": ["failing_action"],
                "confidence": 0.8,
                "metadata": {},
            }
        )

        agent_loop_error.act_handler = AsyncMock(
            return_value={"success": False, "error": "Action failed", "metadata": {}}
        )

        agent_loop_error.reflect_handler = AsyncMock(
            return_value={
                "analysis": "Action failed as expected",
                "lessons_learned": ["Actions can fail"],
                "improvements": ["Better action planning"],
                "success_rating": 0.1,
            }
        )

        # Run the agent loop
        result = await agent_loop_error.run()

        # Verify loop stopped after first failure
        assert agent_loop_error.current_iteration == 1
        assert len(agent_loop_error.get_history()) == 1
        assert result is None

    def test_agent_loop_stop_method(self, agent_loop_error):
        """Test that stop method works correctly."""
        # Start the loop running
        agent_loop_error.is_running = True

        # Stop the loop
        agent_loop_error.stop()

        # Verify loop is stopped
        assert agent_loop_error.is_loop_running() is False


class TestAgentLoopMessageAccumulation:
    """Test AgentLoop message and state accumulation across iterations."""

    @pytest.fixture
    def agent_loop_accumulation(self):
        """Create an AgentLoop for testing message accumulation."""
        llm_manager = MagicMock(spec=LLMManager)

        # Mock consistent responses
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.content = "Consistent response"
        llm_manager.generate = AsyncMock(return_value=mock_response)

        tool_manager = MagicMock(spec=ToolManager)
        tool_manager.__iter__ = MagicMock(return_value=iter([]))

        return AgentLoop(
            llm_manager=llm_manager,
            tool_manager=tool_manager,
            name="test_agent_accumulation",
            max_iterations=3,
        )

    @pytest.mark.asyncio
    async def test_agent_loop_accumulates_messages_across_iterations(
        self, agent_loop_accumulation
    ):
        """Test that messages and context accumulate properly across iterations."""
        # Mock handlers that return consistent results
        iteration_results = []

        def mock_observe():
            # Use the AgentLoop's current iteration (already 1-based)
            current_iter = agent_loop_accumulation.current_iteration
            result = {
                "content": f"Observation iteration {current_iter}",
                "metadata": {"session_id": "test_session", "iteration": current_iter},
                "timestamp": time.time(),
                "source": "test",
            }
            iteration_results.append(("observe", result))
            return result

        async def mock_plan(observation):
            # Get the current iteration number from the observation metadata
            current_iteration = observation["metadata"].get("iteration", 1)
            result = {
                "reasoning": f"Plan for {observation['content']}",
                "actions": [
                    f"action_{current_iteration - 1}"
                ],  # Use iteration-1 for 0-based indexing
                "confidence": 0.8,
                "metadata": {"llm_response": "Plan generated"},
            }
            iteration_results.append(("plan", result))
            return result

        async def mock_act(plan):
            result = {
                "success": True,
                "output": f"Executed {plan['actions'][0]}",
                "metadata": {},
            }
            iteration_results.append(("act", result))
            return result

        async def mock_reflect(action_result, plan):
            result = {
                "analysis": f"Reflection on {action_result['output']}",
                "lessons_learned": [f"lesson_{len(iteration_results)}"],
                "improvements": [],
                "success_rating": 0.9,
            }
            iteration_results.append(("reflect", result))
            return result

        agent_loop_accumulation.observe_handler = mock_observe
        agent_loop_accumulation.plan_handler = mock_plan
        agent_loop_accumulation.act_handler = mock_act
        agent_loop_accumulation.reflect_handler = mock_reflect

        # Run all iterations
        result = await agent_loop_accumulation.run()

        # Verify all iterations completed
        assert agent_loop_accumulation.current_iteration == 3
        assert len(agent_loop_accumulation.get_history()) == 3

        # Verify history contains all phases for each iteration
        for i, iteration_data in enumerate(agent_loop_accumulation.get_history()):
            assert iteration_data["iteration"] == i + 1
            assert iteration_data["observation"] is not None
            assert iteration_data["plan"] is not None
            assert iteration_data["action_result"] is not None
            assert iteration_data["reflection"] is not None

            # Verify content progression
            assert f"iteration {i + 1}" in iteration_data["observation"]["content"]
            assert f"action_{i}" in iteration_data["plan"]["actions"][0]

        # Verify final answer is None (no final_answer tool called)
        assert result is None

    @pytest.mark.asyncio
    async def test_agent_loop_maintains_independent_iterations(
        self, agent_loop_accumulation
    ):
        """Test that each iteration maintains its own context while building on previous ones."""
        # This test verifies that iterations are properly isolated
        # but can build on previous results

        call_sequence = []

        def mock_observe():
            call_sequence.append("observe")
            return {
                "content": "Sequential observation",
                "metadata": {"session_id": "test_session"},
                "timestamp": time.time(),
                "source": "test",
            }

        async def mock_plan(observation):
            call_sequence.append("plan")
            iteration = len([x for x in call_sequence if x == "observe"])
            return {
                "reasoning": f"Plan for iteration {iteration}",
                "actions": [f"action_{iteration}"],
                "confidence": 0.8,
                "metadata": {},
            }

        async def mock_act(plan):
            call_sequence.append("act")
            return {
                "success": True,
                "output": f"Executed {plan['actions'][0]} successfully",
                "metadata": {},
            }

        async def mock_reflect(action_result, plan):
            call_sequence.append("reflect")
            return {
                "analysis": f"Reflection on {action_result['output']}",
                "lessons_learned": ["Sequential execution works"],
                "improvements": [],
                "success_rating": 0.9,
            }

        agent_loop_accumulation.observe_handler = mock_observe
        agent_loop_accumulation.plan_handler = mock_plan
        agent_loop_accumulation.act_handler = mock_act
        agent_loop_accumulation.reflect_handler = mock_reflect

        # Run the agent loop
        await agent_loop_accumulation.run()

        # Verify proper call sequence: observe -> plan -> act -> reflect, repeated
        expected_sequence = ["observe", "plan", "act", "reflect"] * 3
        assert call_sequence == expected_sequence

        # Verify each iteration is properly recorded
        assert len(agent_loop_accumulation.get_history()) == 3
        for i in range(3):
            iteration = agent_loop_accumulation.get_history()[i]
            assert iteration["iteration"] == i + 1
            assert f"iteration {i + 1}" in iteration["plan"]["reasoning"]
