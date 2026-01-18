"""Unit tests for LangGraphAgent functionality."""

from unittest.mock import MagicMock

import pytest

from local_coding_assistant.agent.langgraph_agent import AgentState, LangGraphAgent
from local_coding_assistant.agent.llm_manager import (
    LLMManager,
)
from local_coding_assistant.core.exceptions import AgentError
from local_coding_assistant.tools.tool_manager import ToolManager


class TestAgentState:
    """Test AgentState functionality."""

    def test_agent_state_initialization(self):
        """Test AgentState initializes with correct default values."""
        state = AgentState()

        assert state.iteration == 0
        assert state.max_iterations == 10
        assert state.final_answer is None
        assert state.history == []
        assert state.session_id.startswith("langgraph_agent_")
        assert state.error is None
        assert state.current_phase == "observe"
        assert state.current_observation is None
        assert state.current_plan is None
        assert state.current_action is None
        assert state.current_reflection is None
        assert state.user_input is None

    def test_agent_state_should_continue(self):
        """Test AgentState should_continue logic."""
        state = AgentState()

        # Should continue by default
        assert state.should_continue() is True

        # Should stop if final answer is set
        state.final_answer = "test answer"
        assert state.should_continue() is False

        # Reset and test max iterations
        state = AgentState()
        state.iteration = 10
        state.max_iterations = 10
        assert state.should_continue() is False

        # Reset and test error with should_stop
        state = AgentState()
        state.error = {"should_stop": True}
        assert state.should_continue() is False

        # Should continue if error doesn't have should_stop
        state.error = {"should_stop": False}
        assert state.should_continue() is True

    def test_agent_state_iteration_management(self):
        """Test AgentState iteration management methods."""
        state = AgentState()

        # Test increment_iteration
        state.increment_iteration()
        assert state.iteration == 1

        state.increment_iteration()
        assert state.iteration == 2

        # Test set_final_answer
        state.set_final_answer("test answer")
        assert state.final_answer == "test answer"

        # Test add_to_history
        test_data = {"test": "data"}
        state.add_to_history("test_phase", test_data)

        assert len(state.history) == 1
        history_entry = state.history[0]
        assert history_entry["phase"] == "test_phase"
        assert history_entry["iteration"] == 2
        assert history_entry["data"] == test_data
        assert "timestamp" in history_entry


class TestLangGraphAgent:
    """Test LangGraphAgent functionality."""

    def test_langgraph_agent_initialization(self):
        """Test LangGraphAgent initializes correctly."""
        llm_manager = MagicMock(spec=LLMManager)
        tool_manager = MagicMock(spec=ToolManager)

        agent = LangGraphAgent(
            llm_manager=llm_manager,
            tool_manager=tool_manager,
            name="test_agent",
            max_iterations=5,
            streaming=True,
        )

        assert agent.llm_manager == llm_manager
        assert agent.tool_manager == tool_manager
        assert agent.name == "test_agent"
        assert agent.max_iterations == 5
        assert agent.streaming is True
        assert hasattr(agent, "graph")

    def test_langgraph_agent_invalid_max_iterations(self):
        """Test LangGraphAgent raises error for invalid max_iterations."""
        llm_manager = MagicMock(spec=LLMManager)
        tool_manager = MagicMock(spec=ToolManager)

        with pytest.raises(AgentError, match="max_iterations must be at least 1"):
            LangGraphAgent(
                llm_manager=llm_manager,
                tool_manager=tool_manager,
                max_iterations=0,
            )

    def test_langgraph_agent_get_tools_methods(self):
        """Test LangGraphAgent tool-related methods."""
        # Create a mock LLM manager
        mock_llm_manager = MagicMock()

        # Create mock tools
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "First tool"
        mock_tool1.available = True
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Second tool"
        mock_tool2.available = True

        # Create a mock tool manager
        mock_tool_manager = MagicMock()
        # Configure the mock tool manager's list_tools method
        mock_tool_manager.list_tools.return_value = [mock_tool1, mock_tool2]

        # Create the LangGraphAgent with the mock managers
        agent = LangGraphAgent(
            llm_manager=mock_llm_manager,
            tool_manager=mock_tool_manager,
            name="test_agent",
        )

        # Test _get_available_tools
        tools = agent._get_available_tools()
        # Verify list_tools was called with available_only=True
        mock_tool_manager.list_tools.assert_called_with(available_only=True, execution_mode='classic')
        assert len(tools) == 2
        assert tools[0]["name"] == "tool1"
        assert tools[1]["name"] == "tool2"

        # Reset the mock call count
        mock_tool_manager.list_tools.reset_mock()

        # Test _get_tools_description
        description = agent._get_tools_description()
        # Verify list_tools was called with available_only=True
        mock_tool_manager.list_tools.assert_called_once_with(available_only=True, execution_mode='classic')
        assert "tool1: First tool" in description
        assert "tool2: Second tool" in description

        # Test with no tool manager
        agent_no_tools = LangGraphAgent(
            llm_manager=mock_llm_manager, tool_manager=None, name="test_agent_no_tools"
        )
        assert agent_no_tools._get_available_tools() == []
        assert agent_no_tools._get_tools_description() == "No tools available"

    def test_langgraph_agent_build_graph(self):
        """Test LangGraphAgent builds graph correctly."""
        llm_manager = MagicMock(spec=LLMManager)
        tool_manager = MagicMock(spec=ToolManager)

        agent = LangGraphAgent(
            llm_manager=llm_manager,
            tool_manager=tool_manager,
        )

        # Check that graph is built
        assert hasattr(agent, "graph")
        assert agent.graph is not None

        # Check that all expected nodes exist
        # For this test, we verify the graph object exists and is callable

    @pytest.mark.asyncio
    async def test_langgraph_agent_run_success(self):
        """Test LangGraphAgent run method with successful execution."""
        pytest.skip(
            "This test requires full LangGraph implementation which is complex to mock"
        )

    @pytest.mark.asyncio
    async def test_langgraph_agent_run_llm_error(self):
        """Test LangGraphAgent handles LLM errors gracefully."""
        pytest.skip(
            "This test requires full LangGraph implementation which is complex to mock"
        )

    @pytest.mark.asyncio
    async def test_langgraph_agent_run_streaming(self):
        """Test LangGraphAgent with streaming enabled."""
        pytest.skip(
            "This test requires full LangGraph implementation which is complex to mock"
        )

    @pytest.mark.asyncio
    async def test_langgraph_agent_max_iterations_reached(self):
        """Test LangGraphAgent stops when max iterations reached."""
        pytest.skip(
            "This test requires full LangGraph implementation which is complex to mock"
        )

    def test_langgraph_agent_state_management(self):
        """Test LangGraphAgent state management methods."""
        llm_manager = MagicMock(spec=LLMManager)
        tool_manager = MagicMock(spec=ToolManager)

        agent = LangGraphAgent(
            llm_manager=llm_manager,
            tool_manager=tool_manager,
        )

        # Test get_current_state (should return empty state initially)
        state = agent.get_current_state()
        assert isinstance(state, AgentState)

        # Test get_history (should return empty list initially)
        history = agent.get_history()
        assert history == []
