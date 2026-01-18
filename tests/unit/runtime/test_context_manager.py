import pytest
from unittest.mock import MagicMock, patch, ANY

from local_coding_assistant.runtime.context_manager import (
    ContextManager,
    ToolSelector,
    MemoryProvider,
    SkillProvider,
    ExecutionMode,
    ToolSpec,
)
from local_coding_assistant.tools.types import ToolExecutionMode
from local_coding_assistant.runtime.runtime_types import PromptContext, AgentProfile

# Import shared test fixtures and mocks
from .conftest import (
    MockTool,
    MockToolManager,
    config_manager,
    tool_manager,
    session_state,
)


class TestToolSelector:
    def test_select_reasoning_mode(self):
        selector = ToolSelector()
        assert selector.select(tool_call_mode="reasoning_only") == []

    def test_select_with_tools(self, tool_manager):
        tool = MockTool("test_tool", "Test tool")
        tool.parameters = {"type": "object", "properties": {}, "required": []}
        tool_manager.tools = [tool]

        selector = ToolSelector(tool_manager=tool_manager)
        tools = selector.select(tool_call_mode="classic")

        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "test_tool"
        assert tools[0]["function"]["description"] == "Test tool"
        assert "parameters" in tools[0]["function"]

    def test_select_ptc_mode(self, tool_manager):
        # Create a tool with PTC execution mode
        tool = MockTool("test_ptc_tool", "Test PTC tool")
        tool.parameters = {"type": "object", "properties": {}, "required": []}
        tool.execution_mode = ToolExecutionMode.PTC
        tool.available = True  # Ensure the tool is marked as available

        # Set up the tool manager to return our tool
        tool_manager.tools = [tool]

        # Create the selector and get tools for PTC mode
        selector = ToolSelector(tool_manager=tool_manager)
        tools = selector.select(tool_call_mode="ptc")

        # Verify we got the expected tool
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "test_ptc_tool"
        assert tools[0]["function"]["description"] == "Test PTC tool"

    def test_resolve_tool_entry_tuple(self):
        tool = MockTool("test_tool", "Test tool")
        selector = ToolSelector()

        # Test with (name, tool) tuple
        result = selector._resolve_tool_entry(("custom_name", tool))
        assert result.name == "custom_name"
        assert result.description == tool.description

        # Test with just the tool object
        result = selector._resolve_tool_entry(tool)
        assert result.name == "test_tool"
        assert result.description == "Test tool"

    def test_resolve_tool_entry_invalid(self):
        selector = ToolSelector()

        # Test with None
        with pytest.raises(ValueError, match="Tool entry cannot be None"):
            selector._resolve_tool_entry(None)

        # Test with empty name in tuple
        tool = MockTool("test", "Test")
        with pytest.raises(
            ValueError, match="Tool name in tuple must be a non-empty string"
        ):
            selector._resolve_tool_entry(("", tool))

        # Test with missing description
        class BadTool:
            name = "bad_tool"

        with pytest.raises(ValueError, match="missing required 'description'"):
            selector._resolve_tool_entry(BadTool())

    def test_build_tool_spec(self):
        selector = ToolSelector()
        tool_spec = ToolSpec(
            name="test_tool",
            description="Test tool description",
            parameters={"type": "object", "properties": {"param": {"type": "string"}}},
        )

        result = selector._build_tool_spec(tool_spec)
        assert result == {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "Test tool description",
                "parameters": {
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                },
            },
        }


class TestContextManager:
    def test_init_with_defaults(self, config_manager):
        manager = ContextManager(config_manager)
        assert isinstance(manager.memory_provider, MemoryProvider)
        assert isinstance(manager.skill_provider, SkillProvider)
        assert isinstance(manager.tool_selector, ToolSelector)

    def test_build_context_reasoning_mode(self, config_manager, session_state):
        manager = ContextManager(config_manager)
        context = manager.build_context(
            session=session_state,
            user_input="test input",
            tool_call_mode="reasoning_only",
        )

        assert isinstance(context, PromptContext)
        assert context.execution_mode == ExecutionMode.REASONING_ONLY
        assert context.user_input == "test input"
        assert context.tool_call_mode == "reasoning_only"
        # We expect a default agent profile even in reasoning_only mode
        assert len(context.agent_profiles) == 1
        assert context.agent_profiles[0].name == "default"
        assert not context.tools

    def test_build_context_validation(self, config_manager, session_state):
        manager = ContextManager(config_manager)

        # Test invalid session
        with pytest.raises(
            ValueError, match="session must be an instance of SessionState"
        ):
            manager.build_context(
                session=None, user_input="test", tool_call_mode="reasoning_only"
            )

        # Test invalid user_input
        with pytest.raises(TypeError, match="user_input must be a string"):
            manager.build_context(
                session=session_state, user_input=123, tool_call_mode="reasoning_only"
            )

        # Test invalid tool_call_mode type
        with pytest.raises(TypeError, match="tool_call_mode must be a string"):
            manager.build_context(
                session=session_state, user_input="test", tool_call_mode=123
            )

        # Test invalid tool_call_mode value
        with pytest.raises(ValueError, match="Invalid tool_call_mode: 'invalid_mode'"):
            manager.build_context(
                session=session_state, user_input="test", tool_call_mode="invalid_mode"
            )

        # Test case insensitivity
        context = manager.build_context(
            session=session_state,
            user_input="test",
            tool_call_mode="REASONING_ONLY",  # Uppercase should work
        )
        assert context.tool_call_mode == "reasoning_only"

        # Test boolean parameter validation
        with pytest.raises(TypeError, match="agent_mode must be a boolean"):
            manager.build_context(
                session=session_state,
                user_input="test",
                tool_call_mode="reasoning_only",
                agent_mode="not_a_boolean",
            )

        with pytest.raises(TypeError, match="graph_mode must be a boolean"):
            manager.build_context(
                session=session_state,
                user_input="test",
                tool_call_mode="reasoning_only",
                graph_mode="not_a_boolean",
            )

    def test_build_context_with_agent_mode(
        self, config_manager, session_state, tool_manager
    ):
        manager = ContextManager(config_manager, tool_manager=tool_manager)
        context = manager.build_context(
            session=session_state,
            user_input="test input",
            tool_call_mode="classic",
            agent_mode=True,
        )

        # Check that we have the default agent profile
        assert len(context.agent_profiles) == 1
        assert context.agent_profiles[0].name == "default"

    def test_resolve_execution_mode_reasoning(self, config_manager):
        manager = ContextManager(config_manager)
        mode = manager._resolve_execution_mode("reasoning_only", False)
        assert mode == ExecutionMode.REASONING_ONLY

    def test_resolve_execution_mode_ptc_sandbox_enabled(
        self, config_manager, tool_manager
    ):
        manager = ContextManager(config_manager, tool_manager=tool_manager)
        tool_manager.has_runtime = MagicMock(return_value=True)
        mode = manager._resolve_execution_mode("ptc", True)
        assert mode == ExecutionMode.SANDBOX_PYTHON

    def test_resolve_execution_mode_ptc_sandbox_disabled(
        self, config_manager, tool_manager
    ):
        manager = ContextManager(config_manager, tool_manager=tool_manager)
        mode = manager._resolve_execution_mode("ptc", False)
        assert mode == ExecutionMode.CLASSIC_TOOLS

    def test_resolve_agents_default(self, config_manager):
        manager = ContextManager(config_manager)
        agents = manager._resolve_agents(agent_mode=False, graph_mode=False)
        assert len(agents) == 1
        assert agents[0].name == "default"

    def test_resolve_agents_graph_mode(self, config_manager):
        # Create mock agent profiles
        planner = AgentProfile(name="planner", description="Planner")
        executor = AgentProfile(name="executor", description="Executor")

        # Create a manager with our custom agent catalog
        manager = ContextManager(config_manager, agent_profiles=[planner, executor])

        # Now when we resolve agents in graph mode, it should use our custom catalog
        agents = manager._resolve_agents(agent_mode=False, graph_mode=True)

        # We should get both agents since we're in graph mode
        assert len(agents) == 2
        assert {a.name for a in agents} == {"planner", "executor"}

    def test_build_context_with_memories(self, config_manager, session_state):
        memory_provider = MemoryProvider()
        memory_provider.fetch = MagicMock(return_value=["memory1", "memory2"])

        manager = ContextManager(config_manager, memory_provider=memory_provider)

        context = manager.build_context(
            session=session_state, user_input="test", tool_call_mode="reasoning_only"
        )

        assert context.memories == ["memory1", "memory2"]
        memory_provider.fetch.assert_called_once_with(session=session_state)

    def test_build_context_with_skills(self, config_manager, session_state):
        skill_provider = SkillProvider()
        skill_provider.resolve = MagicMock(return_value=["skill1", "skill2"])

        manager = ContextManager(config_manager, skill_provider=skill_provider)

        context = manager.build_context(
            session=session_state, user_input="test", tool_call_mode="reasoning_only"
        )

        assert context.active_skills == ["skill1", "skill2"]
        skill_provider.resolve.assert_called_once_with(
            execution_mode=ExecutionMode.REASONING_ONLY
        )

    @patch("local_coding_assistant.runtime.context_manager.log")
    def test_execution_mode_fallback_warnings(self, mock_log, config_manager):
        # Test fallback when tool manager is None
        manager = ContextManager(config_manager, tool_manager=None)
        mode = manager._resolve_execution_mode("classic", False)
        assert mode == ExecutionMode.REASONING_ONLY
        mock_log.warning.assert_called_with(
            "No tool manager available. Tool-based execution modes require a valid tool manager. "
            "Falling back to REASONING_ONLY mode."
        )

    def test_custom_agent_catalog(self, config_manager):
        custom_agent = AgentProfile(name="custom", description="Custom agent")
        manager = ContextManager(config_manager, agent_profiles=[custom_agent])

        # Should use the provided agent catalog instead of config
        agents = manager._resolve_agents(agent_mode=True, graph_mode=False)
        assert len(agents) == 1
        assert agents[0].name == "custom"
