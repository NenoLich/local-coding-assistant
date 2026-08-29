from unittest.mock import MagicMock

import pytest

from local_coding_assistant.runtime.context_manager import (
    ContextManager,
    ExecutionMode,
    MemoryProvider,
    SkillProvider,
    StaticComponentCache,
    ToolsBundle,
    ToolSelector,
    ToolSpec,
)
from local_coding_assistant.runtime.runtime_types import AgentProfile, PromptContext
from local_coding_assistant.tools.types import ToolExecutionMode

# Import shared test fixtures and mocks
from .conftest import (
    MockTool,
)


class TestToolSelector:
    def test_select_reasoning_mode(self):
        selector = ToolSelector()
        assert selector.select(tool_call_mode="reasoning_only") == ([], [])

    def test_select_with_tools(self, tool_manager):
        tool = MockTool("test_tool", "Test tool")
        tool.parameters = {"type": "object", "properties": {}, "required": []}
        tool_manager.tools = [tool]

        selector = ToolSelector(tool_manager=tool_manager)
        tools, tools_prompt = selector.select(tool_call_mode="classic")

        assert len(tools) == 1
        tool_spec = tools[0]
        openai_format = tool_spec.to_openai_function()
        assert openai_format["function"]["name"] == "test_tool"
        assert openai_format["function"]["description"] == "Test tool"
        assert "parameters" in openai_format["function"]

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
        tools, tools_prompt = selector.select(tool_call_mode="ptc")

        # Verify we got the expected tool
        assert len(tools) == 1
        tool_spec = tools[0]
        openai_format = tool_spec.to_openai_function()
        assert openai_format["function"]["name"] == "test_ptc_tool"
        assert openai_format["function"]["description"] == "Test PTC tool"

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
        tool_spec = ToolSpec(
            name="test_tool",
            description="Test tool description",
            parameters={"type": "object", "properties": {"param": {"type": "string"}}},
        )

        result = tool_spec.to_openai_function()
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
        assert isinstance(manager._cache, StaticComponentCache)

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
        assert context.agent_profile is not None
        assert context.agent_profile.name == "default"
        assert not context.tools

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
        assert context.agent_profile is not None
        assert context.agent_profile.name == "default"

    def test_resolve_execution_mode_reasoning(self, config_manager):
        manager = ContextManager(config_manager)
        mode = manager._map_validated_mode_to_execution_mode("reasoning_only")
        assert mode == ExecutionMode.REASONING_ONLY

    def test_resolve_execution_mode_ptc_sandbox_enabled(
        self, config_manager, tool_manager
    ):
        manager = ContextManager(config_manager, tool_manager=tool_manager)
        mode = manager._map_validated_mode_to_execution_mode("ptc")
        assert mode == ExecutionMode.SANDBOX_PYTHON

    def test_resolve_execution_mode_ptc_sandbox_disabled(
        self, config_manager, tool_manager
    ):
        manager = ContextManager(config_manager, tool_manager=tool_manager)
        mode = manager._map_validated_mode_to_execution_mode("ptc")
        assert mode == ExecutionMode.SANDBOX_PYTHON

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

    def test_custom_agent_catalog(self, config_manager, session_state):
        custom_agent = AgentProfile(name="custom", description="Custom agent")
        manager = ContextManager(config_manager, agent_profiles=[custom_agent])

        # Should use the provided agent catalog instead of config
        context = manager.build_context(
            session=session_state, user_input="test", tool_call_mode="reasoning_only"
        )
        assert context.agent_profile.name == "custom"


class TestStaticComponentCache:
    def test_cache_hit_agent_profile(self, config_manager):
        cache = StaticComponentCache(config_manager)
        profile1 = cache.get_agent_profile(agent_mode=False, graph_mode=False)
        profile2 = cache.get_agent_profile(agent_mode=False, graph_mode=False)
        assert profile1 is profile2  # Same object from cache

    def test_cache_miss_agent_profile_different_params(self, config_manager):
        cache = StaticComponentCache(config_manager)
        profile1 = cache.get_agent_profile(agent_mode=False, graph_mode=False)
        profile2 = cache.get_agent_profile(agent_mode=True, graph_mode=False)
        assert profile1.name == "default"
        assert profile2.name == "default"
        # Different cache keys, but same profile from config

    def test_cache_hit_tools(self, config_manager, tool_manager):
        cache = StaticComponentCache(config_manager, tool_manager=tool_manager)
        tools1, prompt1 = cache.get_tools("reasoning_only", tool_manager)
        tools2, prompt2 = cache.get_tools("reasoning_only", tool_manager)
        assert tools1 is tools2  # Same object from cache
        assert prompt1 is prompt2

    def test_cache_miss_tools_different_mode(self, config_manager, tool_manager):
        cache = StaticComponentCache(config_manager, tool_manager=tool_manager)
        tools1, _ = cache.get_tools("reasoning_only", tool_manager)
        tools2, _ = cache.get_tools("classic", tool_manager)
        # Different modes should return different results
        assert tools1 != tools2

    def test_cache_hit_skills(self, config_manager):
        cache = StaticComponentCache(config_manager)
        skills1 = cache.get_skills("reasoning_only")
        skills2 = cache.get_skills("reasoning_only")
        assert skills1 is skills2  # Same object from cache

    def test_invalidate_all(self, config_manager, tool_manager):
        cache = StaticComponentCache(config_manager, tool_manager=tool_manager)
        # Populate caches
        cache.get_agent_profile(agent_mode=False, graph_mode=False)
        cache.get_tools("reasoning_only", tool_manager)
        cache.get_skills("reasoning_only")

        # Invalidate
        cache.invalidate_all()

        # Caches should be empty
        assert len(cache._agent_profile_cache) == 0
        assert len(cache._tools_cache) == 0
        assert len(cache._skills_cache) == 0


class TestToolsBundle:
    def test_from_tool_manager_reasoning_only(self, tool_manager):
        bundle = ToolsBundle.from_tool_manager(tool_manager, "reasoning_only")
        assert bundle.tool_call_mode == "reasoning_only"
        assert bundle.execution_mode == ToolExecutionMode.CLASSIC

    def test_from_tool_manager_ptc(self, tool_manager):
        bundle = ToolsBundle.from_tool_manager(tool_manager, "ptc")
        assert bundle.tool_call_mode == "ptc"
        assert bundle.execution_mode == ToolExecutionMode.PTC

    def test_from_tool_manager_classic(self, tool_manager):
        bundle = ToolsBundle.from_tool_manager(tool_manager, "classic")
        assert bundle.tool_call_mode == "classic"
        assert bundle.execution_mode == ToolExecutionMode.CLASSIC

    def test_tools_bundle_equality(self, tool_manager):
        bundle1 = ToolsBundle.from_tool_manager(tool_manager, "reasoning_only")
        bundle2 = ToolsBundle.from_tool_manager(tool_manager, "reasoning_only")
        assert bundle1 == bundle2

    def test_tools_bundle_inequality_different_mode(self, tool_manager):
        bundle1 = ToolsBundle.from_tool_manager(tool_manager, "reasoning_only")
        bundle2 = ToolsBundle.from_tool_manager(tool_manager, "classic")
        assert bundle1 != bundle2

    def test_tools_bundle_frozen(self, tool_manager):
        bundle = ToolsBundle.from_tool_manager(tool_manager, "reasoning_only")
        with pytest.raises(Exception):  # FrozenInstanceError
            bundle.tool_call_mode = "classic"


class TestContextManagerCaching:
    def test_cache_hit_repeated_calls(self, config_manager, session_state):
        manager = ContextManager(config_manager)
        context1 = manager.build_context(
            session=session_state, user_input="test", tool_call_mode="reasoning_only"
        )
        context2 = manager.build_context(
            session=session_state, user_input="test2", tool_call_mode="reasoning_only"
        )
        # Agent profile should be cached (same object)
        assert context1.agent_profile is context2.agent_profile

    def test_cache_miss_mode_change(self, config_manager, session_state):
        manager = ContextManager(config_manager)
        context1 = manager.build_context(
            session=session_state, user_input="test", tool_call_mode="reasoning_only"
        )
        context2 = manager.build_context(
            session=session_state, user_input="test", tool_call_mode="classic"
        )
        # Different modes should have different execution modes
        assert context1.execution_mode == ExecutionMode.REASONING_ONLY
        assert context2.execution_mode == ExecutionMode.CLASSIC_TOOLS

    def test_dynamic_components_always_fresh(self, config_manager, session_state):
        manager = ContextManager(config_manager)
        context1 = manager.build_context(
            session=session_state, user_input="test1", tool_call_mode="reasoning_only"
        )
        # Add a message to history
        from local_coding_assistant.runtime.session import Message

        session_state.history.append(Message(role="user", content="new message"))

        context2 = manager.build_context(
            session=session_state, user_input="test2", tool_call_mode="reasoning_only"
        )
        # History should be different (dynamic component)
        assert len(context1.history) != len(context2.history)
        # User input should be different
        assert context1.user_input == "test1"
        assert context2.user_input == "test2"

    def test_invalidate_all_clears_cache(self, config_manager, session_state):
        manager = ContextManager(config_manager)
        # Populate cache
        manager.build_context(
            session=session_state, user_input="test", tool_call_mode="reasoning_only"
        )

        # Invalidate
        manager.invalidate_all()

        # Cache should be cleared
        assert len(manager._cache._agent_profile_cache) == 0
        assert len(manager._cache._tools_cache) == 0
        assert len(manager._cache._skills_cache) == 0
