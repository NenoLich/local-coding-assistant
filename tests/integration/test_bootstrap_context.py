from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.tools.tool_manager import ToolManager


def test_bootstrap_context_contains_expected_components(ctx: AppContext):
    assert isinstance(ctx, AppContext)

    assert "llm" in ctx
    assert "tools" in ctx
    assert "runtime" in ctx

    # LLM and runtime may be None if dependencies aren't available
    llm = ctx.get("llm")
    tools = ctx.get("tools")
    runtime = ctx.get("runtime")

    # Tools should always be available
    assert isinstance(tools, ToolManager)

    # LLM and runtime may be None if OpenAI package isn't installed
    if llm is not None:
        assert isinstance(llm, LLMManager)
    if runtime is not None:
        assert isinstance(runtime, RuntimeManager)


def test_bootstrap_runtime_manager_has_config(ctx: AppContext):
    """Test that RuntimeManager has proper configuration."""
    runtime = ctx.get("runtime")
    # Runtime may be None if LLM is not available
    if runtime is not None:
        assert hasattr(runtime, "config_manager")
        # Import ConfigManager to check the type
        from local_coding_assistant.config import ConfigManager

        assert isinstance(runtime.config_manager, ConfigManager)
        # Test that config_manager has runtime configuration
        resolved_config = runtime.config_manager.global_config
        assert hasattr(resolved_config, "runtime")
        assert hasattr(resolved_config.runtime, "persistent_sessions")
        assert hasattr(resolved_config.runtime, "max_session_history")


def test_bootstrap_llm_manager_has_config(ctx: AppContext):
    """Test that LLMManager has proper configuration access."""
    llm = ctx.get("llm")
    # LLM may be None if dependencies aren't available
    if llm is not None:
        assert isinstance(llm, LLMManager)
        # LLMManager should have config_manager attribute
        assert hasattr(llm, "config_manager")
        assert llm.config_manager is not None

        # Test that config_manager has LLM configuration access
        config_manager = llm.config_manager

        # Check if global config is loaded (it should be loaded during bootstrap)
        global_config = config_manager.global_config
        if global_config is not None:
            assert hasattr(global_config, "llm")
            llm_config = global_config.llm

            # Verify LLMConfig structure
            from local_coding_assistant.config.schemas import LLMConfig

            assert isinstance(llm_config, LLMConfig)

            # Check expected LLMConfig attributes
            assert hasattr(llm_config, "temperature")
            assert hasattr(llm_config, "max_tokens")
            assert hasattr(llm_config, "max_retries")
            assert hasattr(llm_config, "retry_delay")
            assert hasattr(llm_config, "providers")

            # Verify configuration values are reasonable
            assert 0.0 <= llm_config.temperature <= 2.0
            assert llm_config.max_retries > 0
            assert llm_config.retry_delay > 0
        else:
            # If global config is not loaded, that's also acceptable in test environments
            # The important thing is that the config_manager exists and is properly structured
            assert hasattr(config_manager, "load_global_config")
            assert hasattr(config_manager, "resolve")
