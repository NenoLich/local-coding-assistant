from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.tools.tool_manager import ToolManager
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.config.schemas import RuntimeConfig


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
        assert hasattr(runtime, "config")
        assert isinstance(runtime.config, RuntimeConfig)
        assert hasattr(runtime.config, "persistent_sessions")
        assert hasattr(runtime.config, "max_session_history")


def test_bootstrap_llm_manager_has_config(ctx: AppContext):
    """Test that LLMManager has proper configuration."""
    llm = ctx.get("llm")
    # LLM may be None if dependencies aren't available
    if llm is not None:
        assert hasattr(llm, "config")
        # LLMConfig should be accessible via the config attribute
        assert hasattr(llm.config, "model_name")
        assert hasattr(llm.config, "provider")
        assert hasattr(llm.config, "temperature")
