from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.tools.tool_registry import ToolRegistry
from local_coding_assistant.runtime.runtime_manager import RuntimeManager


def test_bootstrap_context_contains_expected_components(ctx: AppContext):
    assert isinstance(ctx, AppContext)

    assert "llm" in ctx
    assert "tools" in ctx
    assert "runtime" in ctx

    assert isinstance(ctx.get("llm"), LLMManager)
    assert isinstance(ctx.get("tools"), ToolRegistry)
    assert isinstance(ctx.get("runtime"), RuntimeManager)
