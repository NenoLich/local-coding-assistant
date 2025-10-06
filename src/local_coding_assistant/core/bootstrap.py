"""Application bootstrap sequence and dependency injection."""

from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.tools.tool_registry import ToolRegistry


def bootstrap(config_path: str | None = None) -> AppContext:
    """Initialize and configure the application.

    Args:
        config_path: Optional path to a configuration file

    Returns:
        initialized application context
    """
    ctx = AppContext()

    # Register in global context
    ctx.register("llm", LLMManager())
    ctx.register("tools", ToolRegistry())
    ctx.register("runtime", RuntimeManager())

    return ctx
