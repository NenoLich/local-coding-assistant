"""Application bootstrap sequence and dependency injection."""

from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.tools.builtin import SumTool
from local_coding_assistant.tools.tool_registry import ToolRegistry
from local_coding_assistant.utils.logging import setup_logging


def bootstrap(
    config_path: str | None = None, *, log_level: int | None = None
) -> AppContext:
    """Initialize and configure the application.

    Args:
        config_path: Optional path to a configuration file
        log_level: Optional logging level (e.g., logging.INFO)

    Returns:
        initialized application context
    """
    # Configure global logging early
    if log_level is not None:
        setup_logging(level=log_level)
    else:
        setup_logging()

    ctx = AppContext()

    # Construct core services
    llm = LLMManager()
    tools = ToolRegistry()
    # Register builtin tools
    tools.register(SumTool())
    runtime = RuntimeManager(llm=llm, tools=tools)

    # Register in global context
    ctx.register("llm", llm)
    ctx.register("tools", tools)
    ctx.register("runtime", runtime)

    return ctx
