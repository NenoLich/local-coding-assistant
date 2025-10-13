"""Application bootstrap sequence and dependency injection."""

from pathlib import Path

from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.config.env_loader import EnvLoader
from local_coding_assistant.config.loader import load_config
from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.tools.builtin import SumTool
from local_coding_assistant.tools.tool_manager import ToolManager
from local_coding_assistant.utils.logging import get_logger, setup_logging

logger = get_logger("core.bootstrap")


# Load .env files before anything else
_env_loader = EnvLoader()
try:
    _env_loader.load_env_files()
except Exception as e:
    print(f"Warning: Failed to load .env files: {e}")


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
    # Load configuration (environment variables already loaded above)
    config_paths = [Path(config_path)] if config_path else None
    config = load_config(config_paths)

    # Configure global logging based on config
    if config.runtime.enable_logging:
        # Handle None log_level by defaulting to INFO
        log_level_str = config.runtime.log_level or "INFO"
        setup_logging(level=getattr(__import__("logging"), log_level_str.upper(), 10))
    else:
        setup_logging(level=50)  # CRITICAL level to effectively disable

    ctx = AppContext()

    # Construct core services
    llm_manager = None
    try:
        llm_manager = LLMManager(config.llm)
        logger.info("LLM manager initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize LLM manager: {e}")
        logger.info("Continuing without LLM functionality")
        # For now, we'll skip LLM initialization if it fails
        # In a production system, you might want to use a mock or fallback
        llm_manager = None

    tool_manager = ToolManager()
    # Register builtin tools
    tool_manager.register_tool(SumTool())

    # Only create runtime manager if we have LLM
    if llm_manager is not None:
        runtime = RuntimeManager(
            llm_manager=llm_manager, tool_manager=tool_manager, config=config.runtime
        )
    else:
        logger.warning("Skipping runtime manager creation due to missing LLM")
        runtime = None

    # Register in global context
    ctx.register("llm", llm_manager)
    ctx.register("tools", tool_manager)
    ctx.register("runtime", runtime)

    return ctx
