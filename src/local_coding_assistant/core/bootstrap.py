"""Application bootstrap sequence and dependency injection."""

from pathlib import Path
from typing import Any

from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.config import get_config_manager, load_config
from local_coding_assistant.config.env_loader import EnvLoader
from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.providers.provider_manager import provider_manager
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.tools.builtin import SumTool
from local_coding_assistant.tools.tool_manager import ToolManager
from local_coding_assistant.utils.logging import get_logger, setup_logging

logger = get_logger("core.bootstrap")


def bootstrap(
    config_path: str | None = None, *, log_level: int | None = None
) -> AppContext:
    """Initialize and configure the application."""
    # Load .env files before anything else
    _env_loader = EnvLoader()
    try:
        _env_loader.load_env_files()
    except Exception as e:
        print(f"Warning: Failed to load .env files: {e}")

    ctx = AppContext()

    # Load configuration
    config = _load_config(config_path)

    # Configure logging
    _setup_logging(config, log_level)

    # Initialize LLM manager
    llm_manager = _initialize_llm_manager(config)

    # Initialize tool manager
    tool_manager = _initialize_tool_manager()

    # Initialize runtime manager if both LLM and tools are available
    runtime = _initialize_runtime_manager(llm_manager, tool_manager, config)

    # Register components in the application context
    ctx.register("llm", llm_manager)
    ctx.register("tools", tool_manager)
    ctx.register("runtime", runtime)

    return ctx


# --- Helper Functions ---


def _load_config(config_path: str | None) -> Any:
    """Load application configuration."""
    config_paths = [Path(config_path)] if config_path else None
    try:
        return load_config(config_paths)
    except Exception as err:
        logger.warning(f"Failed to load configuration: {err}")
        logger.info("Continuing with default configuration")
        return None


def _setup_logging(config: Any, log_level: int | None) -> None:
    """Configure application logging."""
    if (
        config is not None
        and hasattr(config, "runtime")
        and config.runtime.enable_logging
    ):
        # Use provided log_level parameter if given, otherwise use config
        if log_level is not None:
            setup_logging(level=log_level)
        else:
            # Handle None log_level by defaulting to INFO
            log_level_str = config.runtime.log_level or "INFO"
            # Ensure log_level_str is a string (handles test mocks)
            if hasattr(log_level_str, "__str__"):
                log_level_str = str(log_level_str)
            setup_logging(
                level=getattr(__import__("logging"), log_level_str.upper(), 10)
            )
    else:
        # Config failed to load or logging explicitly disabled - use CRITICAL level to disable
        setup_logging(level=50)  # CRITICAL level to effectively disable


def _initialize_llm_manager(config: Any) -> LLMManager | None:
    """Initialize the LLM manager."""
    try:
        if config is not None:
            llm_manager = LLMManager(config.llm, provider_manager)
            logger.info("LLM manager initialized successfully")
            return llm_manager
        else:
            logger.info(
                "Skipping LLM manager initialization due to missing configuration"
            )
            return None
    except Exception as err:
        logger.warning(f"Failed to initialize LLM manager: {err}")
        logger.info("Continuing without LLM functionality")
        return None


def _initialize_tool_manager() -> ToolManager | None:
    """Initialize the tool manager."""
    try:
        tool_manager = ToolManager()
        tool_manager.register_tool(SumTool())
        logger.info("Tool manager initialized successfully")
        return tool_manager
    except Exception as err:
        logger.warning(f"Failed to initialize tool manager: {err}")
        logger.info("Continuing without tool functionality")
        return None


def _initialize_runtime_manager(
    llm_manager: LLMManager | None, tool_manager: ToolManager | None, _config: Any
) -> RuntimeManager | None:
    """Initialize the runtime manager."""
    if llm_manager is not None and tool_manager is not None:
        config_manager = get_config_manager()
        if config_manager.global_config is None:
            config_manager.load_global_config()
        return RuntimeManager(
            llm_manager=llm_manager,
            tool_manager=tool_manager,
            config_manager=config_manager,
        )
    else:
        if llm_manager is None:
            logger.warning("Skipping runtime manager creation due to missing LLM")
        if tool_manager is None:
            logger.warning("Skipping runtime manager creation due to missing tools")
        return None
