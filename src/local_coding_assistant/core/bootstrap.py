"""Application bootstrap sequence and dependency injection."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.config.env_manager import EnvManager
from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.core.dependencies import AppDependencies
from local_coding_assistant.core.protocols import IConfigManager, IToolManager
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.utils.logging import get_logger, setup_logging

# Imported here to avoid circular imports
if TYPE_CHECKING:
    pass

logger = get_logger("core.bootstrap")


def _setup_environment() -> None:
    """Load environment variables and perform environment setup."""
    env_manager = EnvManager()
    try:
        env_manager.load_env_files()
    except Exception as e:
        logger.warning("Failed to load .env files: %s", e, exc_info=True)


def bootstrap(
    config_path: str | None = None,
    *,
    log_level: int | None = None,
    config_manager: IConfigManager | None = None,
) -> AppContext:
    """Initialize and configure the application.

    Args:
        config_path: Optional path to configuration file
        log_level: Override log level
        config_manager: Optional pre-configured config manager (for testing)

    Returns:
        Initialized AppContext with all dependencies

    Raises:
        RuntimeError: If application initialization fails
    """
    # 1. Setup basic console logging first with just the log level
    setup_logging(level=log_level or logging.INFO)
    global logger
    logger = get_logger("core.bootstrap")

    try:
        logger.debug("Starting application bootstrap")

        # 2. Setup environment
        _setup_environment()
        logger.debug("Environment setup completed")

        # 3. Initialize configuration
        config, config_manager = _initialize_config(config_path, config_manager)
        logger.debug("Configuration loaded successfully")

        # 4. Now setup full logging with the loaded config
        _setup_logging(config, log_level)
        logger = get_logger("core.bootstrap")  # Reinitialize logger after setup

        # 4. Initialize dependencies with the config manager
        deps = _initialize_dependencies(config_manager)

        # 5. Create context with dependencies
        ctx = AppContext(dependencies=deps)

        # 6. Initialize components
        deps.llm_manager = _initialize_llm_manager(config_manager, config)
        deps.tool_manager = _initialize_tool_manager(config_manager)

        # 7. Initialize runtime manager with required config_manager
        deps.runtime_manager = _initialize_runtime_manager(
            config_manager=config_manager,
            llm_manager=deps.llm_manager,
            tool_manager=deps.tool_manager,
        )

        # 8. Register components in context
        if deps.llm_manager:
            ctx.register("llm", deps.llm_manager)
        if deps.tool_manager:
            ctx.register("tools", deps.tool_manager)
        if deps.runtime_manager:
            ctx.register("runtime", deps.runtime_manager)

        # 9. Mark dependencies as fully initialized
        deps.mark_initialized()

        logger.info("Application bootstrap completed successfully")
        return ctx

    except Exception as e:
        logger.critical("Failed to bootstrap application", exc_info=True)
        raise RuntimeError("Failed to initialize application") from e


# --- Helper Functions ---


def _initialize_config(
    config_path: str | None, config_manager: IConfigManager | None = None
) -> tuple[Any, IConfigManager]:
    """Initialize configuration and config manager.

    Args:
        config_path: Optional path to configuration file
        config_manager: Optional pre-configured config manager

    Returns:
        Tuple of (config, config_manager)
    """

    from local_coding_assistant.config.config_manager import ConfigManager

    # Use provided config manager or create a new one
    if config_manager is None:
        config_paths = [Path(config_path)] if config_path else []
        config_manager = ConfigManager(config_paths=config_paths)

    # Load config if not already loaded
    if not hasattr(config_manager, "_config"):
        config = config_manager.load_global_config()
    else:
        # Access the config directly from the instance if it's a ConfigManager
        config = getattr(config_manager, "_config", None)

    if config is None:
        raise RuntimeError("Failed to load configuration")

    return config, config_manager


def _initialize_dependencies(config_manager: IConfigManager) -> AppDependencies:
    """Initialize application dependencies.

    Args:
        config_manager: Initialized config manager

    Returns:
        Initialized AppDependencies instance
    """
    return AppDependencies(config_manager=config_manager)


def _setup_logging(config: Any | None = None, log_level: int | None = None) -> None:
    """Configure application logging.

    Args:
        config: Optional loaded configuration. If None, only basic logging is configured.
        log_level: Optional log level override. Takes precedence over config.
    """
    try:
        # If log_level is provided, use it directly
        if log_level is not None:
            setup_logging(level=log_level)
            return

        # If we have a config with logging settings, use those
        if config is not None and hasattr(config, "logging"):
            # Check if logging is explicitly disabled
            if hasattr(config.logging, "enabled") and not config.logging.enabled:
                setup_logging(level=logging.CRITICAL)
                return

            # Use log level from config if available
            if hasattr(config.logging, "level"):
                log_level_str = str(config.logging.level)
                if log_level_str.isdigit():
                    setup_logging(level=int(log_level_str))
                else:
                    # Use explicit mapping for log levels
                    log_level_map = {
                        "DEBUG": logging.DEBUG,
                        "INFO": logging.INFO,
                        "WARNING": logging.WARNING,
                        "ERROR": logging.ERROR,
                        "CRITICAL": logging.CRITICAL,
                    }
                    level = log_level_map.get(log_level_str.upper(), logging.INFO)
                    setup_logging(level=level)
                return

        # Default to INFO level if no specific config found
        setup_logging(level=logging.INFO)

    except Exception as e:
        # If logging setup fails, configure basic logging as fallback
        logging.basicConfig(level=logging.INFO)
        global logger
        logger = logging.getLogger("core.bootstrap")
        logger.warning("Failed to configure logging: %s", str(e), exc_info=True)


def _initialize_llm_manager(
    config_manager: IConfigManager, config: Any
) -> LLMManager | None:
    """Initialize the LLM manager.

    Args:
        config_manager: The config manager implementing IConfigManager
        config: Loaded configuration

    Returns:
        Initialized LLMManager instance or None if initialization fails
    """
    try:
        # Lazy import to avoid circular imports
        from local_coding_assistant.providers.provider_manager import provider_manager

        llm_manager = LLMManager(config_manager, provider_manager)
        logger.info("LLM manager initialized successfully")
        return llm_manager

    except Exception as e:
        logger.error("Failed to initialize LLM manager: %s", str(e), exc_info=True)
        return None


def _initialize_tool_manager(config_manager: IConfigManager) -> IToolManager | None:
    """Initialize the tool manager.

    Args:
        config_manager: The config manager implementing IConfigManager

    Returns:
        Initialized ToolManager instance or None if initialization fails
    """
    try:
        from local_coding_assistant.tools.tool_manager import (
            ToolManager,  # Local import
        )

        tool_manager: IToolManager = ToolManager(config_manager=config_manager)
        return tool_manager
    except Exception as e:
        logger.error("Failed to initialize tool manager: %s", str(e), exc_info=True)
        return None


def _initialize_runtime_manager(
    config_manager: IConfigManager,
    llm_manager: LLMManager | None = None,
    tool_manager: IToolManager | None = None,
) -> RuntimeManager | None:
    """Initialize the runtime manager.

    Args:
        config_manager: The config manager instance (required)
        llm_manager: Optional LLM manager instance
        tool_manager: Optional tool manager instance

    Returns:
        Initialized RuntimeManager instance or None if initialization fails
    """
    try:
        if not llm_manager:
            logger.warning(
                "LLM manager not available, skipping runtime manager initialization"
            )
            return None

        runtime_manager = RuntimeManager(
            config_manager=config_manager,
            llm_manager=llm_manager,
            tool_manager=tool_manager,
        )

        logger.info("Runtime manager initialized successfully")
        return runtime_manager

    except Exception as e:
        logger.error("Failed to initialize runtime manager: %s", str(e), exc_info=True)
        return None
