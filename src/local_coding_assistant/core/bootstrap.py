"""Application bootstrap sequence and dependency injection."""

import logging
from pathlib import Path
from typing import Any

from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.config.env_manager import EnvManager, get_env_manager
from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.core.dependencies import AppDependencies
from local_coding_assistant.core.protocols import IConfigManager, IToolManager
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.sandbox.manager import SandboxManager
from local_coding_assistant.utils.logging import get_logger, setup_logging

logger = get_logger("core.bootstrap")


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

        # 2. Get environment manager (automatically loads .env files on first access)
        env_manager = get_env_manager()
        logger.debug("Environment setup completed")

        # 3. Initialize configuration
        config_manager = _initialize_config(config_path, config_manager, env_manager)
        logger.debug("Configuration loaded successfully")

        # 4. Now setup full logging with the loaded config
        _setup_logging(config_manager, log_level, logging_to_file=True)
        logger = get_logger("core.bootstrap")  # Reinitialize logger after setup

        # 4. Initialize dependencies with the config manager
        deps = _initialize_dependencies(config_manager)

        # 5. Create context with dependencies
        ctx = AppContext(dependencies=deps)

        # 6. Initialize components
        deps.config_manager = config_manager
        deps.sandbox_manager = _initialize_sandbox_manager(config_manager)
        deps.llm_manager = _initialize_llm_manager(config_manager)
        deps.tool_manager = _initialize_tool_manager(
            config_manager=config_manager,
            sandbox_manager=deps.sandbox_manager,
        )
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
        if deps.sandbox_manager:
            ctx.register("sandbox", deps.sandbox_manager)
        if deps.config_manager:
            ctx.register("config", deps.config_manager)

        # 9. Mark dependencies as fully initialized
        deps.mark_initialized()

        logger.info("Application bootstrap completed successfully")
        return ctx

    except Exception as e:
        logger.critical("Failed to bootstrap application", exc_info=True)
        raise RuntimeError("Failed to initialize application") from e


# --- Helper Functions ---


def _initialize_config(
    config_path: str | None,
    config_manager: IConfigManager | None = None,
    env_manager: EnvManager | None = None,
) -> IConfigManager:
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
        config_manager = ConfigManager(
            config_paths=config_paths, env_manager=env_manager
        )

    # Load config if not already loaded
    if not hasattr(config_manager, "_config"):
        config = config_manager.load_global_config()
    else:
        # Access the config directly from the instance if it's a ConfigManager
        config = getattr(config_manager, "get_global_config", None)

    if config is None:
        raise RuntimeError("Failed to load configuration")

    return config_manager


def _initialize_dependencies(config_manager: IConfigManager) -> AppDependencies:
    """Initialize application dependencies.

    Args:
        config_manager: Initialized config manager

    Returns:
        Initialized AppDependencies instance
    """
    deps = AppDependencies(config_manager=config_manager)

    return deps


def _setup_logging(
    config_manager: IConfigManager | None = None,
    log_level: int | None = None,
    logging_to_file: bool = False,
) -> None:
    """Configure application logging with the following priority:
    1. Use log_level if provided
    2. If config.global_config.runtime.enabled_logging is True, use config.global_config.runtime.log_level
    3. Default to logging.INFO
    4. If enabled_logging is False, use logging.CRITICAL

    If logging_to_file is True, logs will be written to:
    config.env_manager.path_manager.get_log_dir() / "LOCCA.log" with daily rotation

    Args:
        config_manager: Optional config manager instance
        log_level: Optional log level override (highest priority)
        logging_to_file: Whether to enable file logging
    """
    try:
        # Default log level
        final_log_level = logging.INFO

        # Check config for logging settings if no explicit log_level provided
        if log_level is None and config_manager:
            try:
                runtime_config = getattr(config_manager, "global_config", {}).get(
                    "runtime", {}
                )
                if not runtime_config.get("enabled_logging", True):
                    final_log_level = logging.CRITICAL
                else:
                    # Try to get log level from config
                    config_log_level = runtime_config.get("log_level", "INFO").upper()
                    final_log_level = getattr(logging, config_log_level, logging.INFO)
            except Exception as e:
                logging.warning(f"Failed to read logging config: {e}")
                final_log_level = logging.INFO

        # Apply explicit log level if provided (highest priority)
        if log_level is not None:
            final_log_level = log_level

        # Set up file logging if enabled
        log_file = None
        if logging_to_file and config_manager:
            try:
                log_dir = config_manager.path_manager.get_log_dir()
                log_file = log_dir / "LOCCA.log"
            except Exception as e:
                logging.warning(f"Failed to set up file logging: {e}")

        # Configure logging
        setup_logging(
            level=final_log_level,
            log_file=str(log_file) if log_file else None,
            time_rotation="midnight" if logging_to_file else None,
        )

    except Exception as e:
        # Fall back to basic logging if our setup fails
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Failed to configure logging: {e}", exc_info=True)


def _initialize_llm_manager(config_manager: IConfigManager) -> LLMManager | None:
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
        logger.error("Failed to initialize LLM manager", str(e), exc_info=True)
        return None


def _initialize_sandbox_manager(
    config_manager: IConfigManager,
) -> "SandboxManager | None":
    """Initialize the sandbox manager.

    Args:
        config_manager: The config manager implementing IConfigManager

    Returns:
        Initialized SandboxManager instance or None if initialization fails
    """
    try:
        sandbox_manager = SandboxManager(config_manager)

        # Check sandbox availability and update config if needed
        if (
            config_manager.global_config.sandbox.enabled
            and not sandbox_manager.ensure_availability()
        ):
            logger.warning(
                "Sandbox is not available. Disabling sandbox in session configuration."
            )

        logger.info("Sandbox manager initialized successfully")
        return sandbox_manager
    except Exception as e:
        logger.error("Failed to initialize sandbox manager", str(e), exc_info=True)
        return None


def _initialize_tool_manager(
    config_manager: IConfigManager,
    sandbox_manager: Any | None = None,
) -> IToolManager | None:
    """Initialize the tool manager.

    Args:
        config_manager: The config manager implementing IConfigManager
        sandbox_manager: Optional SandboxManager instance

    Returns:
        Initialized ToolManager instance or None if initialization fails
    """
    try:
        from local_coding_assistant.tools.tool_manager import (
            ToolManager,  # Local import
        )

        tool_manager: IToolManager = ToolManager(
            config_manager=config_manager, sandbox_manager=sandbox_manager
        )
        logger.info("Tool manager initialized successfully")

        return tool_manager
    except Exception as e:
        logger.error("Failed to initialize tool manager", str(e), exc_info=True)
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
        logger.error("Failed to initialize runtime manager", str(e), exc_info=True)
        return None
