"""Application bootstrap sequence and dependency injection."""

import logging
from pathlib import Path
from typing import Any

from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.config import get_config_manager
from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.config.env_manager import EnvManager
from local_coding_assistant.core.app_context import AppContext
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
    _env_manager = EnvManager()
    try:
        _env_manager.load_env_files()
    except Exception as e:
        print(f"Warning: Failed to load .env files: {e}")

    ctx = AppContext()

    # Load configuration
    config, config_is_valid = _load_config(config_path)

    # Configure logging
    # Determine the appropriate log level
    effective_log_level = log_level
    if effective_log_level is None and config is not None:
        if hasattr(config, "runtime") and not config.runtime.enable_logging:
            effective_log_level = 50  # CRITICAL level to disable logging

    _setup_logging(config, effective_log_level)

    # Initialize LLM manager
    llm_manager = _initialize_llm_manager(config, config_is_valid, config_path)

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


def _load_config(config_path: str | None) -> tuple[Any, bool]:
    """Load application configuration.

    Returns:
        Tuple of (config, is_valid) where is_valid indicates if the loaded config passed validation
    """
    if config_path:
        config_paths = [Path(config_path)]
        try:
            # Check if file exists first
            if not Path(config_path).exists():
                from local_coding_assistant.core.exceptions import ConfigError

                raise ConfigError(f"Configuration file not found: {config_path}")

            # Create ConfigManager with specific paths and load config
            # This becomes our global manager for the rest of the application
            global_manager = ConfigManager(config_paths)
            config = global_manager.load_global_config()
            global_manager._session_overrides = {}

            # Set this as the global manager for the rest of the app
            import local_coding_assistant.config.config_manager as cm

            cm._config_manager = global_manager

            return config, True
        except Exception as err:
            # Re-raise ConfigError for file not found or invalid YAML
            from local_coding_assistant.core.exceptions import ConfigError

            if isinstance(err, ConfigError):
                # Check if this is a validation error (not file not found or YAML error)
                if (
                    "validation" in str(err).lower()
                    or "invalid global configuration" in str(err).lower()
                ):
                    logger.warning(f"Configuration schema validation failed: {err}")
                    logger.info("Falling back to default configuration")
                    # Return None config but mark as invalid
                    return None, False
                else:
                    # Re-raise for actual file errors
                    raise

            # Check for validation errors specifically
            if "validation" in str(err).lower() or (
                hasattr(err, "__class__")
                and "validation" in err.__class__.__name__.lower()
            ):
                logger.warning(f"Configuration schema validation failed: {err}")
                logger.info("Falling back to default configuration")
                return None, False

            # For all other errors, fall back to defaults
            logger.warning(f"Failed to load configuration: {err}")
            logger.info("Falling back to default configuration")
            return None, False
    else:
        # No custom config path, use default global manager
        try:
            global_manager = get_config_manager()
            config = global_manager.load_global_config()
            global_manager._session_overrides = {}
            return config, True
        except Exception as err:
            logger.warning(f"Failed to load default configuration: {err}")
            logger.info("Continuing without configuration")
            return None, False


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
            # Convert string log level to integer using explicit mapping
            log_level_str = config.runtime.log_level or "INFO"
            # Ensure log_level_str is a string (handles test mocks)
            if hasattr(log_level_str, "__str__"):
                log_level_str = str(log_level_str)

            # Use explicit mapping instead of getattr for reliability
            log_level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            level = log_level_map.get(log_level_str.upper(), logging.INFO)
            setup_logging(level=level)
    else:
        # Config failed to load or logging explicitly disabled - use provided log_level or CRITICAL
        if log_level is not None:
            setup_logging(level=log_level)
        else:
            setup_logging(level=50)  # CRITICAL level to effectively disable


def _initialize_llm_manager(
    config: Any, config_is_valid: bool, config_path: str | None
) -> LLMManager | None:
    """Initialize the LLM manager."""

    # Lazy import to avoid circular imports
    from local_coding_assistant.providers.provider_manager import provider_manager

    # If config was invalid, don't initialize LLM manager even if defaults are available
    if not config_is_valid and config_path is not None:
        logger.info("Skipping LLM manager initialization due to invalid configuration")
        return None

    try:
        # Use the global config manager which already has the correct config loaded
        config_manager = get_config_manager()

        # Try to initialize LLM manager - this may fail if config is invalid
        llm_manager = LLMManager(config_manager, provider_manager)
        logger.info("LLM manager initialized successfully")
        return llm_manager
    except Exception as err:
        # Check if this is a schema validation error
        if (
            "validation" in str(err).lower()
            or "schema" in str(err).lower()
            or "invalid" in str(err).lower()
        ):
            logger.warning(
                f"LLM manager initialization failed due to config validation: {err}"
            )
            logger.info("Continuing without LLM functionality")
        else:
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
    if llm_manager is not None:
        # Use the global config manager which already has the correct config loaded
        runtime_config_manager = get_config_manager()

        return RuntimeManager(
            llm_manager=llm_manager,
            tool_manager=tool_manager,
            config_manager=runtime_config_manager,
        )
    else:
        if llm_manager is None:
            logger.warning("Skipping runtime manager creation due to missing LLM")
        return None
