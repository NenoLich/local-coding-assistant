"""Application bootstrap sequence and dependency injection."""

from __future__ import annotations

import asyncio
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_coding_assistant.config.env_manager import EnvManager, get_env_manager
from local_coding_assistant.core import telemetry as telemetry_module
from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.core.dependencies import AppDependencies
from local_coding_assistant.core.protocols import IConfigManager
from local_coding_assistant.utils.logging import (
    get_logger,
    setup_logging,
    temporary_log_level,
)

if TYPE_CHECKING:
    from local_coding_assistant.agent.llm.service import LLMService
    from local_coding_assistant.repository.service import RepositoryContextService
    from local_coding_assistant.runtime.events.pipeline import EventPipeline
    from local_coding_assistant.runtime.integrations.dashboard import (
        DashboardIntegration,
    )
    from local_coding_assistant.runtime.manager import RuntimeManager
    from local_coding_assistant.sandbox.manager import SandboxManager
    from local_coding_assistant.storage import IExecutionStore
    from local_coding_assistant.storage.execution_store import ExecutionStore

logger = get_logger("core.bootstrap")


def bootstrap(
    config_path: str | None = None,
    *,
    log_level: int | None = None,
    config_manager: IConfigManager | None = None,
    target_project_root: str | Path | None = None,
) -> AppContext:
    """Initialize and configure the application.

    Synchronous by design: builds configuration-driven objects only. Async
    resources (the execution store connection) are opened by
    :func:`initialize` inside the caller's event loop.

    Args:
        config_path: Optional path to configuration file
        log_level: Override log level
        config_manager: Optional pre-configured config manager (for testing)
        target_project_root: Optional target project root override.
            If provided, sets the target project root before initializing
            repository context service and other components that depend on it.

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

        # 2. Get environment manager
        env_manager = get_env_manager()
        logger.debug("Environment setup completed")

        # 3. Initialize configuration
        config_manager = _initialize_config(config_path, config_manager, env_manager)
        logger.debug("Configuration loaded successfully")

        # Apply target_project_root override before initializing services that depend on it
        if target_project_root is not None:
            config_manager.path_manager.set_target_project_root(target_project_root)
            logger.debug(
                "Applied target_project_root override",
                target_project_root=str(target_project_root),
            )

        # 4. Now setup full logging with the loaded config
        _setup_logging(config_manager, log_level, logging_to_file=True)
        logger = get_logger("core.bootstrap")  # Reinitialize logger after setup

        # 4. Initialize dependencies with the config manager
        deps = _initialize_dependencies(config_manager)

        # 5. Create context with dependencies
        ctx = AppContext(dependencies=deps)

        # 6. Initialize components
        deps.config_manager = config_manager

        # Create execution store for persistence (connection opened in initialize())
        execution_store = _initialize_execution_store(config_manager)
        deps.execution_store = execution_store
        ctx.register("execution_store", execution_store)

        # 7. Initialize dashboard integration if enabled (before pipeline so
        #    the DashboardEventSubscriber can be subscribed alongside the
        #    StoreSubscriber).
        dashboard_integration = _initialize_dashboard_integration(config_manager)

        # Create event pipeline with default subscribers before the runtime,
        # so RuntimeManager can create its turn/frame components.
        event_pipeline = _initialize_event_pipeline(
            config_manager, execution_store, dashboard_integration
        )
        deps.event_pipeline = event_pipeline
        ctx.register("event_pipeline", event_pipeline)

        # 8. Auto-start dashboard if configured and not already running.
        _auto_start_dashboard(config_manager, ctx)

        deps.sandbox_manager = _initialize_sandbox_manager(config_manager)
        deps.llm_service = _initialize_llm_service(config_manager)
        deps.repository_context_service = _initialize_repository_context_service(
            config_manager
        )

        # Initialize langgraph capabilities after LLM service
        _initialize_langgraph_capabilities(config_manager)

        deps.tool_gateway = _initialize_tool_gateway(
            config_manager=config_manager,
            repository_context_service=deps.repository_context_service,
        )
        deps.runtime_manager = _initialize_runtime_manager(
            config_manager=config_manager,
            llm_service=deps.llm_service,
            tool_gateway=deps.tool_gateway,
            repository_context_service=deps.repository_context_service,
            event_pipeline=event_pipeline,
            store=execution_store,
        )
        if deps.llm_service:
            ctx.register("llm", deps.llm_service)
        if deps.tool_gateway:
            ctx.register("tool_gateway", deps.tool_gateway)
        if deps.runtime_manager:
            ctx.register("runtime", deps.runtime_manager)
        if deps.execution_store:
            ctx.register("store", deps.execution_store)
        if deps.repository_context_service:
            ctx.register("repository_context", deps.repository_context_service)
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


async def initialize(ctx: AppContext) -> None:
    """Open async resources for a bootstrapped context.

    Must be awaited inside the same event loop that will run the runtime
    (aiosqlite resolves futures in the calling loop). Idempotent.

    Args:
        ctx: Context returned by :func:`bootstrap`

    Raises:
        RuntimeError: If the execution store cannot be opened
    """
    store = ctx.get("execution_store")
    if store is None:
        raise RuntimeError("Execution store missing; did bootstrap() run?")

    try:
        await store.init()
        # Resolve the persisted workspace identity now that the store
        # connection is open: reuse an existing record for the resolved root,
        # otherwise create one, and refresh the stored config_hash so config
        # drift is detectable rather than silently invalidating history on
        # resume.
        tool_gateway = ctx.get("tool_gateway")
        if tool_gateway is not None and hasattr(tool_gateway, "init_identity"):
            await tool_gateway.init_identity(store=store)
        logger.info("Application initialize completed successfully")
    except Exception as e:
        logger.critical("Failed to initialize application", exc_info=True)
        raise RuntimeError("Failed to initialize application") from e


async def shutdown(ctx: AppContext) -> None:
    """Flush and close async resources for a bootstrapped context.

    Must be awaited in the same loop as :func:`initialize`. Flushes pending
    events through the pipeline/store writer queue before closing.

    Args:
        ctx: Context returned by :func:`bootstrap`
    """
    pipeline = ctx.get("event_pipeline")
    if pipeline is not None:
        await _close_async_resource(pipeline.close, "event pipeline")

    store = ctx.get("execution_store")
    if store is not None:
        await _close_async_resource(store.close, "execution store")

    # Close repository context service (stops file monitoring)
    # Runs in a worker thread: observer.join() is blocking and can hang
    # indefinitely on a stuck watchdog thread, which would pin the event loop.
    repository_context_service = ctx.get("repository_context")
    if repository_context_service is not None:
        await asyncio.to_thread(
            _close_sync_resource,
            repository_context_service.close,
            "repository context service",
        )
        logger.info("Repository context service closed")

    # Terminate dashboard if bootstrap started it (externally started
    # dashboards are not managed by bootstrap).
    # Runs in a worker thread: process.wait() is blocking.
    dashboard_process = ctx.get("dashboard_process")
    if dashboard_process is not None:
        await asyncio.to_thread(_terminate_process, dashboard_process)

    # Telemetry force_flush/shutdown can block on OTLP export.
    await asyncio.to_thread(telemetry_module.shutdown_telemetry)
    logger.debug("Application shutdown completed")


async def _close_async_resource(close_call, label: str) -> None:
    """Close an async resource, treating CancelledError as expected during shutdown.

    Ctrl-C injects CancelledError into the running task; asyncio.gather with
    return_exceptions=True swallows only Exception, so CancelledError can
    propagate from pipeline/store close. That is expected, not a failure.
    """
    try:
        await close_call()
    except asyncio.CancelledError:
        logger.debug("%s close cancelled during shutdown", label)
    except BaseException as e:
        logger.warning("Failed to close %s", label, error=str(e))


def _close_sync_resource(close_call, label: str) -> None:
    """Close a sync resource, treating CancelledError as expected during shutdown."""
    try:
        close_call()
    except asyncio.CancelledError:
        logger.debug("%s close cancelled during shutdown", label)
    except BaseException as e:
        logger.warning("Failed to close %s", label, error=str(e))


def _terminate_process(process) -> None:
    """Terminate a subprocess, treating CancelledError as expected during shutdown."""
    try:
        process.terminate()
        process.wait(timeout=5)
        logger.info("Dashboard auto-started process terminated")
    except asyncio.CancelledError:
        logger.debug("Dashboard process terminate cancelled during shutdown")
    except BaseException as e:
        logger.warning(f"Failed to terminate dashboard process: {e}")
        try:
            process.kill()
        except asyncio.CancelledError:
            logger.debug("Dashboard process kill cancelled during shutdown")
        except BaseException as e:
            logger.warning(f"Failed to kill dashboard process: {e}")


# --- Helper Functions ---


def _build_litellm_custom_models(providers: dict[str, Any]) -> dict[str, dict]:
    """Build LiteLLM custom model entries from loaded provider instances.

    Args:
        providers: Mapping of provider name to provider instance.

    Returns:
        Dict of litellm model keys to model configuration dicts.
    """
    custom_models: dict[str, dict] = {}
    for _provider_name, provider in providers.items():
        driver = getattr(provider, "driver", "")
        models = getattr(provider, "models", [])

        # Only register models for openai_chat and openai_responses drivers
        if driver not in ("openai_chat", "openai_responses"):
            continue

        for model in models:
            model_name = model if isinstance(model, str) else getattr(model, "name", "")
            if not model_name:
                continue

            # Determine provider prefix based on driver
            litellm_provider = "openai"
            model_key = (
                f"openai/{model_name}"
                if driver == "openai_chat"
                else f"openai/responses/{model_name}"
            )

            custom_models[model_key] = {
                "max_tokens": 1_000_000,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "litellm_provider": litellm_provider,
                "mode": "chat" if driver == "openai_chat" else "responses",
            }

    return custom_models


def _register_litellm_models_from_config(
    provider_manager: Any = None,
) -> None:
    """Register provider models with LiteLLM from configuration.

    Reads provider configurations from the loaded provider manager and
    registers models with LiteLLM to bypass cost mapping for unmapped models.
    Only called if auto_register_litellm_models is enabled.

    Args:
        provider_manager: The provider manager with loaded provider instances
    """
    try:
        import litellm

        providers = {}
        if provider_manager is not None:
            for name in provider_manager.list_providers():
                provider = provider_manager.get_provider(name)
                if provider:
                    providers[name] = provider

        if not providers:
            logger.debug("No providers configured, skipping LiteLLM model registration")
            return

        custom_models = _build_litellm_custom_models(providers)

        if custom_models:
            with temporary_log_level({"LiteLLM": logging.ERROR}):
                litellm.register_model(custom_models)
            logger.info(
                f"Registered {len(custom_models)} models with LiteLLM from provider config"
            )
        else:
            logger.debug("No models to register with LiteLLM")

    except ImportError:
        logger.debug("LiteLLM not available, skipping model registration")
    except Exception as e:
        logger.warning(f"Failed to register models with LiteLLM: {e}")


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
        config_paths: list[Path | str] = [Path(config_path)] if config_path else []
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


def _initialize_execution_store(
    config_manager: IConfigManager,
) -> ExecutionStore | None:
    """Initialize the execution store from configuration.

    Only constructs the store; the SQLite connection is opened by
    :func:`initialize` inside the caller's event loop.

    Args:
        config_manager: Initialized config manager

    Returns:
        Configured ExecutionStore instance or None if disabled
    """
    try:
        from local_coding_assistant.storage.execution_store import ExecutionStore

        store_config = config_manager.global_config.execution_store
        if not store_config.enabled:
            logger.info("Execution store disabled in configuration")
            return None

        db_path = config_manager.path_manager.resolve_path(store_config.db_path)
        store = ExecutionStore(
            db_path,
            flush_interval_ms=store_config.flush_interval_ms,
            batch_size=store_config.batch_size,
        )
        config_manager.register_module("execution_store", {"store_available": True})
        logger.info("Execution store configured", db_path=str(db_path))
        return store

    except Exception as e:
        logger.error(
            "Failed to initialize execution store", error=str(e), exc_info=True
        )
        return None


def _initialize_event_pipeline(
    config_manager: IConfigManager,
    execution_store: IExecutionStore | None,
    dashboard_integration: DashboardIntegration | None = None,
) -> EventPipeline:
    """Create the runtime event pipeline with its default subscribers.

    Args:
        config_manager: Initialized config manager
        execution_store: Optional store for durable persistence
        dashboard_integration: Optional DashboardIntegration for event
            forwarding. When provided and enabled, a DashboardEventSubscriber
            is subscribed alongside the StoreSubscriber.

    Returns:
        Configured EventPipeline
    """
    from local_coding_assistant.runtime.engine.subscribers.dashboard import (
        DashboardEventSubscriber,
    )
    from local_coding_assistant.runtime.engine.subscribers.store_subscriber import (
        StoreSubscriber,
    )
    from local_coding_assistant.runtime.events.pipeline import EventPipeline

    pipeline = EventPipeline(execution_store=execution_store)

    if execution_store is not None:
        pipeline.subscribe(StoreSubscriber(execution_store))

    if dashboard_integration is not None and dashboard_integration.is_enabled():
        pipeline.subscribe(DashboardEventSubscriber(dashboard_integration))

    if telemetry_module.is_enabled(config_manager):
        telemetry_module.configure_telemetry(config_manager)
        try:
            from local_coding_assistant.runtime.engine.subscribers.otel_subscriber import (
                OTelSubscriber,
            )

            pipeline.subscribe(OTelSubscriber())
            config_manager.register_capability(["otel"])
            logger.info("OTel subscriber enabled")
        except ImportError:
            logger.warning("OTel subscriber requested but not available")
    else:
        logger.debug("Telemetry disabled, OTel subscriber not attached")

    return pipeline


def _initialize_llm_service(config_manager: IConfigManager) -> LLMService | None:
    """Initialize the LLM service.

    Args:
        config_manager: The config manager implementing IConfigManager

    Returns:
        Initialized LLMService instance or None if initialization fails
    """
    try:
        # Lazy import to avoid circular imports and heavy module loading at startup
        from local_coding_assistant.agent.llm.service import LLMService

        llm_service = LLMService(config_manager)
        config_manager.register_module("llm_service")
        logger.info("LLM service initialized successfully")

        # Register models with LiteLLM if auto-registration is enabled
        if config_manager.global_config.llm.auto_register_litellm_models:
            _register_litellm_models_from_config(llm_service.provider_manager)

        return llm_service

    except Exception as e:
        logger.error("Failed to initialize LLM service", str(e), exc_info=True)
        return None


def _check_langgraph_availability() -> bool:
    """Check if langgraph is installed and available."""
    import importlib.util

    return (
        importlib.util.find_spec("langgraph") is not None
        and importlib.util.find_spec("langgraph.graph") is not None
    )


def _initialize_langgraph_capabilities(
    config_manager: IConfigManager,
) -> dict[str, bool]:
    """Initialize and register langgraph capabilities.

    Args:
        config_manager: The config manager implementing IConfigManager

    Returns:
        Dictionary of langgraph-related capabilities
    """
    langgraph_capabilities = {"langgraph_installed": _check_langgraph_availability()}

    if langgraph_capabilities["langgraph_installed"]:
        config_manager.register_capability(["langgraph_installed"])
        logger.debug("LangGraph capabilities registered successfully")
    else:
        logger.debug("LangGraph not available - graph mode will be disabled")

    return langgraph_capabilities


def _initialize_sandbox_manager(
    config_manager: IConfigManager,
) -> SandboxManager | None:
    """Initialize the sandbox manager.

    Args:
        config_manager: The config manager implementing IConfigManager

    Returns:
        Initialized SandboxManager instance or None if initialization fails
    """
    if not config_manager.global_config.sandbox.enabled:
        return None
    try:
        from local_coding_assistant.sandbox.manager import SandboxManager

        sandbox_manager = SandboxManager(config_manager)

        # Register sandbox capabilities through ConfigManager
        sandbox_capabilities = {
            "sandbox_available": sandbox_manager.ensure_availability()
        }
        config_manager.register_module("sandbox_manager", sandbox_capabilities)

        logger.info("Sandbox manager initialized successfully")
        return sandbox_manager
    except Exception as e:
        logger.error("Failed to initialize sandbox manager", str(e), exc_info=True)
        return None


def _initialize_repository_context_service(
    config_manager: IConfigManager,
) -> RepositoryContextService | None:
    """Initialize the repository context service.

    Args:
        config_manager: The config manager implementing IConfigManager

    Returns:
        Initialized RepositoryContextService instance or None if initialization fails
    """
    try:
        from local_coding_assistant.repository.service import RepositoryContextService

        # Get repository configuration
        repo_config = config_manager.global_config.repository

        if not repo_config.enabled:
            return None

        # Get project root from config or use current directory
        target_project_root = config_manager.path_manager.get_target_project_root()

        # Initialize repository service
        repository_service = RepositoryContextService(
            target_project_root=target_project_root,
            config_manager=config_manager,
        )

        # Register repository capabilities through ConfigManager
        repo_capabilities = {
            "repository_available": True,
            "storage_mode": repo_config.storage.mode,
        }
        config_manager.register_module("repository_context_service", repo_capabilities)

        logger.info("Repository context service initialized successfully")
        return repository_service

    except Exception as e:
        logger.error(
            "Failed to initialize repository context service", str(e), exc_info=True
        )
        return None


def _initialize_runtime_manager(
    config_manager: IConfigManager,
    llm_service: LLMService | None = None,
    tool_gateway: Any | None = None,
    repository_context_service: RepositoryContextService | None = None,
    event_pipeline: EventPipeline | None = None,
    store: IExecutionStore | None = None,
) -> RuntimeManager | None:
    """Initialize the runtime manager.

    Args:
        config_manager: The config manager instance (required)
        llm_service: Optional LLM service instance
        tool_gateway: Optional tool gateway instance
        repository_context_service: Optional repository context service instance
        event_pipeline: Optional event pipeline for runtime events
        store: Optional execution store instance

    Returns:
        Initialized RuntimeManager instance or None if initialization fails
    """
    try:
        from local_coding_assistant.runtime.manager import RuntimeManager

        if not llm_service:
            logger.warning(
                "LLM service not available, skipping runtime manager initialization"
            )
            return None

        if event_pipeline is None:
            logger.warning(
                "Event pipeline not available, runtime manager will not persist events"
            )

        runtime_manager = RuntimeManager(
            config_manager=config_manager,
            llm_service=llm_service,
            tool_gateway=tool_gateway,
            repository_context_service=repository_context_service,
            event_pipeline=event_pipeline,
            store=store,
        )

        logger.info("Runtime manager initialized successfully")
        return runtime_manager

    except Exception as e:
        logger.error(
            "Failed to initialize runtime manager", error=str(e), exc_info=True
        )
        return None


def _initialize_tool_gateway(
    config_manager: IConfigManager,
    repository_context_service: RepositoryContextService | None = None,
) -> Any | None:
    """Initialize the ToolGateway for the new tools platform.

    All gateway behavior is resolved through the ConfigManager
    (`tools_platform` section of the global config): backend selection,
    workspace root and path policy, snapshots, approval policy, timeout
    and output limits, scheduler concurrency, tool exposure, and tool
    input limits. Nothing gateway-related is hardcoded here.

    Args:
        config_manager: The config manager instance
        repository_context_service: Optional repository context service for
            symbol search operations

    Returns:
        Initialized ToolGateway instance or None if initialization fails
    """
    try:
        from local_coding_assistant.tools_platform.gateway import ToolGateway
        from local_coding_assistant.tools_platform.registry import ToolRegistry
        from local_coding_assistant.tools_platform.tools.register import (
            register_all_tools,
        )

        # Initialize tool registry (code-owned definitions, config is policy only)
        registry = ToolRegistry()
        register_all_tools(registry)

        # Initialize gateway with config_manager for dynamic config resolution
        gateway = ToolGateway(
            registry=registry,
            config_manager=config_manager,
            repository_context_service=repository_context_service,
        )

        config_manager.register_module("tool_gateway", {"tools_available": True})
        logger.info("Tool gateway initialized successfully")
        return gateway

    except Exception as e:
        logger.error("Failed to initialize tool gateway", error=str(e), exc_info=True)
        return None


def _initialize_dashboard_integration(
    config_manager: IConfigManager,
) -> DashboardIntegration | None:
    """Initialize dashboard integration if enabled.

    Returns the DashboardIntegration instance when enabled, None otherwise.
    """
    try:
        # Check if dashboard is enabled
        dashboard_config = config_manager.global_config.dashboard
        if not dashboard_config.enabled:
            logger.debug("Dashboard integration disabled in configuration")
            return None

        # Import and enable dashboard integration with URL from config
        from local_coding_assistant.runtime.integrations.dashboard import (
            enable_dashboard_integration,
            get_dashboard_integration,
        )

        dashboard_url = f"http://{dashboard_config.host}:{dashboard_config.port}"
        enable_dashboard_integration(dashboard_url=dashboard_url)

        config_manager.register_capability(["dashboard_enabled"])
        logger.info(f"Dashboard integration enabled, URL: {dashboard_url}")
        return get_dashboard_integration()

    except ImportError:
        logger.debug("Dashboard module not available, skipping integration")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize dashboard integration: {e}")
        return None


def _is_dashboard_running(host: str, port: int) -> bool:
    """Check if a dashboard server is already listening on host:port."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _auto_start_dashboard(config_manager: IConfigManager, ctx: AppContext) -> None:
    """Auto-start the dashboard server if configured and not already running.

    If bootstrap spawns the dashboard, the process is registered in ctx
    so shutdown() can terminate it. If a dashboard is already running,
    auto-start is skipped and the external process is not managed.
    """
    try:
        dashboard_config = config_manager.global_config.dashboard
        if not dashboard_config.auto_start:
            return

        host = dashboard_config.host
        port = dashboard_config.port

        if _is_dashboard_running(host, port):
            logger.info(
                "Dashboard already running on %s:%d, skipping auto-start",
                host,
                port,
            )
            return

        # Resolve the execution store db path for the dashboard subprocess
        store_config = config_manager.global_config.execution_store
        db_path = (
            config_manager.path_manager.resolve_path(store_config.db_path)
            if store_config and store_config.enabled
            else "data/execution_store.db"
        )

        project_root = config_manager.path_manager.get_project_root()
        runner_path = (
            project_root
            / "src"
            / "local_coding_assistant"
            / "dashboard"
            / "run_dashboard.py"
        )
        cmd = [
            sys.executable,
            str(runner_path),
            "--host",
            host,
            "--port",
            str(port),
            "--store-db-path",
            str(db_path),
            "--log-level",
            "info",
        ]

        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        process = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=project_root,
            shell=False,
            **kwargs,
        )

        # Give it a moment to start
        time.sleep(1)

        if process.poll() is None:
            logger.info(
                "Dashboard auto-started with PID %d on %s:%d",
                process.pid,
                host,
                port,
            )
            ctx.register("dashboard_process", process)
        else:
            logger.warning(
                "Dashboard auto-start failed (return code: %s)",
                process.returncode,
            )
    except Exception as e:
        logger.warning(f"Failed to auto-start dashboard: {e}")
