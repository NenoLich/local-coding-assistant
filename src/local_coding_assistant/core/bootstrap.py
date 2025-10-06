"""Application bootstrap sequence and dependency injection."""

from .app_context import AppContext


def bootstrap(config_path: str | None = None) -> AppContext:
    """Initialize and configure the application.

    Args:
        config_path: Optional path to a configuration file

    Returns:
        Initialized application context
    """
    ctx = AppContext()

    # Register core components
    # TODO: Add component registration here

    return ctx
