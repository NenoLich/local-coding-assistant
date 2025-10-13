import inspect
import traceback
from collections.abc import Callable
from functools import wraps
from typing import Any

from local_coding_assistant.core.exceptions import LocalAssistantError
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("core.error_handler")


def handle_error(
    error: Exception,
    *,
    context: str | None = None,
    verbose: bool = False,
) -> None:
    """Central error handler for the application.

    Args:
        error: The exception instance to handle.
        context: Optional string describing where the error occurred.
        verbose: If True, log detailed traceback for debugging.
    """
    ctx = f"[{context}]" if context else ""

    if isinstance(error, LocalAssistantError):
        # Known, well-defined errors — just log nicely
        logger.error(f"{ctx} {error}".strip())
    else:
        # Unknown or unhandled exceptions
        logger.critical(
            f"{ctx} Unexpected error: {error.__class__.__name__}: {error}".strip()
        )

    if verbose:
        trace = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        # Use DEBUG for verbose trace details
        logger.debug(f"Traceback:\n{trace}")


def safe_entrypoint(context: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to wrap entrypoint functions with unified error handling.

    This version preserves the original function's signature for Typer/Click
    by using functools.wraps and setting __signature__ explicitly.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any | None:
            verbose = bool(kwargs.get("verbose", False))
            try:
                return func(*args, **kwargs)
            except Exception as e:  # central handler
                handle_error(e, context=context, verbose=verbose)
                return None

        # Ensure Typer sees the original callable signature
        try:
            wrapper.__signature__ = inspect.signature(func)
        except Exception:
            pass

        return wrapper

    return decorator
