import traceback
from collections.abc import Callable
from typing import ParamSpec, TypeVar

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


P = ParamSpec("P")
R = TypeVar("R")


def safe_entrypoint(context: str) -> Callable[[Callable[P, R]], Callable[P, R | None]]:
    """Decorator to wrap entrypoint functions with unified error handling.

    The decorated function must accept a keyword argument `verbose` (defaulting to False)
    or at least tolerate it in **kwargs. On exception, the error is logged and None is
    returned.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R | None]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
            verbose = bool(kwargs.get("verbose", False))
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_error(e, context=context, verbose=verbose)
                return None

        return wrapper

    return decorator
