import inspect
import traceback
from collections.abc import Callable
from functools import wraps
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


T = TypeVar("T")
P = ParamSpec("P")


def safe_entrypoint(context: str) -> Callable[[Callable[P, T]], Callable[P, T | None]]:
    """Decorator to wrap entrypoint functions with unified error handling.
    Does not require the wrapped function to accept a `verbose` parameter.
    If `verbose` is not provided, defaults to `False`.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T | None]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            # Safely get `verbose` from kwargs, defaulting to False if not provided
            verbose = kwargs.get("verbose", False)
            try:
                return func(*args, **kwargs)
            except Exception as err:
                # Re-raise typer/click Exit exceptions
                if "Exit" in err.__class__.__name__:
                    raise err
                handle_error(err, context=context, verbose=verbose)
                return None

        # Preserve the original function's signature
        try:
            wrapper.__signature__ = inspect.signature(func)
        except Exception as e:
            # Use `cast` to tell the type checker that `func` has a `__name__` attribute
            func_name = getattr(func, "__name__", str(func))
            print(f"Warning: Failed to set signature for {func_name}: {e}")

        return wrapper

    return decorator
