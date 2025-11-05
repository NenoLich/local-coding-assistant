import inspect
import traceback
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from local_coding_assistant.core.exceptions import LocalAssistantError
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("core.error_handler")


def handle_error(
    error: Exception | None = None,
    *,
    context: str | None = None,
    verbose: bool = False,
    error_str: str | None = None,
) -> None:
    """Central error handler for the application.

    Args:
        error: The exception instance to handle (can be None if error_str is provided).
        context: Optional string describing where the error occurred.
        verbose: If True, log detailed traceback for debugging.
        error_str: Optional error message string if no exception object is available.
    """
    ctx = f"[{context}]" if context else ""

    # Handle case where error is None, but we have an error string
    if error is None and error_str:
        logger.critical(f"{ctx} {error_str}".strip())
        return

    # If we have neither an error object nor a string, log a generic message
    if error is None:
        logger.critical(f"{ctx} An unknown error occurred")
        return

    # Handle the error object
    try:
        if isinstance(error, LocalAssistantError):
            # Known, well-defined errors — just log nicely
            logger.error(f"{ctx} {error}".strip())
        else:
            # Unknown or unhandled exceptions
            error_name = getattr(error, "__name__", str(type(error).__name__))
            error_msg = str(error) if str(error) else "No error message provided"
            logger.critical(
                f"{ctx} Unexpected error: {error_name}: {error_msg}".strip()
            )

        if verbose:
            trace = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            # Use DEBUG for verbose trace details
            logger.debug(f"Traceback:\n{trace}")

    except Exception as e:
        # If something goes wrong during error handling, log it safely
        logger.critical(
            f"{ctx} Error while handling error: {e}\n"
            f"Original error type: {type(error).__name__}\n"
            f"Original error: {error}"
        )


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
