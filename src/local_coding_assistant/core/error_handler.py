import inspect
import traceback
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from local_coding_assistant.core.exceptions import LocalAssistantError
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("core.error_handler")

# Type variables for generic function typing
T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

# Type aliases
type ErrorHandler = Callable[[Exception, str | None, bool], None]


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


def error_handler[**P, R](
    func: Callable[P, R] | None = None,
    *,
    context: str | None = None,
    verbose: bool = False,
    error_callback: ErrorHandler | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R | None]] | Callable[P, R | None]:
    """Decorator to wrap functions with error handling.

    Args:
        func: The function to decorate (automatically passed by the decorator).
        context: Optional context string for error messages.
        verbose: Whether to include full traceback in logs.
        error_callback: Optional custom error handler function.

    Returns:
        The decorated function with error handling.
    """

    def decorator(f: Callable[P, R]) -> Callable[P, R | None]:
        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
            try:
                return f(*args, **kwargs)
            except Exception as err:
                if error_callback:
                    error_callback(err, context, verbose)
                else:
                    handle_error(err, context=context, verbose=verbose)
                return None

        # Copy the signature from the original function
        _copy_function_signature(f, wrapper)
        return wrapper

    # Handle both @error_handler and @error_handler() syntax
    if func is not None:
        return decorator(func)
    return decorator


def async_error_handler[**P, R](
    func: Callable[P, Awaitable[R]] | None = None,
    *,
    context: str | None = None,
    verbose: bool = False,
    error_callback: ErrorHandler | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R | None]]]:
    """Decorator to wrap async functions with error handling.

    Args:
        func: The async function to decorate.
        context: Optional context string for error messages.
        verbose: Whether to include full traceback in logs.
        error_callback: Optional custom error handler function.

    Returns:
        The decorated async function with error handling.
    """

    def decorator(f: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R | None]]:
        @wraps(f)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
            try:
                return await f(*args, **kwargs)
            except Exception as err:
                if error_callback:
                    error_callback(err, context, verbose)
                else:
                    handle_error(err, context=context, verbose=verbose)
                return None

        # Safely update annotations if they exist
        if hasattr(f, "__annotations__"):
            wrapper.__annotations__ = {
                "return": f"typing.Awaitable[{R.__name__} | None]",
                **getattr(f, "__annotations__", {}),
            }

        # Copy the signature from the original function
        _copy_function_signature(f, wrapper)
        return wrapper

    # Handle both @async_error_handler and @async_error_handler() syntax
    if func is not None:
        return decorator(func)  # type: ignore[arg-type]
    return decorator


def _copy_function_signature(
    original_func: Callable[..., Any], wrapper_func: Callable[..., Any]
) -> None:
    """Copy function signature and metadata from original to wrapper function.

    Args:
        original_func: The original function to copy metadata from.
        wrapper_func: The wrapper function to copy metadata to.
    """
    try:
        # Copy signature if available
        if hasattr(original_func, "__signature__"):
            wrapper_func.__signature__ = original_func.__signature__  # type: ignore[attr-defined]
        else:
            signature = inspect.signature(original_func)
            wrapper_func.__signature__ = signature  # type: ignore[attr-defined]

        # Copy other metadata if they exist
        if hasattr(original_func, "__module__"):
            wrapper_func.__module__ = original_func.__module__

        if hasattr(original_func, "__doc__"):
            wrapper_func.__doc__ = original_func.__doc__

        if hasattr(original_func, "__annotations__"):
            wrapper_func.__annotations__ = original_func.__annotations__.copy()

        # Copy other attributes that might be useful
        for attr in ("__name__", "__qualname__"):
            try:
                if hasattr(original_func, attr):
                    setattr(wrapper_func, attr, getattr(original_func, attr))
            except (AttributeError, TypeError) as e:
                logger.debug(
                    f"Could not copy attribute {attr}", error=str(e), exc_info=True
                )
    except Exception as e:
        # If we can't copy the signature, log a warning but continue
        func_name = getattr(original_func, "__qualname__", str(original_func))
        logger.warning(
            f"Failed to copy signature for {func_name}", error=str(e), exc_info=True
        )


def safe_entrypoint(context: str) -> Callable[[Callable[P, T]], Callable[P, T | None]]:
    """Decorator to wrap entrypoint functions with unified error handling.

    This is a specialized version of the error_handler decorator for CLI entry points.
    It automatically handles verbose flag and re-raises Exit exceptions from CLI frameworks.

    Args:
        context: Context string for error messages.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T | None]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            verbose = bool(kwargs.get("verbose", False))
            try:
                return func(*args, **kwargs)
            except Exception as err:
                # Re-raise typer/click Exit exceptions
                if "Exit" in err.__class__.__name__:
                    raise err
                handle_error(err, context=context, verbose=verbose)
                return None

        # Copy the signature and other metadata
        _copy_function_signature(func, wrapper)
        return wrapper

    return decorator
