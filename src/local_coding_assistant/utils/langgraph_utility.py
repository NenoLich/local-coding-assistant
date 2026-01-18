"""LangGraph utilities for error handling, logging, and node decoration."""

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

import structlog

from local_coding_assistant.core.exceptions import AgentError, ToolRegistryError
from local_coding_assistant.utils.logging import get_logger

log = get_logger("agent")


def node_logger(node_name: str) -> structlog.BoundLogger:
    """Factory function to create loggers for specific LangGraph nodes.

    Args:
        node_name: Name of the node for logging context

    Returns:
        Logger instance configured for the specific node
    """
    return get_logger(f"agent.langgraph.node.{node_name}")


def handle_graph_error(error: Exception, state: dict[str, Any]) -> None:
    """Handle errors that occur during graph execution and update state accordingly.

    Args:
        error: The exception that occurred
        state: Current graph state to modify based on error type
    """
    log.error(f"Graph execution error: {error}")

    # Update state based on error type
    if isinstance(error, AgentError):
        state["error"] = {
            "type": "agent_error",
            "message": str(error),
            "should_stop": True,
        }
        log.warning(f"Agent error in graph execution: {error}")
    elif isinstance(error, ToolRegistryError):
        state["error"] = {
            "type": "tool_error",
            "message": str(error),
            "should_stop": False,  # Allow retry for tool errors
        }
        log.warning(f"Tool error in graph execution: {error}")
    else:
        # Generic error handling
        state["error"] = {
            "type": "unknown_error",
            "message": str(error),
            "should_stop": True,  # Stop for unknown errors
        }
        log.error(f"Unknown error in graph execution: {error}")


def _extract_args(args: tuple, kwargs: dict) -> tuple[Any, Any, Any]:
    """Extract state, writer, and instance from function arguments.

    Args:
        args: Positional arguments passed to the function
        kwargs: Keyword arguments passed to the function

    Returns:
        Tuple of (state, writer, instance)
    """
    if args and hasattr(args[0], "__self__"):  # It's a bound method
        return (
            kwargs.get("state") or (args[1] if len(args) > 1 else {}),  # state
            kwargs.get("writer"),  # writer
            args[0],  # instance
        )
    # It's a standalone function
    return (
        kwargs.get("state") or (args[0] if args else {}),  # state
        kwargs.get("writer"),  # writer
        None,  # instance
    )


def _validate_and_process_result(result: Any, node_name: str) -> dict[str, Any]:
    """Validate and process the result from a node function.

    Args:
        result: The result from the node function
        node_name: Name of the node for error messages

    Returns:
        Processed result as a dictionary

    Raises:
        TypeError: If the result is not a dict or Pydantic model
    """
    if hasattr(result, "model_dump"):  # It's a Pydantic model like AgentState
        return result.model_dump()
    if not isinstance(result, dict):
        raise TypeError(
            f"Node {node_name} must return a dict or Pydantic model, got {type(result)}"
        )
    return result


def _create_node_wrapper(node_func: Callable, node_name: str, is_async: bool):
    """Create a wrapper function for a node with the specified execution mode.

    Args:
        node_func: The node function to wrap
        node_name: Name of the node for logging
        is_async: Whether the function is async

    Returns:
        Wrapped function with logging and error handling
    """

    @wraps(node_func)
    async def async_wrapper(*args, **kwargs) -> dict[str, Any]:
        state, writer, instance = _extract_args(args, kwargs)
        logger = node_logger(node_name)
        logger.info(f"✨ Executing node: {node_name}")
        logger.debug(
            f"Input state keys: {list(state.keys()) if isinstance(state, dict) else 'non-dict'}"
        )

        try:
            # Run the node function
            if instance is not None and hasattr(instance, "__class__"):  # Method call
                result = await (
                    node_func(instance, state, writer)
                    if writer is not None
                    else node_func(instance, state)
                )
            else:  # Function call
                result = await (
                    node_func(state, writer=writer)
                    if writer is not None
                    else node_func(state)
                )

            result = _validate_and_process_result(result, node_name)
            logger.info(f"✅ Node {node_name} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in node {node_name}", error=str(e), exc_info=True)
            handle_graph_error(
                e, state if isinstance(state, dict) else {"state": state}
            )
            return state

    @wraps(node_func)
    def sync_wrapper(*args, **kwargs) -> dict[str, Any]:
        state, writer, instance = _extract_args(args, kwargs)
        logger = node_logger(node_name)
        logger.info(f"✨ Executing node: {node_name}")
        logger.debug(
            f"Input state keys: {list(state.keys()) if isinstance(state, dict) else 'non-dict'}"
        )

        try:
            # Run the node function
            if instance is not None and hasattr(instance, "__class__"):  # Method call
                result = (
                    node_func(instance, state, writer)
                    if writer is not None
                    else node_func(instance, state)
                )
            else:  # Function call
                result = (
                    node_func(state, writer=writer)
                    if writer is not None
                    else node_func(state)
                )

            result = _validate_and_process_result(result, node_name)
            logger.info(f"✅ Node {node_name} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in node {node_name}", error=str(e), exc_info=True)
            handle_graph_error(
                e, state if isinstance(state, dict) else {"state": state}
            )
            return state

    return async_wrapper if is_async else sync_wrapper


def safe_node(node_name: str) -> Callable:
    """Decorator to wrap LangGraph nodes with unified logging and error handling.

    This decorator provides:
    - Per-node logging with context
    - Error handling and state modification
    - Structured error information in state

    Args:
        node_name: Name of the node for logging and error context

    Returns:
        Decorator function that wraps the node function

    Example:
        @safe_node("observe")
        def observe_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger = node_logger("observe")
            logger.info("Starting observation phase")
            # Node implementation
            return state

        @safe_node("observe")
        async def observe_node(state: Dict[str, Any], writer=None) -> Dict[str, Any]:
            # Support for streaming writer parameter
            logger = node_logger("observe")
            logger.info("Starting observation phase")
            # Node implementation
            return state
    """

    def decorator(
        node_func: Callable[..., Any],
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        import inspect

        is_async = inspect.iscoroutinefunction(node_func)
        return _create_node_wrapper(node_func, node_name, is_async)

    return decorator
