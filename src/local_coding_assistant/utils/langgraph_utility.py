"""LangGraph utilities for error handling, logging, and node decoration."""

import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from local_coding_assistant.core.exceptions import AgentError, ToolRegistryError
from local_coding_assistant.utils.logging import get_logger

log = get_logger("agent")


def node_logger(node_name: str) -> logging.Logger:
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
        # Check if function supports writer parameter (streaming)
        import inspect

        is_async = inspect.iscoroutinefunction(node_func)

        @wraps(node_func)
        async def async_wrapper(*args, **kwargs) -> dict[str, Any]:
            # Extract state from arguments - handle both standalone functions and methods
            if args and hasattr(args[0], "__self__"):  # It's a bound method
                state = kwargs.get("state") or (args[1] if len(args) > 1 else {})
                writer = kwargs.get("writer")
                instance = args[0]
            else:  # It's a standalone function
                state = kwargs.get("state") or (args[0] if args else {})
                writer = kwargs.get("writer")
                instance = None

            logger = node_logger(node_name)
            logger.info(f"✨ Executing node: {node_name}")
            logger.debug(
                f"Input state keys: {list(state.keys()) if isinstance(state, dict) else 'non-dict'}"
            )

            try:
                # Run the node function
                if instance and hasattr(instance, "__class__"):  # Method call
                    if writer is not None:
                        result = await node_func(instance, state, writer)
                    else:
                        result = await node_func(instance, state)
                else:  # Function call
                    if writer is not None:
                        result = await node_func(state, writer=writer)
                    else:
                        result = await node_func(state)

                # Validate return type - handle both dict and AgentState
                if hasattr(
                    result, "model_dump"
                ):  # It's a Pydantic model like AgentState
                    result = result.model_dump()
                elif not isinstance(result, dict):
                    raise TypeError(
                        f"Node {node_name} must return a dict or Pydantic model, got {type(result)}"
                    )

                logger.info(f"Node {node_name} completed successfully")
                return result

            except Exception as e:
                logger.error(f"Error in node {node_name}: {e}")
                handle_graph_error(
                    e, state if isinstance(state, dict) else {"state": state}
                )
                return state

        @wraps(node_func)
        def sync_wrapper(*args, **kwargs) -> dict[str, Any]:
            # Extract state from arguments - handle both standalone functions and methods
            if args and hasattr(args[0], "__self__"):  # It's a bound method
                state = kwargs.get("state") or (args[1] if len(args) > 1 else {})
                writer = kwargs.get("writer")
                instance = args[0]
            else:  # It's a standalone function
                state = kwargs.get("state") or (args[0] if args else {})
                writer = kwargs.get("writer")
                instance = None

            logger = node_logger(node_name)
            logger.info(f"✨ Executing node: {node_name}")
            logger.debug(
                f"Input state keys: {list(state.keys()) if isinstance(state, dict) else 'non-dict'}"
            )

            try:
                # Run the node function
                if instance and hasattr(instance, "__class__"):  # Method call
                    if writer is not None:
                        result = node_func(instance, state, writer)
                    else:
                        result = node_func(instance, state)
                else:  # Function call
                    if writer is not None:
                        result = node_func(state, writer=writer)
                    else:
                        result = node_func(state)

                # Validate return type - handle both dict and AgentState
                if hasattr(
                    result, "model_dump"
                ):  # It's a Pydantic model like AgentState
                    result = result.model_dump()
                elif not isinstance(result, dict):
                    raise TypeError(
                        f"Node {node_name} must return a dict or Pydantic model, got {type(result)}"
                    )

                logger.info(f"✅ Node {node_name} completed successfully")
                return result

            except Exception as e:
                logger.error(f"Error in node {node_name}: {e}")
                handle_graph_error(
                    e, state if isinstance(state, dict) else {"state": state}
                )
                return state

        return async_wrapper if is_async else sync_wrapper

    return decorator
