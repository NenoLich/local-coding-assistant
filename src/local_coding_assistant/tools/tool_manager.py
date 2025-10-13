"""Enhanced tool management with proper error handling and logging."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from local_coding_assistant.core.exceptions import ToolRegistryError
from local_coding_assistant.tools.base import Tool
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("tools.tool_manager")


class ToolRegistration(BaseModel):
    name: str = Field(..., description="Unique name for the tool")
    tool_class: type = Field(..., description="Tool class to register")
    description: str | None = Field(default=None, description="Tool description")
    category: str | None = Field(
        default=None, description="Tool category for organization"
    )


class ToolExecutionRequest(BaseModel):
    """Model for tool execution requests."""

    tool_name: str = Field(..., description="Name of the tool to execute")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Input payload for the tool"
    )


class ToolExecutionResponse(BaseModel):
    """Model for tool execution responses."""

    tool_name: str = Field(..., description="Name of the executed tool")
    success: bool = Field(..., description="Whether execution was successful")
    result: dict[str, Any] | None = Field(
        default=None, description="Tool execution result"
    )
    error_message: str | None = Field(
        default=None, description="Error message if execution failed"
    )
    execution_time_ms: float | None = Field(
        default=None, description="Execution time in milliseconds"
    )


class ToolInfo(BaseModel):
    """Model for tool information."""

    name: str = Field(..., description="Tool name")
    class_name: str = Field(..., description="Tool class name")
    description: str | None = Field(default=None, description="Tool description")
    category: str | None = Field(default=None, description="Tool category")
    has_input_validation: bool = Field(
        default=False, description="Whether tool has input validation"
    )
    has_output_validation: bool = Field(
        default=False, description="Whether tool has output validation"
    )


class ToolManager(Iterable[Any]):
    """Enhanced tool manager with comprehensive logging, error handling, and validation.

    This class provides an improved replacement for ToolRegistry with:
    - Proper error handling using ToolRegistryError
    - Comprehensive logging for debugging and monitoring
    - Pydantic models for type safety
    - Enhanced tool registration and execution features
    - Backward compatibility with existing ToolRegistry interface
    """

    def __init__(self) -> None:
        """Initialize the tool manager."""
        self._tools: list[Any] = []
        self._by_name: dict[str, Any] = {}
        self._tool_info: dict[str, ToolInfo] = {}
        self._execution_stats: dict[str, int] = {}

        logger.info("ToolManager initialized")

    def register_tool(self, tool: Any, category: str | None = None) -> None:
        """Register a tool with enhanced validation and logging.

        Args:
            tool: Tool instance or class to register. If class, will be instantiated.
            category: Optional category for tool organization.

        Raises:
            ToolRegistryError: If registration fails due to validation or naming conflicts.
        """
        try:
            # Handle both tool instances and classes
            if isinstance(tool, type):
                # It's a class, try to instantiate it
                try:
                    tool_instance = tool()
                    tool_class_name = tool.__name__
                except Exception as e:
                    raise ToolRegistryError(
                        f"Failed to instantiate tool class {tool.__name__}: {e}"
                    ) from e
            else:
                # It's already an instance
                tool_instance = tool
                tool_class_name = tool.__class__.__name__

            # Extract tool information
            tool_name = getattr(tool_instance, "name", None)
            if tool_name is None:
                # Try alternative name sources
                tool_name = getattr(tool_instance, "__name__", None) or tool_class_name

            if not isinstance(tool_name, str):
                raise ToolRegistryError(
                    f"Tool name must be a string, got {type(tool_name)}"
                )

            tool_description = getattr(tool_instance, "description", None)

            # Check for naming conflicts
            if tool_name in self._by_name:
                existing_tool = self._by_name[tool_name]
                raise ToolRegistryError(
                    f"Tool '{tool_name}' already registered as {existing_tool.__class__.__name__}"
                )

            # Validate tool structure if it follows Tool contract
            input_validation = False
            output_validation = False

            if isinstance(tool_instance, Tool):
                if hasattr(tool_instance, "Input") and tool_instance.Input is not None:
                    input_validation = True
                if (
                    hasattr(tool_instance, "Output")
                    and tool_instance.Output is not None
                ):
                    output_validation = True

                # Additional validation for Tool contract compliance
                if not hasattr(tool_instance, "run"):
                    raise ToolRegistryError(
                        f"Tool '{tool_name}' missing required 'run' method"
                    )

                # Check if run method is properly implemented (not just raising NotImplementedError)
                try:
                    # Try to call run with a dummy payload to see if it's properly implemented
                    if (
                        hasattr(tool_instance, "Input")
                        and tool_instance.Input is not None
                    ):
                        # Create a dummy input to test the run method
                        dummy_input = tool_instance.Input()
                        tool_instance.run(dummy_input)
                        # If run returns without error, it's properly implemented
                    else:
                        # No input validation, just try calling run with empty dict
                        tool_instance.run({})
                except NotImplementedError as e:
                    raise ToolRegistryError(
                        f"Tool '{tool_name}' missing required 'run' method implementation"
                    ) from e
                except Exception:
                    # Other exceptions are OK - the method exists and is callable
                    pass

            # Validate that the tool is callable (for both Tool and non-Tool objects)
            if isinstance(tool_instance, Tool):
                if not hasattr(tool_instance, "run") or not callable(tool_instance.run):
                    raise ToolRegistryError(
                        f"Tool '{tool_name}' missing required 'run' method"
                    )
            elif not callable(tool_instance):
                raise ToolRegistryError(f"Tool '{tool_name}' is not callable")

            # Register the tool
            self._tools.append(tool_instance)
            self._by_name[tool_name] = tool_instance

            # Store tool information
            tool_info = ToolInfo(
                name=tool_name,
                class_name=tool_class_name,
                description=tool_description,
                category=category,
                has_input_validation=input_validation,
                has_output_validation=output_validation,
            )
            self._tool_info[tool_name] = tool_info

            logger.info(
                f"Registered tool '{tool_name}' ({tool_class_name}) in category '{category or 'default'}'"
            )

        except ToolRegistryError as e:
            logger.error(f"Tool registration failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during tool registration: {e}")
            raise ToolRegistryError(f"Tool registration failed: {e}") from e

    def run_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool with enhanced error handling and performance tracking.

        Args:
            tool_name: Name of the tool to execute.
            payload: Input payload for the tool.

        Returns:
            Tool execution result.

        Raises:
            ToolRegistryError: If tool execution fails.
        """
        import time

        start_time = time.time()
        self._execution_stats[tool_name] = self._execution_stats.get(tool_name, 0) + 1

        try:
            logger.debug(f"Executing tool '{tool_name}' with payload: {payload}")
            tool = self.get(tool_name)
            if tool is None:
                raise ToolRegistryError(f"Unknown tool: {tool_name}")

            # If tool adheres to Tool contract, validate input
            if (
                isinstance(tool, Tool)
                and hasattr(tool, "Input")
                and tool.Input is not None
            ):
                try:
                    input_model = tool.Input(**payload)
                    logger.debug(f"Input validation passed for tool '{tool_name}'")
                except ValidationError as e:
                    error_msg = f"Invalid input for tool '{tool_name}': {e}"
                    logger.error(error_msg)
                    raise ToolRegistryError(error_msg) from e
            else:
                input_model = payload

            if isinstance(tool, Tool):
                result = tool.run(input_model)
            elif callable(tool):
                result = tool(input_model)
            else:
                raise ToolRegistryError(f"Tool '{tool_name}' is not callable")

            # If tool has output validation, validate the result
            if (
                isinstance(tool, Tool)
                and hasattr(tool, "Output")
                and tool.Output is not None
            ):
                try:
                    # Check if result is already the expected Output type
                    if isinstance(result, tool.Output):
                        # Result is already a Pydantic model, validate it and convert to dict
                        try:
                            validated_result = result.model_dump()
                        except ValidationError as e:
                            # The model contains invalid data, this shouldn't happen but handle it
                            raise ValidationError.from_exception_data(
                                title="Output",
                                line_errors=[
                                    {
                                        "type": "model_validation_error",
                                        "loc": ("result",),
                                        "input": result,
                                    }
                                ],
                            ) from e
                    else:
                        # Result is a dict, create Output model from it
                        output_model = tool.Output(**result)
                        validated_result = output_model.model_dump()
                    logger.debug(f"Output validation passed for tool '{tool_name}'")
                except ValidationError as e:
                    error_msg = f"Invalid output from tool '{tool_name}': {e}"
                    logger.error(error_msg)
                    raise ToolRegistryError(error_msg) from e
            else:
                validated_result = result

            execution_time = (time.time() - start_time) * 1000
            logger.info(
                f"Tool '{tool_name}' executed successfully in {execution_time:.2f} ms"
            )

            return validated_result

        except ToolRegistryError as e:
            logger.error(f"Tool execution failed: {e}")
            raise
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Tool '{tool_name}' execution failed after {execution_time:.2f} ms: {e}"
            logger.error(error_msg)
            raise ToolRegistryError(error_msg) from e

    def get_tool_info(self, tool_name: str) -> ToolInfo | None:
        """Get detailed information about a registered tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            ToolInfo object with tool details, or None if not found.
        """
        return self._tool_info.get(tool_name)

    def list_tools(self, category: str | None = None) -> list[ToolInfo]:
        """List all registered tools, optionally filtered by category.

        Args:
            category: Optional category filter.

        Returns:
            List of ToolInfo objects.
        """
        if category is None:
            return list(self._tool_info.values())

        return [info for info in self._tool_info.values() if info.category == category]

    def get_execution_stats(self) -> dict[str, int]:
        """Get execution statistics for all tools.

        Returns:
            Dictionary mapping tool names to execution counts.
        """
        return self._execution_stats.copy()

    # Backward compatibility methods (matching ToolRegistry interface)
    def register(self, tool: Any) -> None:
        """Legacy method for backward compatibility."""
        self.register_tool(tool)

    def get(self, name: str) -> Any | None:
        """Get a tool by name."""
        return self._by_name.get(name)

    def invoke(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Legacy method for backward compatibility."""
        return self.run_tool(name, payload)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over registered tools."""
        return iter(self._tools)

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)
