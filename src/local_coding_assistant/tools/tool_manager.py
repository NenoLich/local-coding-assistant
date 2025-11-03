"""Enhanced tool management with proper error handling and logging."""

from __future__ import annotations

import inspect
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
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


@dataclass
class ToolValidationResult:
    """Named tuple for the validation result."""

    has_input_validation: bool
    has_output_validation: bool


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

    def _instantiate_tool(self, tool: Any) -> tuple[Any, str]:
        """Instantiate tool if it's a class and get its class name.

        Args:
            tool: Tool class or instance to instantiate.

        Returns:
            Tuple of (tool_instance, class_name)

        Raises:
            ToolRegistryError: If instantiation fails.
        """
        if isinstance(tool, type):
            try:
                return tool(), tool.__name__
            except Exception as e:
                raise ToolRegistryError(
                    f"Failed to instantiate tool class {tool.__name__}: {e}"
                ) from e
        return tool, tool.__class__.__name__

    def _get_tool_name(self, tool_instance: Any, class_name: str) -> str:
        """Extract and validate the tool name.

        Args:
            tool_instance: The tool instance.
            class_name: The tool's class name.

        Returns:
            The tool name.

        Raises:
            ToolRegistryError: If the name is invalid.
        """
        tool_name = (
            getattr(tool_instance, "name", None)
            or getattr(tool_instance, "__name__", None)
            or class_name
        )

        if not isinstance(tool_name, str):
            raise ToolRegistryError(
                f"Tool name must be a string, got {type(tool_name)}"
            )

        return tool_name

    def _check_naming_conflicts(self, tool_name: str) -> None:
        """Check for duplicate tool names.

        Args:
            tool_name: Name of the tool to check.

        Raises:
            ToolRegistryError: If a naming conflict is found.
        """
        if tool_name in self._by_name:
            existing_tool = self._by_name[tool_name]
            raise ToolRegistryError(
                f"Tool '{tool_name}' already registered as {existing_tool.__class__.__name__}"
            )

    def _validate_tool_contract(
        self, tool_instance: Any, tool_name: str
    ) -> ToolValidationResult:
        """Validate that the tool follows the expected contract.

        Args:
            tool_instance: The tool instance to validate.
            tool_name: Name of the tool for error messages.

        Returns:
            ToolValidationResult: Named tuple with validation results.

        Raises:
            ToolRegistryError: If the tool doesn't meet the contract requirements.
        """
        # Early exit for non-Tool callables
        if not isinstance(tool_instance, Tool):
            if not callable(tool_instance):
                raise ToolRegistryError(f"Tool '{tool_name}' is not callable")
            return ToolValidationResult(False, False)

        # Check for required 'run' method
        if not hasattr(tool_instance, "run") or not callable(tool_instance.run):
            raise ToolRegistryError(f"Tool '{tool_name}' missing required 'run' method")

        # Check if 'run' is implemented (not abstract)
        try:
            if "raise NotImplementedError" in inspect.getsource(tool_instance.run):
                raise ToolRegistryError(
                    f"Tool '{tool_name}' has not implemented the required 'run' method"
                )
        except (OSError, TypeError) as e:
            logger.warning(
                "Could not inspect source for tool '%s'. Skipping implementation check. Error: %s",
                tool_name,
                e,
            )

        # Check for Input/Output validation
        input_validation = hasattr(tool_instance, "Input") and isinstance(
            tool_instance.Input, type
        )
        output_validation = hasattr(tool_instance, "Output") and isinstance(
            tool_instance.Output, type
        )

        return ToolValidationResult(input_validation, output_validation)

    def _create_tool_info(
        self,
        tool_instance: Any,
        tool_name: str,
        class_name: str,
        category: str | None,
        input_validation: bool,
        output_validation: bool,
    ) -> None:
        """Create and store tool information.

        Args:
            tool_instance: The tool instance.
            tool_name: Name of the tool.
            class_name: Class name of the tool.
            category: Optional category for the tool.
            input_validation: Whether the tool has input validation.
            output_validation: Whether the tool has output validation.
        """
        tool_description = getattr(tool_instance, "description", None)

        tool_info = ToolInfo(
            name=tool_name,
            class_name=class_name,
            description=tool_description,
            category=category,
            has_input_validation=input_validation,
            has_output_validation=output_validation,
        )
        self._tool_info[tool_name] = tool_info
        logger.info(
            f"Registered tool '{tool_name}' ({class_name}) in category '{category or 'default'}'"
        )

    def register_tool(self, tool: Any, category: str | None = None) -> None:
        """Register a tool with enhanced validation and logging.

        Args:
            tool: Tool instance or class to register. If class, will be instantiated.
            category: Optional category for tool organization.

        Raises:
            ToolRegistryError: If registration fails due to validation or naming conflicts.
        """
        try:
            # Instantiate tool if it's a class
            tool_instance, class_name = self._instantiate_tool(tool)

            # Get and validate tool name
            tool_name = self._get_tool_name(tool_instance, class_name)

            # Check for naming conflicts
            self._check_naming_conflicts(tool_name)

            # Validate tool contract and get validation info
            validation_result = self._validate_tool_contract(tool_instance, tool_name)

            # Register the tool
            self._tools.append(tool_instance)
            self._by_name[tool_name] = tool_instance

            # Create and store tool information
            self._create_tool_info(
                tool_instance=tool_instance,
                tool_name=tool_name,
                class_name=class_name,
                category=category,
                input_validation=validation_result.has_input_validation,
                output_validation=validation_result.has_output_validation,
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
        start_time = time.time()
        self._execution_stats[tool_name] = self._execution_stats.get(tool_name, 0) + 1

        try:
            logger.debug(f"Executing tool '{tool_name}' with payload: {payload}")
            tool = self._get_tool(tool_name)
            input_model = self._prepare_input(tool, tool_name, payload)
            result = self._execute_tool(tool, tool_name, input_model)
            validated_result = self._validate_output(tool, tool_name, result)

            execution_time = (time.time() - start_time) * 1000
            logger.info(
                f"Tool '{tool_name}' executed successfully in {execution_time:.2f} ms"
            )
            return validated_result

        except ToolRegistryError:
            raise
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Unexpected error executing tool '{tool_name}' after {execution_time:.2f} ms: {e}"
            logger.exception(error_msg)
            raise ToolRegistryError(error_msg) from e

    def _get_tool(self, tool_name: str) -> Any:
        """Get tool by name with validation."""
        tool = self.get(tool_name)
        if tool is None:
            raise ToolRegistryError(f"Unknown tool: {tool_name}")
        return tool

    def _prepare_input(self, tool: Any, tool_name: str, payload: Any) -> Any:
        """Prepare and validate input for the tool."""
        if not (
            isinstance(tool, Tool) and hasattr(tool, "Input") and tool.Input is not None
        ):
            return payload

        try:
            if isinstance(payload, tool.Input):
                return payload
            if isinstance(payload, dict):
                return tool.Input.model_validate(payload)
            if hasattr(payload, "model_dump"):
                return tool.Input.model_validate(payload.model_dump())
            if hasattr(payload, "dict"):
                return tool.Input.model_validate(payload.dict())
            return tool.Input.model_validate_json(payload)
        except (ValidationError, ValueError, AttributeError) as e:
            raise ToolRegistryError(f"Invalid input for tool '{tool_name}': {e}") from e

    def _execute_tool(self, tool: Any, tool_name: str, input_model: Any) -> Any:
        """Execute the tool with the prepared input."""
        try:
            if isinstance(tool, Tool):
                return tool.run(input_model)
            if callable(tool):
                return tool(input_model)
            raise ToolRegistryError(f"Tool '{tool_name}' is not callable")
        except Exception as e:
            raise ToolRegistryError(f"Error executing tool '{tool_name}': {e}") from e

    def _validate_output(self, tool: Any, tool_name: str, result: Any) -> Any:
        """Validate and process the tool's output."""
        if not (
            isinstance(tool, Tool)
            and hasattr(tool, "Output")
            and tool.Output is not None
        ):
            return result

        try:
            if isinstance(result, tool.Output):
                return result.model_dump()

            # Handle callable results
            if callable(result):
                result = result()

            # Convert result to output model
            if isinstance(result, dict):
                output_model = tool.Output.model_validate(result)
            elif hasattr(result, "model_dump"):
                output_model = tool.Output.model_validate(result.model_dump())
            elif hasattr(result, "dict"):
                output_model = tool.Output.model_validate(result.dict())
            else:
                output_model = tool.Output.model_validate(dict(result))

            return output_model.model_dump()

        except (ValidationError, ValueError, TypeError) as e:
            raise ToolRegistryError(
                f"Invalid output from tool '{tool_name}': {e}"
            ) from e

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
