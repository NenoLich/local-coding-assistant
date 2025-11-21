from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TypeVar

from pydantic import BaseModel

from local_coding_assistant.utils.logging import get_logger

logger = get_logger("tools.tool_manager")

T = TypeVar("T", bound="Tool")


class Tool(ABC):
    """Base class for all tools.

    To create a new tool, inherit from this class and implement the run method.
    Define Input and Output as inner classes that inherit from pydantic.BaseModel
    to specify the expected input and output schemas.
    """

    class Input(BaseModel):
        """Base input model for tool execution. Override this in your tool class."""

        pass

    class Output(BaseModel):
        """Base output model for tool execution. Override this in your tool class."""

        pass

    def __init_subclass__(cls, **kwargs):
        """Validate that subclasses properly implement the required methods."""
        super().__init_subclass__(**kwargs)

        # Skip validation for abstract classes
        if cls.__dict__.get("__abstractmethods__", None):
            return

        # Check if run method is properly implemented
        if "run" not in cls.__dict__:
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} "
                f"without implementing the 'run' method"
            )

        # Check if run method is the abstract one
        run_method = cls.run
        if getattr(run_method, "__isabstractmethod__", False):
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} "
                f"with abstract method 'run'"
            )

        # Check if Input and Output are properly defined
        if not (hasattr(cls, "Input") and issubclass(cls.Input, BaseModel)):
            raise TypeError(
                f"Tool '{cls.__name__}' must define an inner 'Input' class "
                f"that inherits from BaseModel"
            )

        if not (hasattr(cls, "Output") and issubclass(cls.Output, BaseModel)):
            raise TypeError(
                f"Tool '{cls.__name__}' must define an inner 'Output' class "
                f"that inherits from BaseModel"
            )

    @abstractmethod
    def run(self, input_data: Input) -> Output:
        """Execute the tool with the given input.

        Args:
            input_data: Validated input data matching the Input model

        Returns:
            Output data matching the Output model

        Raises:
            ToolExecutionError: If tool execution fails
        """
        pass

    async def stream(self, input_data: Input) -> AsyncIterator[Output | dict]:
        """Stream results from the tool (optional).

        This method should be overridden by tools that support streaming.

        Args:
            input_data: Validated input data matching the Input model

        Yields:
            Chunks of output data, either as Output model instances or dictionaries

        Raises:
            NotImplementedError: If the tool doesn't support streaming
            ToolExecutionError: If streaming fails

        Example:
            class MyStreamingTool(Tool):
                async def stream(self, input_data):
                    for item in some_data:
                        # Process item asynchronously
                        result = await process_async(item)
                        yield result
        """
        # Default implementation that wraps the run method for backward compatibility
        result = self.run(input_data)

        if inspect.isawaitable(result):
            result = await result

        # Handle different result types safely
        try:
            # Check if the result has model_dump method (Pydantic v2+)
            if hasattr(result, "model_dump") and callable(result.model_dump):
                yield result.model_dump()
            else:
                yield result
        except Exception as e:
            # If anything goes wrong during serialization, yield the original result
            logger.warning("Failed to serialize tool result: %s", str(e))
            yield result

    @classmethod
    def validate_input(cls, input_data: dict) -> bool:
        """Validate input data against the tool's Input model.

        Args:
            input_data: Input data to validate

        Returns:
            bool: True if input is valid, False otherwise
        """
        try:
            cls.Input.model_validate(input_data)
            return True
        except Exception:
            return False

    @classmethod
    def get_input_schema(cls) -> dict:
        """Get the JSON Schema for the tool's input."""
        return cls.Input.model_json_schema()

    @classmethod
    def get_output_schema(cls) -> dict:
        """Get the JSON Schema for the tool's output."""
        return cls.Output.model_json_schema()


class ToolExecutionError(Exception):
    """Exception raised when a tool fails to execute."""

    def __init__(self, message: str, tool_name: str | None = None):
        self.message = message
        super().__init__(
            f"Tool '{tool_name}' failed: {message}"
            if tool_name
            else f"Tool failed: {message}"
        )
