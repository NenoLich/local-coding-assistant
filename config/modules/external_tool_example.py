"""Example external math tool implementation."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from local_coding_assistant.tools.tool_registry import register_tool
from local_coding_assistant.tools.types import ToolCategory, ToolPermission, ToolTag


@register_tool(
    name="math_operations",
    description="Perform basic arithmetic operations (add, subtract, multiply, divide)",
    category=ToolCategory.MATH,
    tags=[ToolTag.MATH, ToolTag.UTILITY],
    permissions=[ToolPermission.COMPUTE],
    is_async=False,
    supports_streaming=False,
)
class MathOperationsTool:
    """A tool that performs basic arithmetic operations."""

    class Input(BaseModel):
        """Input model for the math operations tool."""

        operation: Literal["add", "subtract", "multiply", "divide"] = Field(
            ..., description="The arithmetic operation to perform"
        )
        numbers: list[float] = Field(
            ...,
            description="List of numbers to perform the operation on (at least 2 required)",
            min_length=2,
        )

        @model_validator(mode="after")
        def validate_numbers(self) -> "MathOperationsTool.Input":
            if len(self.numbers) < 2:
                raise ValueError("At least 2 numbers are required")
            return self

    class Output(BaseModel):
        """Output model for the math operations tool."""

        result: float = Field(..., description="The result of the arithmetic operation")
        operation: str = Field(..., description="The operation that was performed")
        operands: list[float] = Field(
            ..., description="The numbers used in the operation"
        )

    def run(self, input_data: Input) -> Output:
        """Perform the requested arithmetic operation.

        Args:
            input_data: Input containing the operation and numbers

        Returns:
            The result of the arithmetic operation

        Raises:
            ValueError: If division by zero is attempted or invalid operation
        """
        operation = input_data.operation
        numbers = input_data.numbers

        if operation == "add":
            result = sum(numbers)
        elif operation == "subtract":
            result = numbers[0] - sum(numbers[1:])
        elif operation == "multiply":
            result = 1
            for num in numbers:
                result *= num
        elif operation == "divide":
            if 0 in numbers[1:]:
                raise ValueError("Cannot divide by zero")
            result = numbers[0]
            for num in numbers[1:]:
                result /= num
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        return self.Output(result=result, operation=operation, operands=numbers)
