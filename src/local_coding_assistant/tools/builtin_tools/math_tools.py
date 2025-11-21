"""Mathematical tools for the local coding assistant."""

import asyncio

from pydantic import BaseModel, Field

from local_coding_assistant.tools.tool_registry import register_tool
from local_coding_assistant.tools.types import ToolCategory, ToolTag


@register_tool(
    name="sum",
    description="Adds two or more numbers together",
    category=ToolCategory.MATH,
    tags=[ToolTag.MATH, ToolTag.UTILITY],
    permissions=[],
    is_async=False,
    supports_streaming=False,
)
class SumTool:
    """A tool that adds numbers together."""

    class Input(BaseModel):
        """Input model for the sum tool."""

        numbers: list[float] = Field(..., description="List of numbers to add together")

    class Output(BaseModel):
        """Output model for the sum tool."""

        result: float = Field(..., description="The sum of the numbers")

    def run(self, input_data: Input) -> Output:
        """Add the numbers together.

        Args:
            input_data: Input containing numbers to add

        Returns:
            The sum of the numbers
        """
        return self.Output(result=sum(input_data.numbers))


@register_tool(
    name="sum_async",
    description="Adds two or more numbers together asynchronously",
    category=ToolCategory.MATH,
    tags=[ToolTag.MATH, ToolTag.UTILITY, ToolTag.AI],
    permissions=[],
    is_async=True,
    supports_streaming=False,
)
class AsyncSumTool:
    """An async tool that adds numbers together."""

    class Input(BaseModel):
        """Input model for the async sum tool."""

        numbers: list[float] = Field(..., description="List of numbers to add together")

    class Output(BaseModel):
        """Output model for the async sum tool."""

        result: float = Field(..., description="The sum of the numbers")

    async def run(self, input_data: Input) -> Output:
        """Add the numbers together asynchronously.

        Args:
            input_data: Input containing numbers to add

        Returns:
            The sum of the numbers

        Note:
            This is an async version for testing purposes. In a real-world scenario,
            you would typically use this for I/O-bound operations.
        """
        # Simulate some async operation
        await asyncio.sleep(0.1)
        return self.Output(result=sum(input_data.numbers))


@register_tool(
    name="multiply",
    description="Multiplies two or more numbers together",
    category=ToolCategory.MATH,
    tags=[ToolTag.MATH, ToolTag.UTILITY],
    permissions=[],
    is_async=False,
    supports_streaming=False,
)
class MultiplyTool:
    """A tool that multiplies numbers together."""

    class Input(BaseModel):
        """Input model for the multiply tool."""

        numbers: list[float] = Field(
            ..., description="List of numbers to multiply together"
        )

    class Output(BaseModel):
        """Output model for the multiply tool."""

        result: float = Field(..., description="The product of the numbers")

    def run(self, input_data: Input) -> Output:
        """Multiply the numbers together.

        Args:
            input_data: Input containing numbers to multiply

        Returns:
            The product of the numbers
        """
        result = 1
        for num in input_data.numbers:
            result *= num
        return self.Output(result=result)
