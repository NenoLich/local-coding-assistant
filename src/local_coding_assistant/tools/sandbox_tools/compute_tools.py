"""Self-contained tools that can run in an isolated sandbox environment.

These tools have no external dependencies and can be safely executed in a sandbox.
"""

from typing import Literal


class MathTool:
    """A tool that performs basic arithmetic operations."""

    async def run(
        self,
        operation: Literal["add", "subtract", "multiply", "divide"],
        numbers: list[float],
    ) -> float:
        """Perform the requested arithmetic operation.

        Args:
            operation: The arithmetic operation to perform (add/subtract/multiply/divide)
            numbers: List of numbers to operate on (at least 2 required)

        Returns:
            The result of the arithmetic operation

        Raises:
            ValueError: If division by zero is attempted or invalid operation/input
        """
        if len(numbers) < 2:
            raise ValueError("At least 2 numbers are required")

        if operation == "add":
            return sum(numbers)

        if operation == "subtract":
            return numbers[0] - sum(numbers[1:])

        if operation == "multiply":
            result = 1
            for num in numbers:
                result *= num
            return result

        if operation == "divide":
            if 0 in numbers[1:]:
                raise ValueError("Cannot divide by zero")
            result = numbers[0]
            for num in numbers[1:]:
                result /= num
            return result

        raise ValueError(f"Unsupported operation: {operation}")
