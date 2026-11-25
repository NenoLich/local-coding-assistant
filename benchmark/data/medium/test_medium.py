"""Medium Python file for benchmarking with more complex structure."""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class OperationType(Enum):
    """Enumeration of operation types."""
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


@dataclass
class OperationResult:
    """Result of a calculation operation."""
    value: float
    operation: OperationType
    success: bool
    error_message: Optional[str] = None


class Calculator:
    """A more complex calculator with history."""

    def __init__(self):
        self.history: List[OperationResult] = []
        self.current_value: float = 0.0

    def add(self, x: float) -> OperationResult:
        """Add a value to the current value."""
        try:
            self.current_value += x
            result = OperationResult(
                value=self.current_value,
                operation=OperationType.ADD,
                success=True
            )
            self.history.append(result)
            return result
        except Exception as e:
            return OperationResult(
                value=0.0,
                operation=OperationType.ADD,
                success=False,
                error_message=str(e)
            )

    def subtract(self, x: float) -> OperationResult:
        """Subtract a value from the current value."""
        try:
            self.current_value -= x
            result = OperationResult(
                value=self.current_value,
                operation=OperationType.SUBTRACT,
                success=True
            )
            self.history.append(result)
            return result
        except Exception as e:
            return OperationResult(
                value=0.0,
                operation=OperationType.SUBTRACT,
                success=False,
                error_message=str(e)
            )

    def multiply(self, x: float) -> OperationResult:
        """Multiply the current value by a factor."""
        try:
            self.current_value *= x
            result = OperationResult(
                value=self.current_value,
                operation=OperationType.MULTIPLY,
                success=True
            )
            self.history.append(result)
            return result
        except Exception as e:
            return OperationResult(
                value=0.0,
                operation=OperationType.MULTIPLY,
                success=False,
                error_message=str(e)
            )

    def divide(self, x: float) -> OperationResult:
        """Divide the current value by a divisor."""
        try:
            if x == 0:
                raise ValueError("Cannot divide by zero")
            self.current_value /= x
            result = OperationResult(
                value=self.current_value,
                operation=OperationType.DIVIDE,
                success=True
            )
            self.history.append(result)
            return result
        except Exception as e:
            return OperationResult(
                value=0.0,
                operation=OperationType.DIVIDE,
                success=False,
                error_message=str(e)
            )

    def get_history(self) -> List[OperationResult]:
        """Get the operation history."""
        return self.history.copy()

    def clear_history(self) -> None:
        """Clear the operation history."""
        self.history.clear()

    def reset(self) -> None:
        """Reset the calculator to initial state."""
        self.current_value = 0.0
        self.clear_history()


class ScientificCalculator(Calculator):
    """A scientific calculator with additional operations."""

    def power(self, exponent: float) -> OperationResult:
        """Raise current value to a power."""
        try:
            self.current_value **= exponent
            result = OperationResult(
                value=self.current_value,
                operation=OperationType.MULTIPLY,
                success=True
            )
            self.history.append(result)
            return result
        except Exception as e:
            return OperationResult(
                value=0.0,
                operation=OperationType.MULTIPLY,
                success=False,
                error_message=str(e)
            )

    def square_root(self) -> OperationResult:
        """Calculate square root of current value."""
        try:
            if self.current_value < 0:
                raise ValueError("Cannot calculate square root of negative number")
            self.current_value **= 0.5
            result = OperationResult(
                value=self.current_value,
                operation=OperationType.MULTIPLY,
                success=True
            )
            self.history.append(result)
            return result
        except Exception as e:
            return OperationResult(
                value=0.0,
                operation=OperationType.MULTIPLY,
                success=False,
                error_message=str(e)
            )


def perform_calculation_sequence(calc: Calculator) -> float:
    """Perform a sequence of calculations."""
    calc.add(100)
    calc.subtract(25)
    calc.multiply(2)
    calc.divide(3)
    return calc.current_value


def main() -> None:
    """Main function to demonstrate calculator usage."""
    calc = ScientificCalculator()
    
    # Perform basic operations
    calc.add(50)
    calc.subtract(10)
    calc.multiply(3)
    calc.divide(4)
    
    # Perform scientific operations
    calc.power(2)
    calc.square_root()
    
    # Perform calculation sequence
    result = perform_calculation_sequence(calc)
    
    print(f"Final result: {result}")
    print(f"History length: {len(calc.get_history())}")


if __name__ == "__main__":
    main()
