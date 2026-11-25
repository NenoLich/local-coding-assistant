"""Small Python file for benchmarking."""


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


class SimpleClass:
    """A simple class for testing."""

    def __init__(self, value: int):
        self.value = value

    def increment(self) -> None:
        """Increment the value."""
        self.value += 1

    def get_value(self) -> int:
        """Get the current value."""
        return self.value


def main() -> None:
    """Main function."""
    obj = SimpleClass(10)
    obj.increment()
    result = add(obj.get_value(), 5)
    print(result)


if __name__ == "__main__":
    main()
