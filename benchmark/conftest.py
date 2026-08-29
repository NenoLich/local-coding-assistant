"""Shared fixtures and utilities for benchmark tests."""

from pathlib import Path

import pytest

from local_coding_assistant.repository.ast_parser import (
    ASTParser,
    CallRelationshipExtractor,
)
from local_coding_assistant.repository.database import RepositoryDatabase


@pytest.fixture
def benchmark_data_dir() -> Path:
    """Get the benchmark data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def ast_parser() -> ASTParser:
    """Get an AST parser instance."""
    return ASTParser()


@pytest.fixture
def call_relationship_extractor() -> CallRelationshipExtractor:
    """Get a call relationship extractor instance."""
    return CallRelationshipExtractor()


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Get a temporary database path for benchmarking."""
    return tmp_path / "benchmark.db"


@pytest.fixture
def repository_db(temp_db_path: Path) -> RepositoryDatabase:
    """Get a repository database instance for benchmarking."""
    db = RepositoryDatabase(temp_db_path)
    return db


@pytest.fixture
def sample_python_code() -> str:
    """Sample Python code for benchmarking."""
    return '''
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

def calculate_product(a: int, b: int) -> int:
    """Calculate the product of two numbers."""
    return a * b

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.value = 0
    
    def add(self, x: int) -> None:
        """Add a value to the calculator."""
        self.value += x
    
    def subtract(self, x: int) -> None:
        """Subtract a value from the calculator."""
        self.value -= x
    
    def get_value(self) -> int:
        """Get the current value."""
        return self.value

def main():
    """Main function."""
    calc = Calculator()
    calc.add(10)
    calc.subtract(5)
    result = calculate_sum(calc.get_value(), 5)
    print(result)

if __name__ == "__main__":
    main()
'''


@pytest.fixture
def sample_javascript_code() -> str:
    """Sample JavaScript code for benchmarking."""
    return """
function calculateSum(a, b) {
    return a + b;
}

function calculateProduct(a, b) {
    return a * b;
}

class Calculator {
    constructor() {
        this.value = 0;
    }
    
    add(x) {
        this.value += x;
    }
    
    subtract(x) {
        this.value -= x;
    }
    
    getValue() {
        return this.value;
    }
}

function main() {
    const calc = new Calculator();
    calc.add(10);
    calc.subtract(5);
    const result = calculateSum(calc.getValue(), 5);
    console.log(result);
}

main();
"""


@pytest.fixture
def sample_go_code() -> str:
    """Sample Go code for benchmarking."""
    return """
package main

import "fmt"

func calculateSum(a int, b int) int {
    return a + b
}

func calculateProduct(a int, b int) int {
    return a * b
}

type Calculator struct {
    value int
}

func (c *Calculator) add(x int) {
    c.value += x
}

func (c *Calculator) subtract(x int) {
    c.value -= x
}

func (c *Calculator) getValue() int {
    return c.value
}

func main() {
    calc := &Calculator{value: 0}
    calc.add(10)
    calc.subtract(5)
    result := calculateSum(calc.getValue(), 5)
    fmt.Println(result)
}
"""


def load_code_from_file(file_path: Path) -> str:
    """Load code from a file for benchmarking."""
    return file_path.read_text(encoding="utf-8")
