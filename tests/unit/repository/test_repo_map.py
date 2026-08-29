"""Unit tests for repository map generation."""

import tempfile
from pathlib import Path

import pytest

from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.models import FileMetadata
from local_coding_assistant.repository.repo_map import (
    MapFormatter,
    RepoMapBuilder,
    RepoMapData,
)

GOLDEN_DIR = Path(__file__).parent / "golden"


def _read_golden(name: str) -> str:
    return (GOLDEN_DIR / name).read_text()


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def database(temp_db_path):
    """Create a test database instance."""
    return RepositoryDatabase(temp_db_path)


@pytest.fixture
def repo_map_builder(database):
    """Create a repo map builder instance."""
    return RepoMapBuilder(database, max_symbols=10, include_docstrings=True)


@pytest.fixture
def sample_file_metadata():
    """Create sample file metadata."""
    return FileMetadata(
        path="test.py",
        language="python",
        last_modified=1234567890,
        last_indexed=1234567890,
        hash="abc123",
        size_bytes=1000,
        content="def python_func() -> None: pass",
    )


@pytest.fixture
def sample_file_metadata_js():
    """Create sample file metadata for JavaScript."""
    return FileMetadata(
        path="test.js",
        language="javascript",
        last_modified=1234567890,
        last_indexed=1234567890,
        hash="def456",
        size_bytes=500,
        content="function jsFunc() {}",
    )


class TestRepoMapBuilder:
    """Tests for RepoMapBuilder."""

    def test_build_empty_repo_map(self, repo_map_builder):
        """Test building a repo map with no symbols."""
        repo_map = repo_map_builder.build(project_name="TestProject")
        assert repo_map.project_name == "TestProject"
        assert repo_map.grouped_symbols == {}

    def test_build_repo_map_with_symbols(self, repo_map_builder, sample_file_metadata):
        """Test building a repo map with symbols."""
        # Add some symbols to the database
        from local_coding_assistant.repository.models import ASTNode, CallRelationship
        from local_coding_assistant.repository.symbol_search import SymbolIndexer

        indexer = SymbolIndexer(repo_map_builder.database)

        symbols = [
            ASTNode(
                type="function_definition",
                name="test_function",
                line_number=10,
                end_line_number=20,
                column=0,
                metadata={"symbol_type": "function", "parent_scope": None},
                signature="def test_function(x: int) -> bool",
            ),
            ASTNode(
                type="class_definition",
                name="TestClass",
                line_number=5,
                end_line_number=30,
                column=0,
                metadata={"symbol_type": "class", "parent_scope": None},
                signature=None,
            ),
        ]

        content = """
class TestClass:
    pass

def test_function(x: int) -> bool:
    return True
"""

        # Add call relationships
        call_relationships = [
            CallRelationship(
                caller_name="test_function",
                caller_line=10,
                callee_name="helper_func",
                callee_line=20,
                relationship_type="call",
                language="python",
                file_path="test_path",
            ),
        ]

        indexer.index_file(
            symbols=symbols,
            file_metadata=sample_file_metadata,
            call_relationships=call_relationships,
        )

        repo_map = repo_map_builder.build(project_name="TestProject")
        assert repo_map.project_name == "TestProject"
        assert "test.py" in repo_map.grouped_symbols
        assert len(repo_map.grouped_symbols["test.py"]) == 2
        assert repo_map.grouped_symbols["test.py"][0]["name"] == "test_function"
        assert repo_map.grouped_symbols["test.py"][0]["symbol_type"] == "function"
        assert repo_map.grouped_symbols["test.py"][0]["line_number"] == 10
        assert repo_map.grouped_symbols["test.py"][0]["language"] == "python"
        assert (
            repo_map.grouped_symbols["test.py"][0]["signature"]
            == "def test_function(x: int) -> bool"
        )
        assert repo_map.grouped_symbols["test.py"][1]["name"] == "TestClass"
        assert repo_map.grouped_symbols["test.py"][1]["symbol_type"] == "class"
        assert repo_map.grouped_symbols["test.py"][1]["line_number"] == 5
        assert repo_map.grouped_symbols["test.py"][1]["language"] == "python"
        assert repo_map.grouped_symbols["test.py"][1]["signature"] is None

    def test_build_repo_map_with_multiple_languages(
        self, repo_map_builder, sample_file_metadata, sample_file_metadata_js
    ):
        """Test building a repo map with multiple languages."""
        from local_coding_assistant.repository.ast_parser import ASTNode
        from local_coding_assistant.repository.symbol_search import SymbolIndexer

        indexer = SymbolIndexer(repo_map_builder.database)

        # Add Python file
        py_symbols = [
            ASTNode(
                type="function_definition",
                name="python_func",
                line_number=10,
                end_line_number=20,
                column=0,
                metadata={"symbol_type": "function", "parent_scope": None},
                signature="def python_func() -> None",
            ),
        ]

        indexer.index_file(
            symbols=py_symbols,
            file_metadata=sample_file_metadata,
        )

        # Add JavaScript file
        js_symbols = [
            ASTNode(
                type="function_declaration",
                name="jsFunc",
                line_number=5,
                end_line_number=10,
                column=0,
                metadata={"symbol_type": "function", "parent_scope": None},
                signature="function jsFunc()",
            ),
        ]

        indexer.index_file(
            symbols=js_symbols,
            file_metadata=sample_file_metadata_js,
        )

        repo_map = repo_map_builder.build(project_name="TestProject")
        assert repo_map.language_distribution == {"python": 50.0, "javascript": 50.0}

    def test_filter_symbols_by_type(self, repo_map_builder):
        """Test filtering symbols by type."""
        symbols = [
            {
                "name": "func1",
                "symbol_type": "function",
                "line_number": 10,
                "file_path": "test.py",
            },
            {
                "name": "class1",
                "symbol_type": "class",
                "line_number": 5,
                "file_path": "test.py",
            },
            {
                "name": "func2",
                "symbol_type": "function",
                "line_number": 15,
                "file_path": "test.py",
            },
        ]

        repo_map_builder.included_symbol_types = {"function"}

        filtered = repo_map_builder._filter_symbols_by_type(symbols)
        assert len(filtered) == 2
        assert all(s["symbol_type"] == "function" for s in filtered)

    def test_select_top_symbols(self, repo_map_builder):
        """Test selecting top N symbols by rank."""
        symbols = [
            {"name": "symbol1", "rank": 0.1},
            {"name": "symbol2", "rank": 0.5},
            {"name": "symbol3", "rank": 0.9},
            {"name": "symbol4", "rank": 0.3},
            {"name": "symbol5", "rank": 0.7},
        ]

        repo_map_builder.max_symbols = 3

        top_symbols = repo_map_builder._select_top_symbols(symbols)
        assert len(top_symbols) == 3
        assert top_symbols[0]["name"] == "symbol3"  # Highest rank
        assert top_symbols[1]["name"] == "symbol5"
        assert top_symbols[2]["name"] == "symbol2"

    def test_group_symbols_by_file(self, repo_map_builder):
        """Test grouping symbols by file path."""
        symbols = [
            {"name": "func1", "file_path": "file1.py"},
            {"name": "func2", "file_path": "file2.py"},
            {"name": "func3", "file_path": "file1.py"},
        ]

        grouped = repo_map_builder._group_symbols_by_file(symbols)
        assert len(grouped) == 2
        assert len(grouped["file1.py"]) == 2
        assert len(grouped["file2.py"]) == 1


class TestMapFormatter:
    """Tests for MapFormatter."""

    def test_format_empty_map(self):
        """Test formatting an empty map."""
        formatter = MapFormatter()

        repo_map_data = RepoMapData(
            project_name="TestProject",
            language_distribution={},
            total_symbols=0,
            grouped_symbols={},
        )
        result = formatter.format(repo_map_data)
        assert result == ""

    def test_format_map_with_symbols(self):
        """Test formatting a map with symbols."""
        formatter = MapFormatter()

        grouped_symbols = {
            "test.py": [
                {
                    "name": "test_function",
                    "symbol_type": "function",
                    "line_number": 10,
                    "signature": "def test_function(x: int) -> bool",
                },
                {
                    "name": "TestClass",
                    "symbol_type": "class",
                    "line_number": 5,
                    "signature": None,
                },
            ]
        }
        repo_map_data = RepoMapData(
            project_name="TestProject",
            language_distribution={"python": 100.0},
            total_symbols=2,
            grouped_symbols=grouped_symbols,
        )
        result = formatter.format(repo_map_data)
        assert (
            "Showing top 2 symbols by usage frequency. Not all symbols are shown."
            in result
        )
        assert "test.py" in result
        assert "[function]" in result
        assert "[class]" in result

    def test_format_symbol_with_signature(self):
        """Test formatting a symbol with a signature."""
        formatter = MapFormatter()

        symbol = {
            "name": "test_function",
            "symbol_type": "function",
            "line_number": 10,
            "end_line_number": 20,
            "signature": "def test_function(x: int) -> bool",
        }

        result = formatter._format_symbol(symbol)
        assert "[function]" in result
        assert "def test_function(x: int) -> bool" in result
        assert "[line: 10-20]" in result

    def test_format_symbol_without_signature(self):
        """Test formatting a symbol without a signature."""
        formatter = MapFormatter()

        symbol = {
            "name": "TestClass",
            "symbol_type": "class",
            "line_number": 5,
            "signature": None,
        }

        result = formatter._format_symbol(symbol)
        assert "[class]" in result
        assert "[line: 5]" in result


class TestMapFormatterGolden:
    """Golden tests for MapFormatter.format()."""

    def test_simple_python_file_golden(self):
        """Test formatting a simple Python file with class and function."""
        formatter = MapFormatter()

        grouped_symbols = {
            "test.py": [
                {
                    "name": "TestClass",
                    "symbol_type": "class",
                    "line_number": 5,
                    "end_line_number": 30,
                    "signature": "class TestClass",
                },
                {
                    "name": "test_function",
                    "symbol_type": "function",
                    "line_number": 10,
                    "end_line_number": 20,
                    "signature": "def test_function(x: int) -> bool",
                },
            ]
        }
        repo_map_data = RepoMapData(
            project_name="TestProject",
            language_distribution={"python": 100.0},
            total_symbols=2,
            grouped_symbols=grouped_symbols,
        )
        result = formatter.format(repo_map_data)
        assert result == _read_golden("simple_python_file.txt")

    def test_multiple_files_golden(self):
        """Test formatting multiple files."""
        formatter = MapFormatter()

        grouped_symbols = {
            "src/core/error_handler.py": [
                {
                    "name": "CustomException",
                    "symbol_type": "class",
                    "line_number": 10,
                    "end_line_number": 15,
                    "signature": "class CustomException",
                },
                {
                    "name": "handle_error",
                    "symbol_type": "function",
                    "line_number": 20,
                    "end_line_number": 30,
                    "signature": "def handle_error(error: Exception) -> None",
                },
            ],
            "src/core/config.py": [
                {
                    "name": "load_config",
                    "symbol_type": "function",
                    "line_number": 15,
                    "end_line_number": 25,
                    "signature": "def load_config(path: str) -> Config",
                },
            ],
        }
        repo_map_data = RepoMapData(
            project_name="TestProject",
            language_distribution={"python": 100.0},
            total_symbols=3,
            grouped_symbols=grouped_symbols,
        )
        result = formatter.format(repo_map_data)
        assert result == _read_golden("multiple_files.txt")

    def test_mixed_symbol_types_golden(self):
        """Test formatting with mixed symbol types (class, method, function)."""
        formatter = MapFormatter()

        grouped_symbols = {
            "test.py": [
                {
                    "name": "MyClass",
                    "symbol_type": "class",
                    "line_number": 5,
                    "end_line_number": 25,
                    "signature": "class MyClass",
                },
                {
                    "name": "my_method",
                    "symbol_type": "method",
                    "line_number": 10,
                    "end_line_number": 15,
                    "signature": "def my_method(self, x: int) -> str",
                },
                {
                    "name": "standalone_func",
                    "symbol_type": "function",
                    "line_number": 20,
                    "end_line_number": 22,
                    "signature": "def standalone_func() -> None",
                },
                {
                    "name": "AnotherClass",
                    "symbol_type": "class",
                    "line_number": 30,
                    "end_line_number": 40,
                    "signature": "class AnotherClass",
                },
                {
                    "name": "another_method",
                    "symbol_type": "method",
                    "line_number": 35,
                    "end_line_number": 38,
                    "signature": "def another_method(self) -> bool",
                },
            ]
        }
        repo_map_data = RepoMapData(
            project_name="TestProject",
            language_distribution={"python": 100.0},
            total_symbols=5,
            grouped_symbols=grouped_symbols,
        )
        result = formatter.format(repo_map_data)
        assert result == _read_golden("mixed_symbol_types.txt")

    def test_nested_paths_golden(self):
        """Test formatting with nested directory paths."""
        formatter = MapFormatter()

        grouped_symbols = {
            "src/local_coding_assistant/agent/llm/service.py": [
                {
                    "name": "LLMService",
                    "symbol_type": "class",
                    "line_number": 10,
                    "end_line_number": 50,
                    "signature": "class LLMService",
                },
                {
                    "name": "generate_completion",
                    "symbol_type": "method",
                    "line_number": 20,
                    "end_line_number": 30,
                    "signature": "def generate_completion(prompt: str) -> str",
                },
                {
                    "name": "helper",
                    "symbol_type": "function",
                    "line_number": 35,
                    "end_line_number": 37,
                    "signature": "def helper() -> None",
                },
            ]
        }
        repo_map_data = RepoMapData(
            project_name="TestProject",
            language_distribution={"python": 100.0},
            total_symbols=3,
            grouped_symbols=grouped_symbols,
        )
        result = formatter.format(repo_map_data)
        assert result == _read_golden("nested_paths.txt")

    def test_symbols_without_signatures_golden(self):
        """Test formatting symbols without signatures."""
        formatter = MapFormatter()

        grouped_symbols = {
            "test.py": [
                {
                    "name": "TestClass",
                    "symbol_type": "class",
                    "line_number": 5,
                    "end_line_number": 15,
                    "signature": None,
                },
                {
                    "name": "AnotherClass",
                    "symbol_type": "class",
                    "line_number": 20,
                    "end_line_number": 30,
                    "signature": None,
                },
                {
                    "name": "simple_function",
                    "symbol_type": "function",
                    "line_number": 35,
                    "end_line_number": 40,
                    "signature": None,
                },
            ]
        }
        repo_map_data = RepoMapData(
            project_name="TestProject",
            language_distribution={"python": 100.0},
            total_symbols=3,
            grouped_symbols=grouped_symbols,
        )
        result = formatter.format(repo_map_data)
        assert result == _read_golden("symbols_without_signatures.txt")
