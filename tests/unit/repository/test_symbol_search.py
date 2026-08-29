"""Unit tests for symbol indexing and search functionality."""

import tempfile
from pathlib import Path

import pytest

from local_coding_assistant.repository.ast_parser import ASTNode
from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.models import FileMetadata, SymbolType
from local_coding_assistant.repository.symbol_search import (
    SymbolIndexer,
    SymbolSearcher,
)


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
def symbol_indexer(database):
    """Create a symbol indexer instance."""
    return SymbolIndexer(database)


@pytest.fixture
def symbol_searcher(database):
    """Create a symbol searcher instance."""
    return SymbolSearcher(database)


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
        content='''
class TestClass:
    def __init__(self):
        pass

def test_function(x: int, y: str) -> bool:
    """A test function."""
    return True
''',
    )


@pytest.fixture
def sample_symbols():
    """Create sample AST symbols."""
    return [
        ASTNode(
            type="function_definition",
            name="test_function",
            line_number=10,
            end_line_number=20,
            column=0,
            metadata={"symbol_type": "function", "parent_scope": None},
            signature="def test_function(x: int, y: str) -> bool",
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


@pytest.fixture
def sample_content():
    """Create sample file content."""
    return '''
class TestClass:
    def __init__(self):
        pass

def test_function(x: int, y: str) -> bool:
    """A test function."""
    return True
'''


class TestSymbolIndexer:
    """Tests for SymbolIndexer."""

    def test_index_file(self, symbol_indexer, sample_file_metadata, sample_symbols):
        """Test indexing symbols from a file."""
        result = symbol_indexer.index_file(
            symbols=sample_symbols,
            file_metadata=sample_file_metadata,
        )

        assert result == 2

    def test_index_file_with_signature(
        self, symbol_indexer, symbol_searcher, sample_file_metadata, sample_symbols
    ):
        """Test that signatures are indexed correctly."""
        symbol_indexer.index_file(
            symbols=sample_symbols,
            file_metadata=sample_file_metadata,
        )

        # Retrieve symbols and check signature
        symbols = symbol_searcher.get_symbols_in_file("test.py")
        assert len(symbols) == 2

        # Find the function symbol
        func_symbol = next((s for s in symbols if s.name == "test_function"), None)
        assert func_symbol is not None
        assert func_symbol.signature == "def test_function(x: int, y: str) -> bool"

    def test_index_file_with_call_relationships(
        self, symbol_indexer, sample_file_metadata, sample_symbols
    ):
        """Test that call relationships are indexed correctly."""
        from local_coding_assistant.repository.models import CallRelationship

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
            CallRelationship(
                caller_name="TestClass",
                caller_line=5,
                callee_name="test_function",
                callee_line=10,
                relationship_type="call",
                language="python",
                file_path="test_path",
            ),
        ]

        symbol_indexer.index_file(
            symbols=sample_symbols,
            file_metadata=sample_file_metadata,
            call_relationships=call_relationships,
        )

        # Retrieve call relationships
        all_relationships = symbol_indexer.database.get_all_call_relationships()
        assert len(all_relationships) == 2
        assert all_relationships[0]["caller_name"] == "test_function"
        assert all_relationships[0]["callee_name"] == "helper_func"

    def test_index_file_updates_existing(
        self, symbol_indexer, sample_file_metadata, sample_symbols
    ):
        """Test that re-indexing a file updates existing symbols."""
        # Index once
        symbol_indexer.index_file(
            symbols=sample_symbols,
            file_metadata=sample_file_metadata,
        )

        # Index again with updated metadata
        updated_metadata = FileMetadata(
            path="test.py",
            language="python",
            last_modified=1234567891,
            last_indexed=1234567891,
            hash="xyz789",
            size_bytes=2000,
        )
        result = symbol_indexer.index_file(
            symbols=sample_symbols,
            file_metadata=updated_metadata,
        )

        assert result == 2

        # Check that file metadata was updated
        file_data = symbol_indexer.database.get_file("test.py")
        assert file_data is not None
        assert file_data.hash == "xyz789"
        assert file_data.size_bytes == 2000

    def test_index_files_bulk(self, symbol_indexer, symbol_searcher):
        """Test bulk indexing of multiple files."""
        from local_coding_assistant.repository.models import CallRelationship

        # Create data for multiple files
        file1_metadata = FileMetadata(
            path="file1.py",
            language="python",
            last_modified=1234567890,
            last_indexed=1234567890,
            hash="abc123",
            size_bytes=1000,
            content="def func1(): pass",
        )
        file1_symbols = [
            ASTNode(
                type="function_definition",
                name="func1",
                line_number=1,
                end_line_number=2,
                column=0,
                metadata={"symbol_type": "function", "parent_scope": None},
                signature="def func1()",
            ),
        ]

        file2_metadata = FileMetadata(
            path="file2.py",
            language="python",
            last_modified=1234567891,
            last_indexed=1234567891,
            hash="def456",
            size_bytes=1000,
            content="def func2(): pass\nclass Class1: pass",
        )
        file2_symbols = [
            ASTNode(
                type="function_definition",
                name="func2",
                line_number=1,
                end_line_number=2,
                column=0,
                metadata={"symbol_type": "function", "parent_scope": None},
                signature="def func2()",
            ),
            ASTNode(
                type="class_definition",
                name="Class1",
                line_number=3,
                end_line_number=4,
                column=0,
                metadata={"symbol_type": "class", "parent_scope": None},
                signature=None,
            ),
        ]

        file3_metadata = FileMetadata(
            path="file3.py",
            language="python",
            last_modified=1234567892,
            last_indexed=1234567892,
            hash="ghi789",
            size_bytes=1000,
            content="def func3(): pass",
        )
        file3_symbols = [
            ASTNode(
                type="function_definition",
                name="func3",
                line_number=1,
                end_line_number=2,
                column=0,
                metadata={"symbol_type": "function", "parent_scope": None},
                signature="def func3()",
            ),
        ]

        # Add call relationships for some files
        call_relationships = [
            CallRelationship(
                caller_name="func1",
                caller_line=1,
                callee_name="func2",
                callee_line=1,
                relationship_type="call",
                language="python",
                file_path="file1.py",
            ),
        ]

        files_data = [
            (file1_symbols, file1_metadata, call_relationships),
            (file2_symbols, file2_metadata, None),
            (file3_symbols, file3_metadata, None),
        ]

        # Bulk index all files
        result = symbol_indexer.index_files(files_data)

        assert result == 4  # Total symbols across all files

        # Verify symbols are indexed
        symbols_file1 = symbol_searcher.get_symbols_in_file("file1.py")
        assert len(symbols_file1) == 1
        assert symbols_file1[0].name == "func1"

        symbols_file2 = symbol_searcher.get_symbols_in_file("file2.py")
        assert len(symbols_file2) == 2
        assert symbols_file2[0].name == "func2"
        assert symbols_file2[1].name == "Class1"

        symbols_file3 = symbol_searcher.get_symbols_in_file("file3.py")
        assert len(symbols_file3) == 1
        assert symbols_file3[0].name == "func3"

        # Verify call relationships are indexed
        all_relationships = symbol_indexer.database.get_all_call_relationships()
        assert len(all_relationships) == 1
        assert all_relationships[0]["caller_name"] == "func1"
        assert all_relationships[0]["callee_name"] == "func2"

    def test_index_files_empty_list(self, symbol_indexer):
        """Test bulk indexing with empty file list."""
        result = symbol_indexer.index_files([])
        assert result == 0

    def test_index_files_with_mixed_relationships(
        self, symbol_indexer, symbol_searcher
    ):
        """Test bulk indexing with mixed call relationships across files."""
        from local_coding_assistant.repository.models import CallRelationship

        file1_metadata = FileMetadata(
            path="file1.py",
            language="python",
            last_modified=1234567890,
            last_indexed=1234567890,
            hash="abc123",
            size_bytes=1000,
            content="def func1(): pass",
        )
        file1_symbols = [
            ASTNode(
                type="function_definition",
                name="func1",
                line_number=1,
                end_line_number=2,
                column=0,
                metadata={"symbol_type": "function", "parent_scope": None},
                signature="def func1()",
            ),
        ]

        file2_metadata = FileMetadata(
            path="file2.py",
            language="python",
            last_modified=1234567891,
            last_indexed=1234567891,
            hash="def456",
            size_bytes=1000,
            content="def func2(): pass",
        )
        file2_symbols = [
            ASTNode(
                type="function_definition",
                name="func2",
                line_number=1,
                end_line_number=2,
                column=0,
                metadata={"symbol_type": "function", "parent_scope": None},
                signature="def func2()",
            ),
        ]

        # Call relationships for both files
        call_relationships = [
            CallRelationship(
                caller_name="func1",
                caller_line=1,
                callee_name="func2",
                callee_line=1,
                relationship_type="call",
                language="python",
                file_path="file1.py",
            ),
            CallRelationship(
                caller_name="func2",
                caller_line=1,
                callee_name="func1",
                callee_line=1,
                relationship_type="call",
                language="python",
                file_path="file2.py",
            ),
        ]

        files_data = [
            (file1_symbols, file1_metadata, call_relationships[:1]),
            (file2_symbols, file2_metadata, call_relationships[1:]),
        ]

        result = symbol_indexer.index_files(files_data)
        assert result == 2

        # Verify all call relationships are indexed
        all_relationships = symbol_indexer.database.get_all_call_relationships()
        assert len(all_relationships) == 2


class TestSymbolSearcher:
    """Tests for SymbolSearcher."""

    def test_search_symbols(
        self, symbol_searcher, symbol_indexer, sample_file_metadata, sample_symbols
    ):
        """Test searching for symbols."""
        # First index some symbols
        symbol_indexer.index_file(
            symbols=sample_symbols,
            file_metadata=sample_file_metadata,
        )

        # Search for symbols
        results = symbol_searcher.search_symbols("test_function")
        assert len(results) >= 1

        # Check that the function is found
        func_result = next((r for r in results if r.name == "test_function"), None)
        assert func_result is not None
        assert func_result.symbol_type == SymbolType.FUNCTION

    def test_search_symbols_with_type_filter(
        self, symbol_searcher, symbol_indexer, sample_file_metadata, sample_symbols
    ):
        """Test searching for symbols with type filter."""
        symbol_indexer.index_file(
            symbols=sample_symbols,
            file_metadata=sample_file_metadata,
        )

        # Search only for functions
        results = symbol_searcher.search_symbols("test", symbol_types=["function"])
        assert all(r.symbol_type == SymbolType.FUNCTION for r in results)

    def test_search_symbols_with_file_path_filter(
        self, symbol_searcher, symbol_indexer, sample_file_metadata, sample_symbols
    ):
        """Test searching for symbols with file path filter."""
        symbol_indexer.index_file(
            symbols=sample_symbols,
            file_metadata=sample_file_metadata,
        )

        # Search with file path filter
        results = symbol_searcher.search_symbols("test", file_path="test.py")
        assert len(results) >= 1
        assert all(r.file_path == "test.py" for r in results)

        # Search with non-existent file path
        results_empty = symbol_searcher.search_symbols(
            "test", file_path="nonexistent.py"
        )
        assert len(results_empty) == 0

    def test_get_symbol_by_id(
        self, symbol_searcher, symbol_indexer, sample_file_metadata, sample_symbols
    ):
        """Test getting a symbol by ID."""
        symbol_indexer.index_file(
            symbols=sample_symbols,
            file_metadata=sample_file_metadata,
        )

        # Get all symbols to find an ID
        symbols = symbol_searcher.get_symbols_in_file("test.py")
        assert len(symbols) > 0

        # Get the first symbol by ID
        symbol = symbol_searcher.get_symbol_by_id(symbols[0].symbol_id)
        assert symbol is not None
        assert symbol.name == symbols[0].name

    def test_get_symbols_in_file(
        self, symbol_searcher, symbol_indexer, sample_file_metadata, sample_symbols
    ):
        """Test getting all symbols in a file."""
        symbol_indexer.index_file(
            symbols=sample_symbols,
            file_metadata=sample_file_metadata,
        )

        symbols = symbol_searcher.get_symbols_in_file("test.py")
        assert len(symbols) == 2

        # Check that symbols are sorted by line number
        line_numbers = [s.line_number for s in symbols]
        assert line_numbers == sorted(line_numbers)

    def test_get_symbols_in_nonexistent_file(self, symbol_searcher):
        """Test getting symbols from a non-existent file."""
        symbols = symbol_searcher.get_symbols_in_file("nonexistent.py")
        assert symbols == []
