"""Unit tests for database operations."""

from pathlib import Path

import pytest

from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.models import (
    FileMetadata,
    ImportInfo,
    ImportType,
    SymbolDetail,
    SymbolType,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> RepositoryDatabase:
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    return RepositoryDatabase(db_path)


class TestRepositoryDatabase:
    """Test cases for RepositoryDatabase class."""

    def test_database_creation(self, temp_db: RepositoryDatabase) -> None:
        """Test that database is created with correct schema."""
        assert temp_db.db_path.exists()

    def test_add_or_update_file(self, temp_db: RepositoryDatabase) -> None:
        """Test adding a file to the database."""
        metadata = FileMetadata(
            path="test.py",
            language="python",
            last_modified=1234567890,
            last_indexed=1234567890,
            hash="abc123",
            size_bytes=100,
        )
        file_id = temp_db.add_or_update_file(metadata)
        assert file_id > 0

    def test_get_file(self, temp_db: RepositoryDatabase) -> None:
        """Test retrieving a file from the database."""
        metadata = FileMetadata(
            path="test.py",
            language="python",
            last_modified=1234567890,
            last_indexed=1234567890,
            hash="abc123",
            size_bytes=100,
        )
        temp_db.add_or_update_file(metadata)

        retrieved = temp_db.get_file("test.py")
        assert retrieved is not None
        assert retrieved.path == "test.py"
        assert retrieved.language == "python"

    def test_get_file_not_found(self, temp_db: RepositoryDatabase) -> None:
        """Test retrieving a non-existent file."""
        retrieved = temp_db.get_file("nonexistent.py")
        assert retrieved is None

    def test_update_file(self, temp_db: RepositoryDatabase) -> None:
        """Test updating an existing file."""
        metadata = FileMetadata(
            path="test.py",
            language="python",
            last_modified=1234567890,
            last_indexed=1234567890,
            hash="abc123",
            size_bytes=100,
        )
        temp_db.add_or_update_file(metadata)

        updated_metadata = FileMetadata(
            path="test.py",
            language="python",
            last_modified=1234567900,
            last_indexed=1234567900,
            hash="def456",
            size_bytes=200,
        )
        temp_db.add_or_update_file(updated_metadata)

        retrieved = temp_db.get_file("test.py")
        assert retrieved.last_modified == 1234567900
        assert retrieved.hash == "def456"

    def test_delete_file(self, temp_db: RepositoryDatabase) -> None:
        """Test deleting a file from the database."""
        metadata = FileMetadata(
            path="test.py",
            language="python",
            last_modified=1234567890,
            last_indexed=1234567890,
            hash="abc123",
            size_bytes=100,
        )
        temp_db.add_or_update_file(metadata)

        temp_db.delete_file("test.py")
        assert temp_db.get_file("test.py") is None

    def test_add_symbol(self, temp_db: RepositoryDatabase) -> None:
        """Test adding a symbol to the database."""
        # First add a file
        file_metadata = FileMetadata(
            path="test.py",
            language="python",
            last_modified=1234567890,
            last_indexed=1234567890,
            hash="abc123",
            size_bytes=100,
        )
        file_id = temp_db.add_or_update_file(file_metadata)

        # Add a symbol
        symbol = SymbolDetail(
            symbol_id=0,  # Will be assigned by database
            file_id=file_id,
            name="test_function",
            symbol_type=SymbolType.FUNCTION,
            line_number=10,
            end_line_number=20,
            parent_scope=None,
            docstring="A test function",
            content="def test_function(): pass",
            signature="def test_function() -> None",
        )
        symbol_id = temp_db.add_symbol(symbol)
        assert symbol_id > 0

    def test_delete_symbols_for_file(self, temp_db: RepositoryDatabase) -> None:
        """Test deleting symbols for a file."""
        # Add a file
        file_metadata = FileMetadata(
            path="test.py",
            language="python",
            last_modified=1234567890,
            last_indexed=1234567890,
            hash="abc123",
            size_bytes=100,
        )
        file_id = temp_db.add_or_update_file(file_metadata)

        # Add a symbol
        symbol = SymbolDetail(
            symbol_id=0,
            file_id=file_id,
            name="test_function",
            symbol_type=SymbolType.FUNCTION,
            line_number=10,
            end_line_number=20,
            parent_scope=None,
            docstring="A test function",
            content="def test_function(): pass",
            signature="def test_function() -> None",
        )
        temp_db.add_symbol(symbol)

        # Delete symbols
        temp_db.delete_symbols_for_file(file_id)

    def test_add_import(self, temp_db: RepositoryDatabase) -> None:
        """Test adding an import to the database."""
        # Add a file
        file_metadata = FileMetadata(
            path="test.py",
            language="python",
            last_modified=1234567890,
            last_indexed=1234567890,
            hash="abc123",
            size_bytes=100,
        )
        file_id = temp_db.add_or_update_file(file_metadata)

        # Add an import
        import_info = ImportInfo(
            file_id=file_id,
            import_statement="import os",
            import_type=ImportType.MODULE,
            imported_symbols=[],
        )
        import_id = temp_db.add_import(import_info)
        assert import_id > 0

    def test_delete_imports_for_file(self, temp_db: RepositoryDatabase) -> None:
        """Test deleting imports for a file."""
        # Add a file
        file_metadata = FileMetadata(
            path="test.py",
            language="python",
            last_modified=1234567890,
            last_indexed=1234567890,
            hash="abc123",
            size_bytes=100,
        )
        file_id = temp_db.add_or_update_file(file_metadata)

        # Add an import
        import_info = ImportInfo(
            file_id=file_id,
            import_statement="import os",
            import_type=ImportType.MODULE,
            imported_symbols=[],
        )
        temp_db.add_import(import_info)

        # Delete imports
        temp_db.delete_imports_for_file(file_id)

    def test_project_metadata(self, temp_db: RepositoryDatabase) -> None:
        """Test project metadata operations."""
        temp_db.set_project_metadata("project_name", "test_project")
        value = temp_db.get_project_metadata("project_name")
        assert value == "test_project"

        # Update metadata
        temp_db.set_project_metadata("project_name", "updated_project")
        value = temp_db.get_project_metadata("project_name")
        assert value == "updated_project"

        # Get non-existent metadata
        assert temp_db.get_project_metadata("nonexistent") is None

    def test_get_all_files(self, temp_db: RepositoryDatabase) -> None:
        """Test retrieving all files from the database."""
        # Add multiple files
        for i in range(3):
            metadata = FileMetadata(
                path=f"test{i}.py",
                language="python",
                last_modified=1234567890 + i,
                last_indexed=1234567890 + i,
                hash=f"hash{i}",
                size_bytes=100 + i,
            )
            temp_db.add_or_update_file(metadata)

        files = temp_db.get_all_files()
        assert len(files) == 3
