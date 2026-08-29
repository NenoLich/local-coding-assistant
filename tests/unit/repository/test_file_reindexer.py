"""Unit tests for FileReindexer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.file_filter import FileFilter
from local_coding_assistant.repository.file_reindexer import FileReindexer
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("test_file_reindexer")


@pytest.fixture
def mock_database():
    """Create a mock database."""
    db = MagicMock(spec=RepositoryDatabase)
    return db


@pytest.fixture
def file_filter():
    """Create a file filter."""
    return FileFilter()


@pytest.fixture
def temp_target_project_root(tmp_path):
    """Create a temporary project root."""
    (tmp_path / "project").mkdir()
    return tmp_path / "project"


@pytest.fixture
def sample_file(temp_target_project_root):
    """Create a sample Python file."""
    file_path = temp_target_project_root / "test.py"
    file_path.write_text("def hello():\n   Pass\n")
    return file_path


class TestFileReindexer:
    """Tests for FileReindexer."""

    def test_init(self, mock_database, file_filter):
        """Test initialization of FileReindexer."""
        reindexer = FileReindexer(
            database=mock_database,
            file_filter=file_filter,
            parallel_workers=2,
        )
        assert reindexer.database == mock_database
        assert reindexer.file_filter == file_filter
        assert reindexer.parallel_workers == 2

    def test_reindex_file_success(self, mock_database, file_filter, sample_file):
        """Test successful re-indexing of a file."""
        reindexer = FileReindexer(
            database=mock_database,
            file_filter=file_filter,
        )

        with patch.object(reindexer, "ast_parser") as mock_parser:
            mock_parser.parse.return_value = ([], MagicMock(), [])

            with patch.object(reindexer, "symbol_indexer") as mock_indexer:
                mock_indexer.index_file.return_value = 1

                result = reindexer.reindex_file(sample_file)
                assert result is True

    def test_reindex_file_filtered(
        self, mock_database, file_filter, temp_target_project_root
    ):
        """Test re-indexing a file that is filtered out."""
        reindexer = FileReindexer(
            database=mock_database,
            file_filter=file_filter,
        )

        # Create a file with unsupported extension
        unsupported_file = temp_target_project_root / "test.txt"
        unsupported_file.write_text("content")

        result = reindexer.reindex_file(unsupported_file)
        assert result is False

    def test_reindex_file_not_exists(
        self, mock_database, file_filter, temp_target_project_root
    ):
        """Test re-indexing a file that doesn't exist."""
        reindexer = FileReindexer(
            database=mock_database,
            file_filter=file_filter,
        )

        non_existent = temp_target_project_root / "nonexistent.py"
        result = reindexer.reindex_file(non_existent)
        assert result is False

    def test_reindex_file_parse_error(self, mock_database, file_filter, sample_file):
        """Test re-indexing a file that fails to parse."""
        reindexer = FileReindexer(
            database=mock_database,
            file_filter=file_filter,
        )

        with patch.object(reindexer, "ast_parser") as mock_parser:
            mock_parser.parse.side_effect = Exception("Parse error")

            result = reindexer.reindex_file(sample_file)
            assert result is False

    def test_parse_and_reindex_files_parallel_empty(self, mock_database, file_filter):
        """Test parallel re-indexing with empty file list."""
        reindexer = FileReindexer(
            database=mock_database,
            file_filter=file_filter,
        )

        result, file_hashes = reindexer.parse_and_reindex_files_parallel([])
        assert result == 0
        assert file_hashes == {}

    def test_parse_and_reindex_files_parallel_success(
        self, mock_database, file_filter, temp_target_project_root
    ):
        """Test successful parallel re-indexing of multiple files."""
        reindexer = FileReindexer(
            database=mock_database,
            file_filter=file_filter,
        )

        # Create multiple Python files
        file1 = temp_target_project_root / "file1.py"
        file1.write_text("def func1(): pass")

        file2 = temp_target_project_root / "file2.py"
        file2.write_text("def func2(): pass")

        with patch.object(reindexer, "ast_parser") as mock_parser:
            mock_parser.parse.return_value = ([], MagicMock(), [])

            with patch.object(reindexer, "symbol_indexer") as mock_indexer:
                mock_indexer.index_files.return_value = 2

                result, file_hashes = reindexer.parse_and_reindex_files_parallel(
                    [file1, file2]
                )
                assert result == 2

    def test_parse_and_reindex_files_parallel_with_filtering(
        self, mock_database, temp_target_project_root
    ):
        """Test parallel re-indexing with file filtering."""
        file_filter = FileFilter()
        reindexer = FileReindexer(
            database=mock_database,
            file_filter=file_filter,
        )

        # Create Python and text files
        py_file = temp_target_project_root / "test.py"
        py_file.write_text("def test(): pass")

        txt_file = temp_target_project_root / "test.txt"
        txt_file.write_text("content")

        with patch.object(reindexer, "ast_parser") as mock_parser:
            mock_parser.parse.return_value = ([], MagicMock(), [])

            with patch.object(reindexer, "symbol_indexer") as mock_indexer:
                mock_indexer.index_files.return_value = 1

                # Should only index the Python file
                files = [py_file, txt_file]
                _, filtered_files = file_filter.filter_files(files)
                result, file_hashes = reindexer.parse_and_reindex_files_parallel(
                    filtered_files
                )
                assert result == 1
