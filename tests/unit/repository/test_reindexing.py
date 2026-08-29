"""Unit tests for re-indexing components."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.file_filter import FileFilter
from local_coding_assistant.repository.file_reindexer import FileReindexer
from local_coding_assistant.repository.file_scope import FileScope
from local_coding_assistant.repository.reindexing import (
    AgentEditReindexer,
    ReindexManager,
    StartupReindexer,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("test_reindexing")


@pytest.fixture
def temp_target_project_root(tmp_path):
    """Create a temporary project root."""
    (tmp_path / "project").mkdir()
    return tmp_path / "project"


@pytest.fixture
def mock_database(temp_target_project_root):
    """Create a mock database."""
    db = MagicMock(spec=RepositoryDatabase)
    db.get_all_files.return_value = []
    db.delete_file = MagicMock()
    db.db_path = temp_target_project_root / "db"
    return db


@pytest.fixture
def file_filter():
    """Create a file filter."""
    return FileFilter()


@pytest.fixture
def sample_file(temp_target_project_root):
    """Create a sample Python file."""
    file_path = temp_target_project_root / "test.py"
    file_path.write_text("def hello():\n    pass\n")
    return file_path


class TestStartupReindexer:
    """Tests for StartupReindexer."""

    def test_init(self, temp_target_project_root, mock_database, file_filter):
        """Test initialization of StartupReindexer."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        reindexer = StartupReindexer(
            database=mock_database,
            file_reindexer=file_reindexer,
        )
        assert reindexer.database == mock_database
        assert reindexer.file_reindexer == file_reindexer

    def test_check_and_reindex_no_changes(
        self, temp_target_project_root, mock_database, file_filter
    ):
        """Test startup re-indexing when no files have changed."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        reindexer = StartupReindexer(
            database=mock_database,
            file_reindexer=file_reindexer,
        )

        stats, file_hashes = reindexer.reindex_files([])
        assert stats == {"added": 0, "updated": 0, "deleted": 0}
        assert file_hashes == {}

    def test_check_and_reindex_with_new_file(
        self, temp_target_project_root, mock_database, file_filter, sample_file
    ):
        """Test startup re-indexing with a new file."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        reindexer = StartupReindexer(
            database=mock_database,
            file_reindexer=file_reindexer,
        )

        # Mock file re-indexer
        with patch.object(reindexer, "file_reindexer") as mock_reindexer:
            mock_reindexer.parse_and_reindex_files_parallel.return_value = (
                1,
                {"test.py": "hash"},
            )

            stats, file_hashes = reindexer.reindex_files([sample_file])
            assert stats["added"] >= 1
            assert file_hashes == {"test.py": "hash"}

    def test_check_and_reindex_with_deleted_file(
        self, temp_target_project_root, mock_database, file_filter
    ):
        """Test startup re-indexing with a deleted file."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        reindexer = StartupReindexer(
            database=mock_database,
            file_reindexer=file_reindexer,
        )

        # Mock database to have a file that doesn't exist on disk
        mock_db_file = MagicMock()
        mock_db_file.path = str(temp_target_project_root / "deleted.py")
        mock_db_file.last_indexed = 0
        mock_database.get_all_files.return_value = [mock_db_file]

        stats, _ = reindexer.reindex_files([])
        assert stats["deleted"] >= 1
        mock_database.delete_file.assert_called()


class TestAgentEditReindexer:
    """Tests for AgentEditReindexer."""

    def test_init(self, mock_database, file_filter):
        """Test initialization of AgentEditReindexer."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        reindexer = AgentEditReindexer(
            file_reindexer=file_reindexer,
        )
        assert reindexer.file_reindexer == file_reindexer

    def test_reindex_file_success(self, mock_database, file_filter, sample_file):
        """Test successful re-indexing of a file."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        reindexer = AgentEditReindexer(
            file_reindexer=file_reindexer,
        )

        with patch.object(reindexer, "file_reindexer") as mock_reindexer:
            mock_reindexer.reindex_file.return_value = True

            result = reindexer.reindex_file(sample_file)
            assert result is True

    def test_reindex_file_filtered(
        self, mock_database, file_filter, temp_target_project_root
    ):
        """Test re-indexing a file that is filtered out."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        reindexer = AgentEditReindexer(
            file_reindexer=file_reindexer,
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
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        reindexer = AgentEditReindexer(
            file_reindexer=file_reindexer,
        )

        non_existent = temp_target_project_root / "nonexistent.py"
        result = reindexer.reindex_file(non_existent)
        assert result is False

    def test_reindex_file_parse_error(self, mock_database, file_filter, sample_file):
        """Test re-indexing a file that fails to parse."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        reindexer = AgentEditReindexer(
            file_reindexer=file_reindexer,
        )

        with patch.object(reindexer, "file_reindexer") as mock_reindexer:
            mock_reindexer.reindex_file.return_value = False

            result = reindexer.reindex_file(sample_file)
            assert result is False


class TestReindexManager:
    """Tests for ReindexManager."""

    def test_init(self, temp_target_project_root, mock_database, file_filter):
        """Test initialization of ReindexManager."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        manager = ReindexManager(
            target_project_root=temp_target_project_root,
            database=mock_database,
            file_reindexer=file_reindexer,
            startup_enabled=True,
            agent_edit_enabled=True,
        )

        assert manager.target_project_root == temp_target_project_root
        assert manager.database == mock_database
        assert manager.file_reindexer == file_reindexer
        assert manager.startup_enabled is True
        assert manager.agent_edit_enabled is True
        assert manager.startup_reindexer is not None
        assert manager.agent_edit_reindexer is not None
        assert manager.startup_reindexer is not None

    def test_init_disabled(self, temp_target_project_root, mock_database, file_filter):
        """Test initialization with re-indexing disabled."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        manager = ReindexManager(
            target_project_root=temp_target_project_root,
            database=mock_database,
            file_reindexer=file_reindexer,
            startup_enabled=False,
            agent_edit_enabled=False,
        )
        assert manager.startup_enabled is False
        assert manager.agent_edit_enabled is False
        assert manager.startup_reindexer is None
        assert manager.agent_edit_reindexer is None

    def test_check_and_reindex_on_startup_enabled(
        self, temp_target_project_root, mock_database, file_filter, sample_file
    ):
        """Test startup re-indexing when enabled."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        manager = ReindexManager(
            target_project_root=temp_target_project_root,
            database=mock_database,
            file_reindexer=file_reindexer,
            startup_enabled=True,
        )

        with patch.object(manager.startup_reindexer, "reindex_files") as mock_reindex:
            mock_reindex.return_value = (
                {"added": 1, "updated": 0, "deleted": 0},
                {str(sample_file): "hash"},
            )

            stats, file_hashes = manager.reindex_on_startup(
                [Path(temp_target_project_root / "test_file.py")]
            )
            assert stats == {"added": 1, "updated": 0, "deleted": 0}
            assert str(sample_file) in file_hashes
            assert file_hashes[str(sample_file)] != ""
            mock_reindex.assert_called_once()

    def test_check_and_reindex_on_startup_disabled(
        self, temp_target_project_root, mock_database, file_filter
    ):
        """Test startup re-indexing when disabled."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        manager = ReindexManager(
            target_project_root=temp_target_project_root,
            database=mock_database,
            file_reindexer=file_reindexer,
            startup_enabled=False,
        )

        stats, file_hashes = manager.reindex_on_startup(
            [Path(temp_target_project_root / "test_file.py")]
        )
        assert stats == {"added": 0, "updated": 0, "deleted": 0}
        assert file_hashes == {}

    def test_reindex_agent_edit_enabled(
        self, temp_target_project_root, mock_database, file_filter, sample_file
    ):
        """Test agent edit re-indexing when enabled."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        manager = ReindexManager(
            target_project_root=temp_target_project_root,
            database=mock_database,
            file_reindexer=file_reindexer,
            agent_edit_enabled=True,
        )

        with patch.object(manager.agent_edit_reindexer, "reindex_file") as mock_reindex:
            mock_reindex.return_value = True

            result = manager.reindex_agent_edit(sample_file)
            assert result is True
            mock_reindex.assert_called_once_with(sample_file)

    def test_reindex_agent_edit_disabled(
        self, temp_target_project_root, mock_database, file_filter, sample_file
    ):
        """Test agent edit re-indexing when disabled."""
        file_reindexer = FileReindexer(database=mock_database, file_filter=file_filter)
        manager = ReindexManager(
            target_project_root=temp_target_project_root,
            database=mock_database,
            file_reindexer=file_reindexer,
            agent_edit_enabled=False,
        )

        result = manager.reindex_agent_edit(sample_file)
        assert result is False
