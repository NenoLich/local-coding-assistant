"""Integration tests for repository context service."""

import asyncio
import os
import tempfile
import time
from pathlib import Path

import pytest

from local_coding_assistant.repository.metadata import MetadataExtractor
from local_coding_assistant.repository.models import ProjectInfo
from local_coding_assistant.repository.repo_map import RepoMapData
from local_coding_assistant.repository.service import RepositoryContextService


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory with sample config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)

        # Create a pyproject.toml file
        pyproject_content = """
[project]
name = "test-project"
description = "A test project"
version = "0.1.0"
license = "MIT"
dependencies = [
    "fastapi>=0.100.0",
    "pydantic>=2.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
"""
        (project_path / "pyproject.toml").write_text(pyproject_content)

        # Create a sample Python file
        (project_path / "src").mkdir()
        (project_path / "src" / "main.py").write_text("""
def hello():
    print("Hello, world!")

class MyClass:
    def method(self):
        pass
""")

        yield project_path


class TestMetadataExtractor:
    """Tests for MetadataExtractor."""

    def test_extract_from_pyproject_toml(self, temp_project_dir):
        """Test extracting metadata from pyproject.toml."""
        extractor = MetadataExtractor()
        tracked_files = {temp_project_dir / "pyproject.toml"}
        info = extractor.extract(tracked_files, temp_project_dir)

        assert info.name == "test-project"
        assert info.description == "A test project"
        assert info.package_manager == "hatch"
        assert info.build_backend == "hatchling.build"
        assert "Black" in info.linter_formatter
        assert "Ruff" in info.linter_formatter
        assert "pytest" in info.test_frameworks
        assert "FastAPI" in info.core_frameworks
        assert len(info.clean_dependencies) > 0

    def test_extract_from_empty_project(self, temp_project_dir):
        """Test extracting metadata from project with no config files."""
        # Remove config files
        (temp_project_dir / "pyproject.toml").unlink()

        extractor = MetadataExtractor()
        tracked_files = {temp_project_dir / "pyproject.toml"}
        info = extractor.extract(tracked_files, temp_project_dir)

        assert info.name is None


class TestRepositoryContextService:
    """Tests for RepositoryContextService."""

    def test_service_initialization(self, temp_project_dir, repository_config_manager):
        """Test service initialization."""
        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.initialize_db_dependencies()

        assert service.target_project_root == temp_project_dir
        assert service.storage_mode == "temporary"
        assert service.database is not None
        assert service.metadata_extractor is not None
        assert service.repo_map_builder is not None

        service.close()

    def test_get_project_info(self, temp_project_dir, repository_config_manager):
        """Test getting project info."""
        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )

        project_info = service.get_project_info()

        assert isinstance(project_info, ProjectInfo)
        assert project_info.name == "test-project"

        service.close()

    def test_get_repo_context(self, temp_project_dir, repository_config_manager):
        """Test getting repository context."""
        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.initialize_db_dependencies()

        project_meta, repo_map_data = service.get_repo_context()

        assert isinstance(project_meta, ProjectInfo)
        assert isinstance(repo_map_data, RepoMapData)

        service.close()

    def test_get_repo_map_data(self, temp_project_dir, repository_config_manager):
        """Test getting repo map as structured data."""
        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.initialize_db_dependencies()

        repo_map_data = service.get_repo_map_data()

        # Database is empty in this test, so project_name will be None
        assert isinstance(repo_map_data, RepoMapData)
        assert isinstance(repo_map_data.language_distribution, dict)
        assert isinstance(repo_map_data.total_symbols, int)
        assert isinstance(repo_map_data.grouped_symbols, dict)

        service.close()

    def test_context_manager(self, temp_project_dir, repository_config_manager):
        """Test using service as context manager."""
        with RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        ) as service:
            assert service is not None
            project_info = service.get_project_info()
            assert project_info.name == "test-project"

    def test_custom_repo_map_config(self, temp_project_dir, repository_config_manager):
        """Test service with custom repo map config."""
        # Update the config in the manager
        repository_config_manager._global_config.repository.repo_map.max_symbols = 50
        repository_config_manager._global_config.repository.repo_map.include_docstrings = False

        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.initialize_db_dependencies()

        assert service.repo_map_builder.max_symbols == 50
        assert service.repo_map_builder.include_docstrings is False

        service.close()

    def test_index_files_batch_small_files(
        self, temp_project_dir, repository_config_manager
    ):
        """Test batch indexing with small files that fit in one chunk."""
        # Create multiple small Python files
        for i in range(5):
            file_path = temp_project_dir / f"src" / f"file{i}.py"
            file_path.write_text(f"""
def func{i}():
    return {i}

class Class{i}:
    pass
""")

        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.initialize_db_dependencies()

        # Get all Python files
        file_paths = list((temp_project_dir / "src").glob("*.py"))

        # Index files in batch with large chunk size (should process in one chunk)
        total_files = service.index_files_batch(
            file_paths, max_cumulative_size_kb=10000
        )

        assert total_files > 0

        # Verify symbols are indexed by checking each file
        from local_coding_assistant.repository.symbol_search import SymbolSearcher

        searcher = SymbolSearcher(service.database)
        all_symbols_count = 0
        for file_path in file_paths:
            symbols = searcher.get_symbols_in_file(str(file_path))
            all_symbols_count += len(symbols)
        assert all_symbols_count >= 10  # 5 files * 2 symbols each

        service.close()

    def test_index_files_batch_chunking(
        self, temp_project_dir, repository_config_manager
    ):
        """Test batch indexing with chunking based on file size."""
        # Create files of varying sizes
        small_file = temp_project_dir / "src" / "small.py"
        small_file.write_text("def small(): pass")

        medium_file = temp_project_dir / "src" / "medium.py"
        medium_file.write_text("\n".join([f"def func{i}(): pass" for i in range(100)]))

        large_file = temp_project_dir / "src" / "large.py"
        large_file.write_text("\n".join([f"class Class{i}: pass" for i in range(200)]))

        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.initialize_db_dependencies()

        file_paths = [small_file, medium_file, large_file]

        # Use small chunk size to force chunking
        total_files = service.index_files_batch(file_paths, max_cumulative_size_kb=1)

        assert total_files > 0

        # Verify all files are indexed by checking each file
        from local_coding_assistant.repository.symbol_search import SymbolSearcher

        searcher = SymbolSearcher(service.database)
        all_symbols_count = 0
        for file_path in file_paths:
            symbols = searcher.get_symbols_in_file(str(file_path))
            all_symbols_count += len(symbols)
        assert all_symbols_count >= 300  # 1 + 100 + 200

        service.close()

    def test_index_files_batch_empty_list(
        self, temp_project_dir, repository_config_manager
    ):
        """Test batch indexing with empty file list."""
        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.initialize_db_dependencies()

        total_files = service.index_files_batch([], max_cumulative_size_kb=10000)

        assert total_files == 0

        service.close()

    def test_index_files_batch_with_errors(
        self, temp_project_dir, repository_config_manager
    ):
        """Test batch indexing with some files that have parsing errors."""
        # Create valid files
        valid_file1 = temp_project_dir / "src" / "valid1.py"
        valid_file1.write_text("def valid1(): pass")

        valid_file2 = temp_project_dir / "src" / "valid2.py"
        valid_file2.write_text("def valid2(): pass")

        # Create a file that might cause parsing issues (invalid syntax)
        invalid_file = temp_project_dir / "src" / "invalid.py"
        invalid_file.write_text("def invalid(:")  # Invalid syntax

        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.initialize_db_dependencies()

        file_paths = [valid_file1, valid_file2, invalid_file]

        # Should continue processing even if one file fails
        total_files = service.index_files_batch(
            file_paths, max_cumulative_size_kb=10000
        )

        # Should have indexed at least the valid files
        assert total_files >= 2

        service.close()

    def test_index_files_batch_single_file(
        self, temp_project_dir, repository_config_manager
    ):
        """Test batch indexing with a single file."""
        single_file = temp_project_dir / "src" / "single.py"
        single_file.write_text("def single_func(): pass")

        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.initialize_db_dependencies()

        total_files = service.index_files_batch(
            [single_file], max_cumulative_size_kb=10000
        )

        assert total_files == 1

        service.close()

    def test_startup_reindexing(self, temp_project_dir, repository_config_manager):
        """Test startup re-indexing functionality."""

        # Create a Python file
        test_file = temp_project_dir / "src" / "test.py"
        test_file.write_text("def test_func(): pass")

        # Enable startup reindexing in config
        repository_config_manager._global_config.repository.indexing.startup_enabled = (
            True
        )

        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.initialize_db_dependencies()

        # Index the file first
        service.index_files_batch([test_file])

        # Modify the file
        test_file.write_text("def test_func():\n    return 42")

        # Force mtime to change by setting it to the past
        past_time = time.time() + 10
        os.utime(test_file, (past_time, past_time))

        # Run startup re-indexing
        tracked_files = service.file_scope.get_tracked_files()
        _, filtered_files = service.file_filter.filter_files(tracked_files)
        stats = service.reindex_on_startup(filtered_files)

        # Should detect the modified file
        assert stats["updated"] >= 1

        service.close()

    def test_agent_edit_reindexing(self, temp_project_dir, repository_config_manager):
        """Test agent edit re-indexing functionality."""
        test_file = temp_project_dir / "src" / "agent_edit.py"
        test_file.write_text("def agent_func(): pass")

        # Enable agent edit reindexing in config
        repository_config_manager._global_config.repository.indexing.agent_edit_enabled = True

        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.initialize_db_dependencies()

        # Index the file first
        service.index_files_batch([test_file])

        # Simulate agent edit by modifying the file
        test_file.write_text("def agent_func():\n    return 'edited'")

        # Re-index the agent edit
        result = service.reindex_agent_edit(test_file)

        assert result is True

        service.close()

    def test_reindexing_with_file_filter(
        self, temp_project_dir, repository_config_manager
    ):
        """Test re-indexing with file filtering."""
        repository_config_manager._global_config.repository.indexing.startup_enabled = (
            True
        )

        # Create Python file
        py_file = temp_project_dir / "src" / "test.py"
        py_file.write_text("def test(): pass")

        # Create unsupported file
        txt_file = temp_project_dir / "src" / "test.txt"
        txt_file.write_text("content")

        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.initialize_db_dependencies()

        # Index Python file
        service.index_files_batch([py_file])

        # Modify both files
        py_file.write_text("def test():\n    return 1")
        txt_file.write_text("modified content")

        # Force mtime to change by setting it to the past
        past_time = time.time() + 10
        os.utime(py_file, (past_time, past_time))
        os.utime(txt_file, (past_time, past_time))

        # Run startup re-indexing
        tracked_files = service.file_scope.get_tracked_files()
        _, filtered_files = service.file_filter.filter_files(tracked_files)
        stats = service.reindex_on_startup(filtered_files)

        # Should only re-index the Python file
        assert stats["updated"] >= 1

        service.close()

    def test_file_monitoring_lifecycle(
        self, temp_project_dir, repository_config_manager
    ):
        """Test file monitoring service lifecycle."""
        # Enable file monitoring in config
        repository_config_manager._global_config.repository.file_monitoring.enabled = (
            True
        )

        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.ensure_initialized()

        # Start monitoring
        service.start_file_monitoring()
        assert service.file_monitoring_service.is_running()

        # Stop monitoring
        service.stop_file_monitoring()
        assert not service.file_monitoring_service.is_running()

        service.close()

    def test_file_monitoring_disabled(
        self, temp_project_dir, repository_config_manager
    ):
        """Test service with file monitoring disabled."""
        # File monitoring is already disabled in the fixture
        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.ensure_initialized()

        assert service.file_monitoring_service is None

        service.close()

    def test_file_change_callback_registration(
        self, temp_project_dir, repository_config_manager
    ):
        """Test registering file change callbacks."""
        # Enable file monitoring in config
        repository_config_manager._global_config.repository.file_monitoring.enabled = (
            True
        )

        service = RepositoryContextService(
            target_project_root=temp_project_dir,
            config_manager=repository_config_manager,
        )
        # Initialize repository service
        service.ensure_initialized()

        callback_called = []

        def test_callback(event):
            callback_called.append(event)

        service.register_file_change_callback(test_callback)

        # Create a file to trigger monitoring (in real scenario)
        test_file = temp_project_dir / "src" / "callback_test.py"
        test_file.write_text("def test(): pass")

        # Note: In a real test, we'd wait for the file monitoring to detect the change
        # For unit testing, we just verify the callback is registered
        assert (
            test_callback
            in service.file_monitoring_service.notification_emitter._callbacks
        )

        service.close()
