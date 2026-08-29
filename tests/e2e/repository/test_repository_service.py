"""E2E tests for RepositoryContextService full workflow."""

import time
from pathlib import Path

import pytest

from local_coding_assistant.repository.models import ProjectInfo, SymbolType
from local_coding_assistant.repository.service import RepositoryContextService


class TestRepositoryServiceWorkflow:
    """Test the complete workflow of RepositoryContextService."""

    def test_service_initialization(self, repository_service: RepositoryContextService):
        """Test that the service initializes correctly."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        assert repository_service is not None
        assert repository_service.target_project_root.exists()
        assert repository_service.database is not None
        assert repository_service.ast_parser is not None
        assert repository_service.repo_map_builder is not None

    def test_get_repo_context(self, repository_service: RepositoryContextService):
        """Test getting repository context including metadata and repo map."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        project_info, repo_map_data = repository_service.get_repo_context()

        # Verify project info
        assert isinstance(project_info, ProjectInfo)
        # project name may be None for some projects
        # package_manager may be None depending on config detection

        # Verify repo map data
        # project_name may be None for some projects
        assert repo_map_data is not None
        assert repo_map_data.language_distribution is not None
        assert len(repo_map_data.language_distribution) > 0
        assert repo_map_data.grouped_symbols is not None

    def test_build_context_string(self, repository_service: RepositoryContextService):
        """Test building the full context string."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        project_info, repo_map_data = repository_service.get_repo_context()

        assert repo_map_data is not None

        context_str = RepositoryContextService.build_context_string(
            project_info, repo_map_data
        )

        assert context_str is not None
        assert len(context_str) > 0
        assert "Project Metadata:" in context_str or "Project:" in context_str
        # Check for project name in context
        if project_info.name:
            assert project_info.name in context_str

    def test_get_project_info(self, repository_service: RepositoryContextService):
        """Test getting project metadata."""
        project_info = repository_service.get_project_info()

        assert isinstance(project_info, ProjectInfo)
        # Should have extracted from pyproject.toml
        assert project_info.name is not None
        # package_manager may be None depending on config detection

    def test_get_project_info_force_refresh(
        self, repository_service: RepositoryContextService
    ):
        """Test force refresh of project info."""
        project_info_1 = repository_service.get_project_info(force_refresh=False)
        project_info_2 = repository_service.get_project_info(force_refresh=True)

        # Both should return valid project info
        assert project_info_1.name is not None
        assert project_info_2.name is not None
        assert project_info_1.name == project_info_2.name

    def test_get_repo_map_data(self, repository_service: RepositoryContextService):
        """Test getting repository map as structured data."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        repo_map_data = repository_service.get_repo_map_data()

        # project_name may be None if not detected
        assert repo_map_data is not None
        assert repo_map_data.language_distribution is not None
        assert isinstance(repo_map_data.grouped_symbols, dict)
        # Should have symbols from the test project
        assert len(repo_map_data.grouped_symbols) > 0

    def test_search_symbols(self, repository_service: RepositoryContextService):
        """Test symbol search functionality."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        # Search for a function that should exist
        results = repository_service.search_symbols("main")

        assert isinstance(results, list)
        # Should find at least the main function
        assert len(results) > 0

        # Verify result structure
        first_result = results[0]
        assert hasattr(first_result, "name")
        assert hasattr(first_result, "symbol_type")
        assert hasattr(first_result, "file_path")

    def test_search_symbols_with_type_filter(
        self, repository_service: RepositoryContextService
    ):
        """Test symbol search with type filter."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        # Search for functions only
        results = repository_service.search_symbols("main", symbol_types=["function"])

        assert isinstance(results, list)
        for result in results:
            assert result.symbol_type == SymbolType.FUNCTION

    def test_search_symbols_with_file_filter(
        self, repository_service: RepositoryContextService
    ):
        """Test symbol search with file path filter."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        # Search in a specific file
        results = repository_service.search_symbols(
            "main", file_path="src/myproject/main.py"
        )

        assert isinstance(results, list)
        for result in results:
            assert "main.py" in result.file_path

    def test_search_symbols_limit(self, repository_service: RepositoryContextService):
        """Test symbol search with result limit."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        # Search with a small limit
        results = repository_service.search_symbols("helper", limit=1)

        assert isinstance(results, list)
        assert len(results) <= 1

    def test_index_files_batch(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch file indexing."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        # Get all Python files
        python_files = list(test_project_structure.rglob("*.py"))

        # Index them in batch
        count = repository_service.index_files_batch(python_files)

        assert count > 0
        assert count <= len(python_files)

    def test_index_files_batch_with_chunking(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch file indexing with small chunk size to force chunking."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        python_files = list(test_project_structure.rglob("*.py"))

        # Use a very small chunk size to force multiple chunks
        count = repository_service.index_files_batch(
            python_files,
            max_cumulative_size_kb=1,  # 1KB chunks
        )

        assert count > 0

    def test_context_manager(self, repository_service: RepositoryContextService):
        """Test using the service as a context manager."""
        with repository_service as service:
            assert service is not None
            project_info = service.get_project_info()
            assert project_info.name is not None

        # Service should be closed after exiting context
        # (Can't easily test this without accessing private state)

    def test_service_close(self, repository_service: RepositoryContextService):
        """Test closing the service."""
        # Get some data first to ensure service is working
        repository_service.get_project_info()

        # Close the service
        repository_service.close()

        # Should not raise an error if called again
        repository_service.close()

    def test_ensure_initialized(self, repository_service: RepositoryContextService):
        """Test ensure_initialized method."""
        # Should not raise an error
        repository_service.ensure_initialized()

        # Can call multiple times
        repository_service.ensure_initialized()

    def test_startup_reindexing(self, repository_service: RepositoryContextService):
        """Test startup re-indexing."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        tracked_files = repository_service.file_scope.get_tracked_files()
        _, filtered_files = repository_service.file_filter.filter_files(tracked_files)
        stats = repository_service.reindex_on_startup(filtered_files)

        assert isinstance(stats, dict)
        assert "added" in stats or "updated" in stats or "deleted" in stats

    def test_agent_edit_reindex(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test re-indexing after agent edit."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        main_file = test_project_structure / "src" / "myproject" / "main.py"

        # Re-index the file
        result = repository_service.reindex_agent_edit(main_file)

        assert isinstance(result, bool)

    def test_full_workflow_integration(
        self, repository_service: RepositoryContextService
    ):
        """Test the complete workflow from initialization to context generation."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        # 1. Get project info
        project_info = repository_service.get_project_info()
        assert project_info.name is not None

        # 2. Get repo map
        repo_map_data = repository_service.get_repo_map_data()
        # project_name may be None if not detected from git

        assert repo_map_data is not None

        # 3. Search for symbols
        results = repository_service.search_symbols("main")
        assert len(results) > 0

        # 4. Build full context
        context_str = RepositoryContextService.build_context_string(
            project_info, repo_map_data
        )
        assert len(context_str) > 0

        # 5. Verify context contains expected information
        assert project_info.name in context_str
        if repo_map_data.language_distribution:
            for lang, pct in repo_map_data.language_distribution.items():
                assert lang in context_str or str(pct) in context_str


class TestRepositoryServiceEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_search_query(self, repository_service: RepositoryContextService):
        """Test search with empty query."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("")

        # Should return empty list or handle gracefully
        assert isinstance(results, list)

    def test_search_nonexistent_symbol(
        self, repository_service: RepositoryContextService
    ):
        """Test search for symbol that doesn't exist."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("nonexistent_function_xyz_123")

        # Should return empty list
        assert isinstance(results, list)
        assert len(results) == 0

    def test_index_empty_file_list(self, repository_service: RepositoryContextService):
        """Test indexing with empty file list."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        count = repository_service.index_files_batch([])

        assert count == 0

    def test_multiple_service_instances(
        self, test_project_structure: Path, config_manager_with_repo
    ):
        """Test creating multiple service instances."""
        service1 = RepositoryContextService(
            target_project_root=test_project_structure,
            config_manager=config_manager_with_repo,
        )

        service2 = RepositoryContextService(
            target_project_root=test_project_structure,
            config_manager=config_manager_with_repo,
        )

        # Both should work independently
        info1 = service1.get_project_info()
        info2 = service2.get_project_info()

        assert info1.name == info2.name

        # Cleanup
        service1.close()
        service2.close()
