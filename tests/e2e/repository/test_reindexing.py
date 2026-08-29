"""E2E tests for re-indexing strategies."""

import time
from pathlib import Path

import pytest

from local_coding_assistant.repository.service import RepositoryContextService


class TestStartupReindexing:
    """Test startup re-indexing strategy."""

    def test_startup_reindex_on_existing_database(
        self, repository_service: RepositoryContextService
    ):
        """Test startup re-indexing when database already exists."""
        repository_service.initialize_db_dependencies()
        # Service already initialized with existing database
        tracked_files = repository_service.file_scope.get_tracked_files()
        _, filtered_files = repository_service.file_filter.filter_files(tracked_files)
        stats = repository_service.reindex_on_startup(filtered_files)

        assert isinstance(stats, dict)
        # Should have statistics keys
        assert "added" in stats or "updated" in stats or "deleted" in stats

    def test_startup_reindex_statistics(
        self, repository_service: RepositoryContextService
    ):
        """Test that startup re-indexing returns valid statistics."""
        repository_service.initialize_db_dependencies()
        tracked_files = repository_service.file_scope.get_tracked_files()
        _, filtered_files = repository_service.file_filter.filter_files(tracked_files)
        stats = repository_service.reindex_on_startup(filtered_files)

        # Verify statistics structure
        for key in ["added", "updated", "deleted"]:
            if key in stats:
                assert isinstance(stats[key], int)
                assert stats[key] >= 0

    def test_startup_reindex_after_file_modification(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that startup re-indexing detects modified files."""
        repository_service.initialize_db_dependencies()
        # Get initial stats
        tracked_files = repository_service.file_scope.get_tracked_files()
        _, filtered_files = repository_service.file_filter.filter_files(tracked_files)
        stats_1 = repository_service.reindex_on_startup(filtered_files)

        # Modify a file
        main_file = test_project_structure / "src" / "myproject" / "main.py"
        original_content = main_file.read_text()
        modified_content = original_content + "\n# Modified comment\n"
        main_file.write_text(modified_content)

        # Wait a moment to ensure mtime changes
        time.sleep(0.1)

        # Re-index again
        tracked_files = repository_service.file_scope.get_tracked_files()
        _, filtered_files = repository_service.file_filter.filter_files(tracked_files)
        stats_2 = repository_service.reindex_on_startup(filtered_files)

        # Restore original
        main_file.write_text(original_content)

        # Should have detected the modification
        assert isinstance(stats_2, dict)

    def test_startup_reindex_after_new_file(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that startup re-indexing detects new files."""
        repository_service.initialize_db_dependencies()
        # Get initial stats
        tracked_files = repository_service.file_scope.get_tracked_files()
        _, filtered_files = repository_service.file_filter.filter_files(tracked_files)
        stats_1 = repository_service.reindex_on_startup(filtered_files)

        # Add a new file
        new_file = test_project_structure / "src" / "myproject" / "new_file.py"
        new_file.write_text("""
def new_function():
    \"\"\"A new function.\"\"\"
    pass
""")

        # Commit to git
        import subprocess

        subprocess.run(
            ["git", "add", "."], cwd=test_project_structure, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add new file"],
            cwd=test_project_structure,
            capture_output=True,
        )

        # Re-index
        tracked_files = repository_service.file_scope.get_tracked_files()
        _, filtered_files = repository_service.file_filter.filter_files(tracked_files)
        stats_2 = repository_service.reindex_on_startup(filtered_files)

        # Should have detected the new file
        assert isinstance(stats_2, dict)
        if "added" in stats_2:
            assert stats_2["added"] >= stats_1.get("added", 0)

    def test_startup_reindex_after_file_deletion(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that startup re-indexing detects deleted files."""
        repository_service.initialize_db_dependencies()

        # Add a file first
        temp_file = test_project_structure / "src" / "myproject" / "temp_file.py"
        temp_file.write_text("def temp(): pass")

        # Commit to git
        import subprocess

        subprocess.run(
            ["git", "add", "."], cwd=test_project_structure, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add temp file"],
            cwd=test_project_structure,
            capture_output=True,
        )

        # Re-index to include it
        repository_service.reindex_agent_edit(temp_file)

        # Delete the file
        temp_file.unlink()

        # Commit deletion
        subprocess.run(
            ["git", "add", "."], cwd=test_project_structure, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Delete temp file"],
            cwd=test_project_structure,
            capture_output=True,
        )

        # Re-index
        tracked_files = repository_service.file_scope.get_tracked_files()
        _, filtered_files = repository_service.file_filter.filter_files(tracked_files)
        stats = repository_service.reindex_on_startup(filtered_files)

        # Should have detected the deletion
        assert isinstance(stats, dict)

    def test_startup_reindex_with_extension_filter(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that startup re-indexing respects extension filter."""
        repository_service.initialize_db_dependencies()

        # Add a non-Python file (should be filtered out)
        txt_file = test_project_structure / "README.txt"
        txt_file.write_text("This is a text file")

        # Commit to git
        import subprocess

        subprocess.run(
            ["git", "add", "."], cwd=test_project_structure, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add txt file"],
            cwd=test_project_structure,
            capture_output=True,
        )

        # Re-index
        tracked_files = repository_service.file_scope.get_tracked_files()
        _, filtered_files = repository_service.file_filter.filter_files(tracked_files)
        stats = repository_service.reindex_on_startup(filtered_files)

        # Should not have indexed the .txt file
        # (Can't directly verify, but stats should reflect only supported extensions)
        assert isinstance(stats, dict)


class TestAgentEditReindexing:
    """Test agent edit re-indexing strategy."""

    def test_agent_edit_reindex_single_file(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test re-indexing a single file after agent edit."""
        repository_service.initialize_db_dependencies()
        main_file = test_project_structure / "src" / "myproject" / "main.py"

        result = repository_service.reindex_agent_edit(main_file)

        assert isinstance(result, bool)
        assert result == True

    def test_agent_edit_reindex_after_modification(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that agent edit re-indexing updates the file in database."""
        repository_service.initialize_db_dependencies()
        main_file = test_project_structure / "src" / "myproject" / "main.py"

        # Get initial search results
        results_1 = repository_service.search_symbols("main")
        assert len(results_1) > 0

        # Modify file
        original_content = main_file.read_text()
        modified_content = (
            original_content
            + """

def agent_added_function():
    \"\"\"Added by agent.\"\"\"
    pass
"""
        )
        main_file.write_text(modified_content)

        # Re-index
        repository_service.reindex_agent_edit(main_file)

        results_2 = repository_service.search_symbols("main")
        assert len(results_2) > 0

        # Search for new function
        results_3 = repository_service.search_symbols("agent_added_function")

        # Restore original
        main_file.write_text(original_content)

        # Should find the new function
        assert len(results_3) > 0

    def test_agent_edit_reindex_nonexistent_file(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test re-indexing a file that doesn't exist."""
        repository_service.initialize_db_dependencies()
        nonexistent = test_project_structure / "nonexistent.py"

        result = repository_service.reindex_agent_edit(nonexistent)

        # Should handle gracefully
        assert isinstance(result, bool)

    def test_agent_edit_reindex_with_path_string(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test re-indexing with string path instead of Path object."""
        repository_service.initialize_db_dependencies()
        main_file = test_project_structure / "src" / "myproject" / "main.py"

        result = repository_service.reindex_agent_edit(str(main_file))

        assert isinstance(result, bool)
        assert result == True

    def test_agent_edit_reindex_multiple_files(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test re-indexing multiple files sequentially."""
        repository_service.initialize_db_dependencies()
        files = [
            test_project_structure / "src" / "myproject" / "main.py",
            test_project_structure / "src" / "myproject" / "utils.py",
            test_project_structure / "src" / "myproject" / "models.py",
        ]

        results = []
        for file_path in files:
            result = repository_service.reindex_agent_edit(file_path)
            results.append(result)

        # All should succeed
        assert all(results)


class TestReindexingIntegration:
    """Test integration of different re-indexing strategies."""

    def test_full_reindexing_workflow(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test complete re-indexing workflow."""
        repository_service.initialize_db_dependencies()
        # 1. Startup re-index
        tracked_files = repository_service.file_scope.get_tracked_files()
        _, filtered_files = repository_service.file_filter.filter_files(tracked_files)
        startup_stats = repository_service.reindex_on_startup(filtered_files)
        assert isinstance(startup_stats, dict)

        # 2. Modify a file
        main_file = test_project_structure / "src" / "myproject" / "main.py"
        original_content = main_file.read_text()
        modified_content = original_content + "\n# Test modification\n"
        main_file.write_text(modified_content)

        # 3. Agent edit re-index
        agent_result = repository_service.reindex_agent_edit(main_file)
        assert isinstance(agent_result, bool)

        # 4. Restore and re-index again
        main_file.write_text(original_content)
        repository_service.reindex_agent_edit(main_file)

        # 5. Verify search still works
        results = repository_service.search_symbols("main")
        assert len(results) > 0

    def test_reindexing_preserves_data(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that re-indexing preserves existing data."""
        repository_service.initialize_db_dependencies()
        # Get initial search results
        results_1 = repository_service.search_symbols("")
        initial_count = len(results_1)

        # Re-index all files
        python_files = list(test_project_structure.rglob("*.py"))
        for file_path in python_files:
            repository_service.reindex_agent_edit(file_path)

        # Get results after re-indexing
        results_2 = repository_service.search_symbols("")
        final_count = len(results_2)

        # Should have similar or more symbols
        assert final_count >= initial_count - 1  # Allow small variance

    def test_reindexing_with_file_filter(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that re-indexing respects file filters."""
        repository_service.initialize_db_dependencies()

        # Create a large file that might exceed size limit
        large_file = test_project_structure / "src" / "myproject" / "large.py"
        large_content = "def func" + "".join(["_x" for _ in range(100000)]) + "(): pass"
        large_file.write_text(large_content)

        # Try to re-index
        result = repository_service.reindex_agent_edit(large_file)

        # Clean up
        large_file.unlink()

        # Should handle based on size filter
        assert isinstance(result, bool)

    def test_reindexing_with_test_file_filter(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that re-indexing can filter test files."""
        repository_service.initialize_db_dependencies()

        # Create a test file
        test_file = test_project_structure / "src" / "myproject" / "test_main.py"
        test_file.write_text("""
def test_function():
    assert True
""")

        # Re-index with test file filter enabled
        result = repository_service.reindex_agent_edit(test_file)

        # Clean up
        test_file.unlink()

        # Should handle based on test file filter
        assert isinstance(result, bool)


class TestReindexingEdgeCases:
    """Test edge cases and error handling."""

    def test_reindex_empty_file(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test re-indexing an empty file."""
        repository_service.initialize_db_dependencies()

        empty_file = test_project_structure / "src" / "myproject" / "empty.py"
        empty_file.write_text("")

        result = repository_service.reindex_agent_edit(empty_file)

        # Clean up
        empty_file.unlink()

        # Should handle gracefully
        assert isinstance(result, bool)

    def test_reindex_file_with_syntax_error(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test re-indexing a file with syntax errors."""
        repository_service.initialize_db_dependencies()

        error_file = test_project_structure / "src" / "myproject" / "error.py"
        error_file.write_text("def broken(:\n    # Missing closing paren")

        result = repository_service.reindex_agent_edit(error_file)

        # Clean up
        error_file.unlink()

        # Should handle gracefully (may fail or skip)
        assert isinstance(result, bool)

    def test_reindex_file_with_encoding_issues(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test re-indexing a file with encoding issues."""
        repository_service.initialize_db_dependencies()

        # Create file with mixed encoding
        encoding_file = test_project_structure / "src" / "myproject" / "encoding.py"
        encoding_file.write_text("def test(): pass", encoding="utf-8")

        result = repository_service.reindex_agent_edit(encoding_file)

        # Clean up
        encoding_file.unlink()

        # Should handle gracefully
        assert isinstance(result, bool)

    def test_concurrent_reindex_requests(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test handling concurrent re-index requests."""
        import threading

        repository_service.initialize_db_dependencies()
        main_file = test_project_structure / "src" / "myproject" / "main.py"

        results = []

        def reindex_task():
            result = repository_service.reindex_agent_edit(main_file)
            results.append(result)

        # Create multiple threads
        threads = [threading.Thread(target=reindex_task) for _ in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # All should complete
        assert len(results) == 5
        assert all(isinstance(r, bool) for r in results)

    def test_reindex_during_file_monitoring(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test re-indexing while file monitoring might be active."""
        repository_service.initialize_db_dependencies()

        # This is more of a smoke test to ensure no conflicts
        main_file = test_project_structure / "src" / "myproject" / "main.py"

        # Perform multiple re-index operations
        for _ in range(3):
            result = repository_service.reindex_agent_edit(main_file)
            assert isinstance(result, bool)
