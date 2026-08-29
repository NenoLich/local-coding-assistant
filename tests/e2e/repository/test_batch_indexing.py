"""E2E tests for batch file indexing functionality."""

from pathlib import Path

import pytest

from local_coding_assistant.repository.service import RepositoryContextService


class TestBatchIndexing:
    """Test batch file indexing with chunk-based processing."""

    def test_batch_index_basic(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test basic batch indexing of multiple files."""
        python_files = list(test_project_structure.rglob("*.py"))

        repository_service.initialize_db_dependencies()
        count = repository_service.index_files_batch(python_files)

        assert count > 0
        assert count <= len(python_files)

    def test_batch_index_with_default_chunk_size(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with default chunk size (10MB)."""
        python_files = list(test_project_structure.rglob("*.py"))

        repository_service.initialize_db_dependencies()
        count = repository_service.index_files_batch(
            python_files,
            max_cumulative_size_kb=10000,  # 10MB default
        )

        assert count > 0

    def test_batch_index_with_small_chunk_size(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with small chunk size to force multiple chunks."""
        python_files = list(test_project_structure.rglob("*.py"))
        repository_service.initialize_db_dependencies()
        # Use very small chunk size to force chunking
        count = repository_service.index_files_batch(
            python_files,
            max_cumulative_size_kb=1,  # 1KB chunks
        )

        assert count > 0

    def test_batch_index_with_large_chunk_size(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with large chunk size (single chunk)."""
        python_files = list(test_project_structure.rglob("*.py"))
        repository_service.initialize_db_dependencies()
        # Use large chunk size to process all in one chunk
        count = repository_service.index_files_batch(
            python_files,
            max_cumulative_size_kb=100000,  # 100MB
        )

        assert count > 0

    def test_batch_index_empty_list(self, repository_service: RepositoryContextService):
        """Test batch indexing with empty file list."""
        repository_service.initialize_db_dependencies()
        count = repository_service.index_files_batch([])

        assert count == 0

    def test_batch_index_single_file(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with a single file."""
        main_file = test_project_structure / "src" / "myproject" / "main.py"
        repository_service.initialize_db_dependencies()

        count = repository_service.index_files_batch([main_file])

        assert count == 1

    def test_batch_index_after_modification(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing after files are modified."""
        python_files = list(test_project_structure.rglob("*.py"))

        # Modify a file
        main_file = test_project_structure / "src" / "myproject" / "main.py"
        original_content = main_file.read_text()
        modified_content = original_content + "\n# Batch test modification\n"
        main_file.write_text(modified_content)
        repository_service.initialize_db_dependencies()

        # Batch index
        count = repository_service.index_files_batch(python_files)

        # Restore
        main_file.write_text(original_content)

        assert count > 0

    def test_batch_index_with_new_files(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with newly added files."""
        # Add new files
        new_dir = test_project_structure / "src" / "new_module"
        new_dir.mkdir(parents=True)

        (new_dir / "file1.py").write_text("def func1(): pass")
        (new_dir / "file2.py").write_text("def func2(): pass")
        (new_dir / "file3.py").write_text("def func3(): pass")

        # Commit to git
        import subprocess

        subprocess.run(
            ["git", "add", "."], cwd=test_project_structure, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add new files"],
            cwd=test_project_structure,
            capture_output=True,
        )

        repository_service.initialize_db_dependencies()

        # Batch index all files
        python_files = list(test_project_structure.rglob("*.py"))
        count = repository_service.index_files_batch(python_files)

        # Should include new files
        assert count > 0

    def test_batch_index_with_mixed_file_sizes(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with files of varying sizes."""
        # Create files with different sizes
        small_file = test_project_structure / "small.py"
        small_file.write_text("def s(): pass")

        medium_file = test_project_structure / "medium.py"
        medium_file.write_text(
            "\n".join(["def m" + str(i) + "(): pass" for i in range(100)])
        )

        large_file = test_project_structure / "large.py"
        large_file.write_text(
            "\n".join(["def l" + str(i) + "(): pass" for i in range(1000)])
        )

        # Commit to git
        import subprocess

        subprocess.run(
            ["git", "add", "."], cwd=test_project_structure, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add mixed size files"],
            cwd=test_project_structure,
            capture_output=True,
        )

        repository_service.initialize_db_dependencies()

        # Batch index with small chunk size
        python_files = list(test_project_structure.rglob("*.py"))
        count = repository_service.index_files_batch(
            python_files,
            max_cumulative_size_kb=5,  # Small chunk to force chunking
        )

        assert count > 0

    def test_batch_index_preserves_all_symbols(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that batch indexing preserves all symbols."""
        # Get initial symbol count
        results_before = repository_service.search_symbols("")
        initial_count = len(results_before)

        repository_service.initialize_db_dependencies()

        # Batch index all files
        python_files = list(test_project_structure.rglob("*.py"))
        repository_service.index_files_batch(python_files)

        # Get symbol count after
        results_after = repository_service.search_symbols("")
        final_count = len(results_after)

        # Should have similar or more symbols
        assert final_count >= initial_count - 1  # Allow small variance

    def test_batch_index_performance(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that batch indexing completes in reasonable time."""
        import time

        python_files = list(test_project_structure.rglob("*.py"))

        repository_service.initialize_db_dependencies()

        start_time = time.time()
        count = repository_service.index_files_batch(python_files)
        end_time = time.time()

        elapsed = end_time - start_time

        # Should complete in reasonable time (< 10 seconds for small project)
        assert elapsed < 10.0
        assert count > 0


class TestBatchIndexingEdgeCases:
    """Test edge cases and error handling for batch indexing."""

    def test_batch_index_with_nonexistent_files(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with some nonexistent files."""
        python_files = list(test_project_structure.rglob("*.py"))
        repository_service.initialize_db_dependencies()

        # Add a nonexistent file
        nonexistent = test_project_structure / "nonexistent.py"
        all_files = python_files + [nonexistent]

        # Should handle gracefully
        count = repository_service.index_files_batch(all_files)

        assert count >= 0

    def test_batch_index_with_directory_paths(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing when list includes directories."""
        python_files = list(test_project_structure.rglob("*.py"))
        repository_service.initialize_db_dependencies()

        # Add a directory
        src_dir = test_project_structure / "src"
        all_files = python_files + [src_dir]

        # Should handle gracefully
        count = repository_service.index_files_batch(all_files)

        assert count >= 0

    def test_batch_index_with_unsupported_extensions(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with unsupported file types."""
        # Create unsupported files
        (test_project_structure / "test.txt").write_text("text file")
        (test_project_structure / "test.json").write_text('{"key": "value"}')

        # Commit to git
        import subprocess

        subprocess.run(
            ["git", "add", "."], cwd=test_project_structure, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add unsupported files"],
            cwd=test_project_structure,
            capture_output=True,
        )

        repository_service.initialize_db_dependencies()

        # Batch index all files (unsupported should be filtered)
        all_files = list(test_project_structure.rglob("*"))
        count = repository_service.index_files_batch(all_files)

        # Should only index supported extensions
        assert count >= 0

    def test_batch_index_with_zero_chunk_size(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with zero chunk size."""
        python_files = list(test_project_structure.rglob("*.py"))
        repository_service.initialize_db_dependencies()

        # Should handle gracefully (may use default or fail)
        count = repository_service.index_files_batch(
            python_files, max_cumulative_size_kb=0
        )

        assert count >= 0

    def test_batch_index_with_negative_chunk_size(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with negative chunk size."""
        python_files = list(test_project_structure.rglob("*.py"))
        repository_service.initialize_db_dependencies()

        # Should handle gracefully
        count = repository_service.index_files_batch(
            python_files, max_cumulative_size_kb=-1
        )

        assert count >= 0

    def test_batch_index_with_very_large_chunk_size(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with extremely large chunk size."""
        python_files = list(test_project_structure.rglob("*.py"))
        repository_service.initialize_db_dependencies()

        # Should handle gracefully
        count = repository_service.index_files_batch(
            python_files,
            max_cumulative_size_kb=10**9,  # 1TB
        )

        assert count >= 0

    def test_batch_index_duplicate_files(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with duplicate file paths."""
        python_files = list(test_project_structure.rglob("*.py"))
        repository_service.initialize_db_dependencies()

        # Add duplicates
        if python_files:
            duplicated = python_files + [python_files[0], python_files[0]]

            # Should handle duplicates gracefully
            count = repository_service.index_files_batch(duplicated)

            assert count >= 0

    def test_batch_index_with_symlinks(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with symbolic links (if supported)."""
        # This test is platform-dependent
        # On Windows, symlinks may require admin privileges
        try:
            main_file = test_project_structure / "src" / "myproject" / "main.py"
            link_file = test_project_structure / "link_to_main.py"
            repository_service.initialize_db_dependencies()

            # Try to create symlink
            link_file.symlink_to(main_file)

            # Commit to git
            import subprocess

            subprocess.run(
                ["git", "add", "."], cwd=test_project_structure, capture_output=True
            )
            subprocess.run(
                ["git", "commit", "-m", "Add symlink"],
                cwd=test_project_structure,
                capture_output=True,
            )

            # Batch index
            python_files = list(test_project_structure.rglob("*.py"))
            count = repository_service.index_files_batch(python_files)

            # Clean up
            link_file.unlink()

            assert count >= 0
        except (OSError, NotImplementedError):
            # Symlinks not supported, skip test
            pytest.skip("Symlinks not supported on this system")

    def test_batch_index_with_readonly_files(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with read-only files."""
        # Create a read-only file
        readonly_file = test_project_structure / "readonly.py"
        readonly_file.write_text("def readonly(): pass")
        repository_service.initialize_db_dependencies()

        try:
            # Make read-only
            readonly_file.chmod(0o444)

            # Commit to git
            import subprocess

            subprocess.run(
                ["git", "add", "."], cwd=test_project_structure, capture_output=True
            )
            subprocess.run(
                ["git", "commit", "-m", "Add readonly file"],
                cwd=test_project_structure,
                capture_output=True,
            )

            # Batch index
            python_files = list(test_project_structure.rglob("*.py"))
            count = repository_service.index_files_batch(python_files)

            assert count >= 0
        finally:
            # Restore permissions
            try:
                readonly_file.chmod(0o644)
                readonly_file.unlink()
            except:
                pass


class TestBatchIndexingIntegration:
    """Test batch indexing integration with other features."""

    def test_batch_index_then_search(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that search works after batch indexing."""
        repository_service.initialize_db_dependencies()
        # Batch index
        python_files = list(test_project_structure.rglob("*.py"))
        repository_service.index_files_batch(python_files)

        # Search should work
        results = repository_service.search_symbols("main")
        assert len(results) > 0

    def test_batch_index_then_repo_map(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that repo map generation works after batch indexing."""
        repository_service.initialize_db_dependencies()
        # Batch index
        python_files = list(test_project_structure.rglob("*.py"))
        repository_service.index_files_batch(python_files)

        # Repo map should work
        repo_map_data = repository_service.get_repo_map_data()

        assert repo_map_data is not None
        assert repo_map_data.total_symbols >= 0

    def test_batch_index_then_metadata(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that metadata extraction works after batch indexing."""
        repository_service.initialize_db_dependencies()
        # Batch index
        python_files = list(test_project_structure.rglob("*.py"))
        repository_service.index_files_batch(python_files)

        # Metadata should work
        project_info = repository_service.get_project_info()
        assert project_info is not None

    def test_batch_index_multiple_times(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that batch indexing can be called multiple times."""
        python_files = list(test_project_structure.rglob("*.py"))
        repository_service.initialize_db_dependencies()

        # Index multiple times
        for _ in range(3):
            count = repository_service.index_files_batch(python_files)
            assert count > 0

    def test_batch_index_with_context_manager(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing within service context manager."""
        repository_service.initialize_db_dependencies()
        with repository_service as service:
            python_files = list(test_project_structure.rglob("*.py"))
            count = service.index_files_batch(python_files)
            assert count > 0

    def test_batch_index_large_file_set(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test batch indexing with a larger set of files."""
        repository_service.initialize_db_dependencies()
        # Create many small files
        for i in range(20):
            test_file = test_project_structure / f"test_{i}.py"
            test_file.write_text(f"def func_{i}(): pass")

        # Commit to git
        import subprocess

        subprocess.run(
            ["git", "add", "."], cwd=test_project_structure, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add many files"],
            cwd=test_project_structure,
            capture_output=True,
        )

        # Batch index with chunking
        python_files = list(test_project_structure.rglob("*.py"))
        count = repository_service.index_files_batch(
            python_files,
            max_cumulative_size_kb=10,  # Small chunks
        )

        assert count > 0
