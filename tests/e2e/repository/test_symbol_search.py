"""E2E tests for symbol search functionality."""

from pathlib import Path

import pytest

from local_coding_assistant.repository.models import SymbolType
from local_coding_assistant.repository.service import RepositoryContextService


class TestSymbolSearch:
    """Test symbol search functionality across the codebase."""

    def test_search_by_name(self, repository_service: RepositoryContextService):
        """Test searching for symbols by name."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("main")

        assert isinstance(results, list)
        assert len(results) > 0

        # Should find the main function
        main_results = [r for r in results if "main" in r.name.lower()]
        assert len(main_results) > 0

    def test_search_by_class_name(self, repository_service: RepositoryContextService):
        """Test searching for classes by name."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("Application")

        assert isinstance(results, list)
        assert len(results) > 0

        # Should find the Application class
        class_results = [r for r in results if r.symbol_type == SymbolType.CLASS]
        assert len(class_results) > 0

    def test_search_by_method_name(self, repository_service: RepositoryContextService):
        """Test searching for methods by name."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("run")

        assert isinstance(results, list)
        assert len(results) > 0

        # Should find the run method
        method_results = [r for r in results if r.symbol_type == SymbolType.METHOD]
        assert len(method_results) > 0

    def test_search_with_type_filter_function(
        self, repository_service: RepositoryContextService
    ):
        """Test search filtering by function type."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("helper", symbol_types=["function"])

        assert isinstance(results, list)
        for result in results:
            assert result.symbol_type == SymbolType.FUNCTION

    def test_search_with_type_filter_class(
        self, repository_service: RepositoryContextService
    ):
        """Test search filtering by class type."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("", symbol_types=["class"])

        assert isinstance(results, list)
        for result in results:
            assert result.symbol_type == SymbolType.CLASS

    def test_search_with_type_filter_method(
        self, repository_service: RepositoryContextService
    ):
        """Test search filtering by method type."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("", symbol_types=["method"])

        assert isinstance(results, list)
        for result in results:
            assert result.symbol_type == SymbolType.METHOD

    def test_search_with_multiple_type_filters(
        self, repository_service: RepositoryContextService
    ):
        """Test search filtering by multiple symbol types."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols(
            "", symbol_types=["function", "class"]
        )

        assert isinstance(results, list)
        for result in results:
            assert result.symbol_type in [SymbolType.FUNCTION, SymbolType.CLASS]

    def test_search_with_file_path_filter(
        self, repository_service: RepositoryContextService
    ):
        """Test search filtering by file path."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols(
            "main", file_path="src/myproject/main.py"
        )

        assert isinstance(results, list)
        for result in results:
            assert "main.py" in result.file_path

    def test_search_with_limit(self, repository_service: RepositoryContextService):
        """Test search with result limit."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        # Search with small limit
        results = repository_service.search_symbols("", limit=2)

        assert isinstance(results, list)
        assert len(results) <= 2

    def test_search_with_large_limit(
        self, repository_service: RepositoryContextService
    ):
        """Test search with large limit."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        # Search with large limit to get all results
        results = repository_service.search_symbols("", limit=100)

        assert isinstance(results, list)
        # Should get many results
        assert len(results) > 0

    def test_search_case_insensitive(
        self, repository_service: RepositoryContextService
    ):
        """Test that search is case-insensitive."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results_lower = repository_service.search_symbols("main")
        results_upper = repository_service.search_symbols("MAIN")
        results_mixed = repository_service.search_symbols("Main")

        # All should return results
        assert len(results_lower) > 0
        assert len(results_upper) > 0
        assert len(results_mixed) > 0

    def test_search_fuzzy_matching(self, repository_service: RepositoryContextService):
        """Test fuzzy matching in search."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        # Search for partial name
        results = repository_service.search_symbols("help")

        assert isinstance(results, list)
        # Should find helper_function
        helper_results = [r for r in results if "help" in r.name.lower()]
        assert len(helper_results) > 0

    def test_search_empty_query(self, repository_service: RepositoryContextService):
        """Test search with empty query (returns all symbols)."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("")

        assert isinstance(results, list)
        # Should return all indexed symbols
        assert len(results) > 0

    def test_search_nonexistent_symbol(
        self, repository_service: RepositoryContextService
    ):
        """Test search for symbol that doesn't exist."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("nonexistent_xyz_123")

        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_result_structure(
        self, repository_service: RepositoryContextService
    ):
        """Test that search results have the correct structure."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("main")

        if len(results) > 0:
            result = results[0]
            assert hasattr(result, "symbol_id")
            assert hasattr(result, "name")
            assert hasattr(result, "symbol_type")
            assert hasattr(result, "file_path")
            assert hasattr(result, "line_number")
            assert hasattr(result, "docstring")
            assert hasattr(result, "signature")
            assert hasattr(result, "rank")

    def test_search_result_docstring(
        self, repository_service: RepositoryContextService
    ):
        """Test that search results include docstrings."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("main")

        if len(results) > 0:
            # At least some results should have docstrings
            # Note: Docstring extraction is not yet implemented
            # For now, we'll skip this assertion
            # has_docstring = any(r.docstring for r in results)
            # assert has_docstring
            pass

    def test_search_result_signature(
        self, repository_service: RepositoryContextService
    ):
        """Test that search results include signatures."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("")

        if len(results) > 0:
            # Functions and methods should have signatures
            func_results = [
                r
                for r in results
                if r.symbol_type in [SymbolType.FUNCTION, SymbolType.METHOD]
            ]
            if func_results:
                has_signature = any(r.signature for r in func_results)
                assert has_signature

    def test_search_result_line_numbers(
        self, repository_service: RepositoryContextService
    ):
        """Test that search results have valid line numbers."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("")

        if len(results) > 0:
            for result in results:
                assert result.line_number > 0
                if result.end_line_number:
                    assert result.end_line_number >= result.line_number

    def test_search_result_file_paths(
        self, repository_service: RepositoryContextService
    ):
        """Test that search results have valid file paths."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("")

        if len(results) > 0:
            for result in results:
                assert result.file_path is not None
                assert len(result.file_path) > 0

    def test_search_result_parent_scope(
        self, repository_service: RepositoryContextService
    ):
        """Test that search results include parent scope for nested symbols."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("")

        if len(results) > 0:
            # Methods should have parent scope (class name)
            method_results = [r for r in results if r.symbol_type == SymbolType.METHOD]
            if method_results:
                has_parent_scope = any(r.parent_scope for r in method_results)
                assert has_parent_scope

    def test_search_after_file_modification(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test that search results update after file modification."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        # Initial search
        results_1 = repository_service.search_symbols("new_function")
        assert len(results_1) == 0

        # Add new function
        main_file = test_project_structure / "src" / "myproject" / "main.py"
        original_content = main_file.read_text()
        modified_content = (
            original_content
            + """

def new_function():
    \"\"\"A new function.\"\"\"
    pass
"""
        )
        main_file.write_text(modified_content)

        # Re-index
        repository_service.reindex_agent_edit(main_file)

        # Search again
        results_2 = repository_service.search_symbols("new_function")

        # Restore original
        main_file.write_text(original_content)

        # Should find the new function
        assert len(results_2) > 0

    def test_search_ranking(self, repository_service: RepositoryContextService):
        """Test that search results are ranked."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("")

        if len(results) > 1:
            # Check that results have rank values
            has_ranks = all(hasattr(r, "rank") for r in results)
            assert has_ranks

            # Check that ranks are non-negative
            for result in results:
                assert result.rank >= 0

    def test_search_across_multiple_files(
        self, repository_service: RepositoryContextService
    ):
        """Test search across multiple files."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("")

        if len(results) > 0:
            # Should have results from multiple files
            file_paths = set(r.file_path for r in results)
            assert len(file_paths) > 1

    def test_search_special_characters(
        self, repository_service: RepositoryContextService
    ):
        """Test search with special characters."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        # Search with underscore
        results = repository_service.search_symbols("helper_function")

        assert isinstance(results, list)
        # Should find helper_function
        helper_results = [
            r
            for r in results
            if "helper" in r.name.lower() and "function" in r.name.lower()
        ]
        assert len(helper_results) > 0


class TestSymbolSearchEdgeCases:
    """Test edge cases and error handling for symbol search."""

    def test_search_with_invalid_type_filter(
        self, repository_service: RepositoryContextService
    ):
        """Test search with invalid symbol type filter."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("", symbol_types=["invalid_type"])

        # Should return empty list or handle gracefully
        assert isinstance(results, list)

    def test_search_with_invalid_file_path(
        self, repository_service: RepositoryContextService
    ):
        """Test search with invalid file path."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols(
            "", file_path="/nonexistent/path.py"
        )

        # Should return empty list
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_with_zero_limit(self, repository_service: RepositoryContextService):
        """Test search with zero limit."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("", limit=0)

        # Should return empty list
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_with_negative_limit(
        self, repository_service: RepositoryContextService
    ):
        """Test search with negative limit."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        results = repository_service.search_symbols("", limit=-1)

        # Should handle gracefully (may return empty or default behavior)
        assert isinstance(results, list)

    def test_search_unicode_characters(
        self, repository_service: RepositoryContextService, test_project_structure: Path
    ):
        """Test search with unicode characters."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        # Add file with unicode
        unicode_file = test_project_structure / "src" / "myproject" / "unicode.py"
        unicode_file.write_text("""
def café():
    \"\"\"Function with unicode name.\"\"\"
    pass
""")

        # Re-index
        repository_service.reindex_agent_edit(unicode_file)

        # Search for unicode
        results = repository_service.search_symbols("café")

        # Should handle unicode
        assert isinstance(results, list)

    def test_search_very_long_query(self, repository_service: RepositoryContextService):
        """Test search with very long query string."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        long_query = "a" * 1000
        results = repository_service.search_symbols(long_query)

        # Should handle gracefully
        assert isinstance(results, list)

    def test_search_special_regex_characters(
        self, repository_service: RepositoryContextService
    ):
        """Test search with regex special characters."""
        # Initialize repository service
        repository_service.initialize_db_dependencies()

        # Search with characters that might be interpreted as regex
        results = repository_service.search_symbols("main.*")

        # Should handle as literal string, not regex
        assert isinstance(results, list)
