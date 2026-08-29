#!/usr/bin/env python
"""Standalone script for manually testing repository context on arbitrary files.

This script allows you to test the full pipeline (ast_parser → repo_map → metadata_extractor)
on any files you place in the manual_test_files directory without needing to run pytest.

Usage:
    python scripts/test_repo_context_manual.py

The script will:
1. Scan tests/integration/repository/manual_test_files/ for files
2. Parse them with ASTParser
3. Index them into a temporary database
4. Build repo map with PageRank
5. Extract metadata
6. Display the final context string
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.file_filter import FileFilter
from local_coding_assistant.repository.file_reindexer import FileReindexer
from local_coding_assistant.repository.file_scope import FileScope
from local_coding_assistant.repository.metadata import (
    MetadataExtractor,
    MetadataFileDetector,
)
from local_coding_assistant.repository.repo_map import RepoMapBuilder
from local_coding_assistant.repository.service import RepositoryContextService

MANUAL_TEST_DIR = project_root / "src" / "local_coding_assistant"
# MANUAL_TEST_DIR = project_root / "tests" / "integration" / "repository" / "rust_test_files"

# Supported extensions for AST parsing
SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".rs",
    ".go",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}


def main():
    """Run the manual test pipeline."""
    start_time = time.time()
    parser = argparse.ArgumentParser()

    # action="store_true" makes these False by default, and True if typed
    parser.add_argument("--nodes", action="store_true")
    parser.add_argument("--calls", action="store_true")
    parser.add_argument("--ranks", action="store_true")

    args = parser.parse_args()
    # Check if manual test directory exists
    if not MANUAL_TEST_DIR.exists():
        print(f"❌ Manual test directory not found: {MANUAL_TEST_DIR}")
        print("Please create it and add some test files.")
        return 1

    # Find all files
    # test_files = list(MANUAL_TEST_DIR.rglob("*"))
    # test_files = [f for f in test_files if f.is_file()]
    file_scope = FileScope(MANUAL_TEST_DIR)
    tracked_files = file_scope._get_files_by_walk()
    if not tracked_files:
        print(f"❌ No test files found in {MANUAL_TEST_DIR}")
        print("Please add some files to test (e.g., .py, .toml, .json, etc.)")
        return 1

    file_filter = FileFilter(supported_extensions=SUPPORTED_EXTENSIONS)
    # Separate code files (for AST parsing) and config files (for metadata only)
    _, code_files = file_filter.filter_files(
        list(tracked_files), apply_test_file_filter=False
    )
    config_files = MetadataFileDetector.find_config_files(
        list(tracked_files), target_project_root=MANUAL_TEST_DIR
    )

    print(f"\n{'=' * 80}")
    print(f"Found {len(tracked_files)} total files:")
    print(f"  - Code files (AST parsing): {len(code_files)}")
    print(f"  - Config files (metadata only): {len(config_files)}")
    print(f"{'=' * 80}\n")

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_repo.db"
        database = RepositoryDatabase(db_path)

        repo_map_builder = RepoMapBuilder(database)

        file_reindexer = FileReindexer(database, file_filter, 4)
        indexed_files, _ = file_reindexer.parse_and_reindex_files_parallel(
            code_files, max_cumulative_size_kb=10000
        )
        print(f"Indexed {indexed_files} files")

        # Show config files (metadata only)
        if config_files:
            print(f"\n{'─' * 80}")
            print("📋 Config Files (metadata extraction only):")
            print(f"{'─' * 80}")
            for file_path in config_files:
                print(f"  - {file_path.relative_to(MANUAL_TEST_DIR)}")

        # Build repo map
        print(f"\n{'=' * 80}")
        print("[Repo Map] Building repo map with PageRank...")
        print(f"{'=' * 80}\n")

        repo_map_data = repo_map_builder.build()

        print(f"Project name: {repo_map_data.project_name}")
        print(f"Language distribution: {repo_map_data.language_distribution}")
        total_symbols = sum(
            len(symbols) for symbols in repo_map_data.grouped_symbols.values()
        )
        print(f"Total symbols (after ranking): {total_symbols}")

        # Extract metadata
        print(f"\n{'=' * 80}")
        print("[Metadata] Extracting project metadata...")
        print(f"{'=' * 80}\n")

        # Metadata extractor expects absolute paths
        metadata_extractor = MetadataExtractor(file_filter)
        project_info = metadata_extractor.extract(
            tracked_files, MANUAL_TEST_DIR.resolve(), force_refresh=False
        )

        print(f"Project name: {project_info.name}")
        print(f"Package manager: {project_info.package_manager}")
        print(f"Core frameworks: {project_info.core_frameworks}")
        print(f"Dependencies: {project_info.clean_dependencies}")
        print(f"Manifest files: {project_info.manifest_files}")

        # Build full context manually (since RepositoryContextService uses FileScope which requires git)
        print(f"\n{'=' * 80}")
        print("[Context Builder] Building full context string...")
        print(f"{'=' * 80}\n")

        context_str = RepositoryContextService.build_context_string(
            project_info, repo_map_data, args.ranks
        )

        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 29 + "FINAL CONTEXT OUTPUT" + " " * 29 + "║")
        print("╚" + "═" * 78 + "╝\n")
        print(context_str)
        print("\n" + "═" * 80 + "\n")
        print(f"\n{'=' * 80}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
