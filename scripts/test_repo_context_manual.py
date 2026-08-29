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

from local_coding_assistant.repository.ast_parser import ASTParser
from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.file_filter import FileFilter
from local_coding_assistant.repository.file_scope import FileScope
from local_coding_assistant.repository.metadata import MetadataExtractor
from local_coding_assistant.repository.models import (
    ASTNode,
    CallRelationship,
    FileMetadata,
)
from local_coding_assistant.repository.repo_map import RepoMapBuilder
from local_coding_assistant.repository.service import RepositoryContextService
from local_coding_assistant.repository.symbol_search import SymbolIndexer

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

    metadata_extractor = MetadataExtractor()
    file_filter = FileFilter(supported_extensions=SUPPORTED_EXTENSIONS)
    # Separate code files (for AST parsing) and config files (for metadata only)
    _, code_files = file_filter.filter_files(
        list(tracked_files), apply_test_file_filter=False
    )
    config_files = [
        f for f in tracked_files if f.name in metadata_extractor.CONFIG_FILE_PATTERNS
    ]

    print(f"\n{'=' * 80}")
    print(f"Found {len(tracked_files)} total files:")
    print(f"  - Code files (AST parsing): {len(code_files)}")
    print(f"  - Config files (metadata only): {len(config_files)}")
    print(f"{'=' * 80}\n")

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_repo.db"
        database = RepositoryDatabase(db_path)

        # Initialize components
        ast_parser = ASTParser()
        indexer = SymbolIndexer(database)

        repo_map_builder = RepoMapBuilder(database)

        # Process code files (AST parsing + indexing)
        max_cumulative_size_bytes = 10 * 1024 * 1024
        chunk_num = 0

        # Group files into chunks based on cumulative size
        current_chunk: list[Path] = []
        current_chunk_size = 0

        for file_path in code_files:
            file_size = file_path.stat().st_size if file_path.exists() else 0

            # If adding this file would exceed the limit, process current chunk first
            if current_chunk and (
                current_chunk_size + file_size > max_cumulative_size_bytes
            ):
                chunk_num += 1
                print(f"Processing chunk {chunk_num} with {len(current_chunk)} files")
                files_data = process_file_chunk(
                    ast_parser, current_chunk, args.nodes, args.calls
                )
                try:
                    # Index into database
                    print("\n[Database] Indexing...")
                    indexer.index_files(files_data)
                    print("  ✓ Indexed successfully")

                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    import traceback

                    traceback.print_exc()

                # Clear memory
                current_chunk = []
                current_chunk_size = 0

            current_chunk.append(file_path)
            current_chunk_size += file_size

        # Process remaining files in the last chunk
        if current_chunk:
            chunk_num += 1
            print(f"Processing chunk {chunk_num} with {len(current_chunk)} files")
            files_data = process_file_chunk(
                ast_parser, current_chunk, args.nodes, args.calls
            )
            try:
                # Index into database
                print("\n[Database] Indexing...")
                indexer.index_files(files_data)
                print("  ✓ Indexed successfully")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback

                traceback.print_exc()

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


def process_file_chunk(
    ast_parser: ASTParser,
    file_paths: list[Path],
    show_nodes: bool = False,
    show_calls: bool = False,
) -> list[tuple[list[ASTNode], FileMetadata, list[CallRelationship] | None]]:
    files_data = []

    for file_path in file_paths:
        print(f"\n{'─' * 80}")
        print(f"📄 Code File: {file_path.relative_to(MANUAL_TEST_DIR)}")
        print(f"{'─' * 80}")
        try:
            # Parse file
            print("\n[AST Parser] Parsing...")
            ast_nodes, file_metadata, call_relationships = ast_parser.parse(file_path)

            print(f"  ✓ Language: {file_metadata.language}")
            print(f"  ✓ Found {len(ast_nodes)} AST nodes")
            if ast_nodes:
                symbol_types = {
                    node.metadata.get("symbol_type", "unknown") for node in ast_nodes
                }
                print(f"  ✓ Symbol types: {symbol_types}")
                if show_nodes:
                    print("\nAST Nodes:")
                    for node in ast_nodes:
                        print(f"  {node}")

            print(f"\n  ✓ Found {len(call_relationships)} call relationships")
            if call_relationships:
                if show_calls:
                    print("\nCall Relationships:")
                    for call in call_relationships:
                        print(f"  {call}")

            # Update file metadata path to be relative
            relative_file_path = str(file_path.relative_to(MANUAL_TEST_DIR))
            file_metadata.path = relative_file_path
            for call in call_relationships:
                call.file_path = relative_file_path
            files_data.append((ast_nodes, file_metadata, call_relationships))
        except Exception as e:
            # Log error but continue with other files
            print(f"Error parsing file {file_path}: {e}")
            continue

    return files_data


if __name__ == "__main__":
    sys.exit(main())
