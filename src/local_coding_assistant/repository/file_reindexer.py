"""Shared file re-indexing utility for repository context service.

This module provides common re-indexing logic that can be used by:
- Startup re-indexing
- Agent edit re-indexing
- File monitoring (user edit re-indexing)
- Manual batch indexing

The logic is centralized to avoid duplication and ensure consistency
across different re-indexing strategies.
"""

import concurrent.futures
from pathlib import Path
from typing import Any

from local_coding_assistant.repository.ast_parser import ASTParser
from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.file_filter import FileFilter
from local_coding_assistant.repository.symbol_search import SymbolIndexer
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("repository.file_reindexer")


class FileReindexer:
    """Shared utility for parsing and re-indexing files.

    This class provides common re-indexing logic that can be used by
    different re-indexing strategies (startup, agent edit, file monitoring).
    """

    def __init__(
        self,
        database: RepositoryDatabase,
        file_filter: FileFilter | None = None,
        parallel_workers: int = 4,
    ) -> None:
        """Initialize the file re-indexer.

        Args:
            database: Repository database instance.
            file_filter: Optional file filter for extension and size filtering.
            parallel_workers: Number of parallel workers for batch re-indexing.
        """
        self.database = database
        self.file_filter = file_filter or FileFilter()
        self.parallel_workers = parallel_workers
        self.ast_parser = ASTParser()
        self.symbol_indexer = SymbolIndexer(database)

    def reindex_file(
        self,
        file_path: str | Path,
        apply_extension_filter: bool = True,
        apply_size_filter: bool = True,
        apply_test_file_filter: bool = True,
    ) -> bool:
        """Re-index a single file.

        Args:
            file_path: Path to the file to re-index.
            apply_extension_filter: Whether to apply extension filtering.
            apply_size_filter: Whether to apply size filtering.
            apply_test_file_filter: Whether to exclude test files.

        Returns:
            True if re-indexing succeeded, False otherwise.
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path

        if not path.exists():
            logger.warning(f"File does not exist: {path}")
            return False

        file_size = path.stat().st_size

        # Check if file should be indexed
        if not self.file_filter.should_index_file(
            path,
            file_size,
            apply_extension_filter=apply_extension_filter,
            apply_size_filter=apply_size_filter,
            apply_test_file_filter=apply_test_file_filter,
        ):
            return False

        try:
            # Parse file
            symbols, file_metadata, call_relationships = self.ast_parser.parse(path)

            # Index file
            self.symbol_indexer.index_file(symbols, file_metadata, call_relationships)

            return True
        except Exception as e:
            logger.warning(f"Error re-indexing file {path}: {e}")
            return False

    def parse_and_reindex_files_parallel(
        self,
        file_paths: list[Path],
        max_cumulative_size_kb: int = 10000,
    ) -> tuple[int, dict[str, str]]:
        """Parse files with AST parser and re-index multiple files in parallel.

        This method uses ThreadPoolExecutor to achieve real multi-core parallelism
        since tree-sitter releases the Python GIL during AST traversal in C.

        Args:
            file_paths: List of file paths to parse and re-index.
            max_cumulative_size_kb: Maximum cumulative size of files to index at a time (in KB).

        Returns:
            Tuple of (total number of successfully indexed files, dict of path -> content_hash).
        """
        if not file_paths:
            return 0, {}

        max_chunk_bytes = max_cumulative_size_kb * 1024
        files_data: list[tuple[Any, Any, Any]] = []
        file_hashes: dict[str, str] = {}
        current_chunk_bytes = 0
        total_indexed = 0

        # Tree-sitter releases the Python GIL during AST traversal in C,
        # making ThreadPoolExecutor achieve real multi-core parallelism.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.parallel_workers
        ) as executor:
            # Dispatch parsing jobs to worker threads
            future_to_path = {
                executor.submit(self.ast_parser.parse, path): path
                for path in file_paths
            }

            # Process results as worker threads finish them
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    symbols, file_metadata, call_relationships = future.result()
                    files_data.append((symbols, file_metadata, call_relationships))

                    # Store the hash from file_metadata
                    file_hashes[str(path)] = file_metadata.hash

                    # Accumulate actual file size
                    file_size = path.stat().st_size if path.exists() else 1024
                    current_chunk_bytes += file_size

                except Exception as e:
                    logger.warning(f"Error parsing {path.name}: {e}")
                    continue

                # Batch write to SQLite once cumulative size threshold is reached
                if current_chunk_bytes >= max_chunk_bytes:
                    self.symbol_indexer.index_files(files_data)
                    total_indexed += len(files_data)

                    # Reset buffer and counter
                    files_data.clear()
                    current_chunk_bytes = 0

        # Flush any remaining files in the final chunk
        if files_data:
            self.symbol_indexer.index_files(files_data)
            total_indexed += len(files_data)

        return total_indexed, file_hashes
