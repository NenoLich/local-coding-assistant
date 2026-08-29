"""File filtering for re-indexing operations.

This module handles extension and file size filtering for re-indexing operations.
Separate from file monitoring which tracks all git-tracked files.
"""

from pathlib import Path
from typing import ClassVar


class FileFilter:
    """Filters files based on extension and size for re-indexing."""

    DEFAULT_SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {
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

    DEFAULT_MAX_FILE_SIZE_KB = 500

    def __init__(
        self,
        supported_extensions: set[str] | None = None,
        max_file_size_kb: int | None = None,
    ) -> None:
        """Initialize the file filter.

        Args:
            supported_extensions: Set of file extensions to include (with leading dot).
                If None, uses DEFAULT_SUPPORTED_EXTENSIONS.
            max_file_size_kb: Maximum file size in kilobytes. If None, uses
                DEFAULT_MAX_FILE_SIZE_KB.
        """
        self.supported_extensions = (
            supported_extensions
            if supported_extensions is not None
            else self.DEFAULT_SUPPORTED_EXTENSIONS
        )
        self.max_file_size_bytes = (
            max_file_size_kb
            if max_file_size_kb is not None
            else self.DEFAULT_MAX_FILE_SIZE_KB
        ) * 1024

    def is_supported_extension(self, file_path: str | Path) -> bool:
        """Check if a file has a supported extension.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file extension is supported, False otherwise.
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        return path.suffix.lower() in self.supported_extensions

    def is_within_size_limit(self, file_size: int) -> bool:
        """Check if a file is within the size limit.

        Args:
            file_size: Size of the file in bytes.

        Returns:
            True if the file size is within the limit, False otherwise.
        """
        return file_size <= self.max_file_size_bytes

    def is_test_file(self, file_path: str | Path) -> bool:
        """Check if a file is a test file.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file is a test file, False otherwise.
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        file_name = path.name.lower()
        file_stem = path.stem.lower()

        # Check if file is inside a tests/ directory
        if "tests" in path.parts or file_name.startswith("test_"):
            return True

        # Python test files: test_*.py or *_test.py
        if file_stem.endswith("_test") and file_name.endswith(".py"):
            return True

        # TypeScript/JavaScript test files: *.test.ts, *.spec.ts, *.test.js, *.spec.js
        if file_stem.endswith(".test") and file_name.endswith(
            (".ts", ".tsx", ".js", ".jsx")
        ):
            return True
        if file_stem.endswith(".spec") and file_name.endswith(
            (".ts", ".tsx", ".js", ".jsx")
        ):
            return True

        # Go test files: *_test.go
        if file_stem.endswith("_test") and file_name.endswith(".go"):
            return True

        # Rust test files: *_test.rs
        if file_stem.endswith("_test") and file_name.endswith(".rs"):
            return True

        # C/C++ test files: *_test.c, *_test.cpp
        if file_stem.endswith("_test") and file_name.endswith((".c", ".cpp")):
            return True

        return False

    def should_index_file(
        self,
        file_path: str | Path,
        file_size: int,
        apply_extension_filter: bool = True,
        apply_size_filter: bool = True,
        apply_test_file_filter: bool = True,
    ) -> bool:
        """Check if a file should be indexed based on extension, size, and test file status.

        Args:
            file_path: Path to the file.
            file_size: Size of the file in bytes.
            apply_extension_filter: Whether to apply extension filtering.
            apply_size_filter: Whether to apply size filtering.
            apply_test_file_filter: Whether to exclude test files.

        Returns:
            True if the file should be indexed, False otherwise.
        """
        if apply_extension_filter and not self.is_supported_extension(file_path):
            return False
        if apply_size_filter and not self.is_within_size_limit(file_size):
            return False
        if apply_test_file_filter and self.is_test_file(file_path):
            return False
        return True

    def filter_files(
        self,
        file_paths: list[str | Path],
        apply_extension_filter: bool = True,
        apply_size_filter: bool = True,
        apply_test_file_filter: bool = True,
    ) -> tuple[int, list[Path]]:
        """Filter a list of files based on extension and size.

        Args:
            file_paths: List of file paths to filter.
            apply_extension_filter: Whether to apply extension filtering.
            apply_size_filter: Whether to apply size filtering.
            apply_test_file_filter: Whether to exclude test files.

        Returns:
            Tuple of cumulative size of filtered files in bytes and list of file paths that pass the filter.
        """
        cumulative_size = 0
        filtered_files = []
        for fp in file_paths:
            path = Path(fp) if isinstance(fp, str) else fp
            try:
                file_size = path.stat().st_size
                if self.should_index_file(
                    fp,
                    file_size,
                    apply_extension_filter,
                    apply_size_filter,
                    apply_test_file_filter,
                ):
                    filtered_files.append(path)
                    cumulative_size += file_size
            except (OSError, FileNotFoundError):
                continue
        return cumulative_size, filtered_files
