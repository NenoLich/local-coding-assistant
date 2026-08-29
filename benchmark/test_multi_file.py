"""Benchmarks for multi-file operations."""

from pathlib import Path

from local_coding_assistant.repository import FileFilter, RepositoryDatabase
from local_coding_assistant.repository.ast_parser import ASTParser
from local_coding_assistant.repository.file_reindexer import FileReindexer


def test_parse_10_files_small(benchmark, benchmark_data_dir: Path):
    """Benchmark parsing 10 small files."""
    parser = ASTParser()
    files = list((benchmark_data_dir / "small").glob("test_small.*"))
    i = 10
    files_len = len(files)
    files = files * ((i // files_len) + 1)

    def parse_files():
        for file_path in files[:i]:
            parser.parse(file_path)

    benchmark(parse_files)


def test_parse_50_files_small(benchmark, benchmark_data_dir: Path):
    """Benchmark parsing 50 files (repeating small files)."""
    parser = ASTParser()
    files = list((benchmark_data_dir / "small").glob("test_small.*"))
    i = 50
    files_len = len(files)
    files = files * ((i // files_len) + 1)

    def parse_files():
        for file_path in files[:i]:
            parser.parse(file_path)

    benchmark(parse_files)


def test_index_10_files_small_batch(
    benchmark, benchmark_data_dir: Path, temp_db_path: Path
):
    """Benchmark indexing 10 small files end-to-end."""
    files = list((benchmark_data_dir / "small").glob("test_small.*"))
    i = 10
    files_len = len(files)
    files = files * ((i // files_len) + 1)

    def index_files():
        db = RepositoryDatabase(temp_db_path)
        file_filter = FileFilter()
        file_reindexer = FileReindexer(
            database=db,
            file_filter=file_filter,
            parallel_workers=4,
        )
        file_reindexer.parse_and_reindex_files_parallel(
            files[:i],
            max_cumulative_size_kb=10000,
        )

        return True

    benchmark(index_files)


def test_index_50_files_small_batch(
    benchmark, benchmark_data_dir: Path, temp_db_path: Path
):
    """Benchmark indexing 50 files (repeating small files)."""
    files = list((benchmark_data_dir / "small").glob("test_small.*"))
    i = 50
    files_len = len(files)
    files = files * ((i // files_len) + 1)

    def index_files():
        db = RepositoryDatabase(temp_db_path)
        file_filter = FileFilter()
        file_reindexer = FileReindexer(
            database=db,
            file_filter=file_filter,
            parallel_workers=4,
        )
        file_reindexer.parse_and_reindex_files_parallel(
            files[:i],
            max_cumulative_size_kb=10000,
        )

        return True

    benchmark(index_files)


def test_parse_10_files_medium(benchmark, benchmark_data_dir: Path):
    """Benchmark parsing 10 small files."""
    parser = ASTParser()
    files = list((benchmark_data_dir / "medium").glob("test_medium.*"))
    i = 10
    files_len = len(files)
    files = files * ((i // files_len) + 1)

    def parse_files():
        for file_path in files[:i]:
            parser.parse(file_path)

    benchmark(parse_files)


def test_parse_50_files_medium(benchmark, benchmark_data_dir: Path):
    """Benchmark parsing 50 files (repeating small files)."""
    parser = ASTParser()
    files = list((benchmark_data_dir / "medium").glob("test_medium.*"))
    i = 50
    files_len = len(files)
    files = files * ((i // files_len) + 1)

    def parse_files():
        for file_path in files[:i]:
            parser.parse(file_path)

    benchmark(parse_files)


def test_index_10_files_medium_batch(
    benchmark, benchmark_data_dir: Path, temp_db_path: Path
):
    """Benchmark indexing 10 small files end-to-end."""
    files = list((benchmark_data_dir / "medium").glob("test_medium.*"))
    i = 10
    files_len = len(files)
    files = files * ((i // files_len) + 1)

    def index_files():
        db = RepositoryDatabase(temp_db_path)
        file_filter = FileFilter()
        file_reindexer = FileReindexer(
            database=db,
            file_filter=file_filter,
            parallel_workers=4,
        )
        file_reindexer.parse_and_reindex_files_parallel(
            files[:i],
            max_cumulative_size_kb=10000,
        )

        return True

    benchmark(index_files)


def test_index_50_files_medium_batch(
    benchmark, benchmark_data_dir: Path, temp_db_path: Path
):
    """Benchmark indexing 50 files (repeating small files)."""
    files = list((benchmark_data_dir / "medium").glob("test_medium.*"))
    i = 50
    files_len = len(files)
    files = files * ((i // files_len) + 1)

    def index_files():
        db = RepositoryDatabase(temp_db_path)
        file_filter = FileFilter()
        file_reindexer = FileReindexer(
            database=db,
            file_filter=file_filter,
            parallel_workers=4,
        )
        file_reindexer.parse_and_reindex_files_parallel(
            files[:i],
            max_cumulative_size_kb=10000,
        )

        return True

    benchmark(index_files)


def test_index_100_files_medium_batch(
    benchmark, benchmark_data_dir: Path, temp_db_path: Path
):
    """Benchmark indexing 10 small files end-to-end."""
    files = list((benchmark_data_dir / "medium").glob("test_medium.*"))
    i = 100
    files_len = len(files)
    files = files * ((i // files_len) + 1)

    def index_files():
        db = RepositoryDatabase(temp_db_path)
        file_filter = FileFilter()
        file_reindexer = FileReindexer(
            database=db,
            file_filter=file_filter,
            parallel_workers=4,
        )
        file_reindexer.parse_and_reindex_files_parallel(
            files[:i],
            max_cumulative_size_kb=10000,
        )

        return True

    benchmark(index_files)
