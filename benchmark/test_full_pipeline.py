"""Benchmarks for full pipeline: parse + index + call graph."""

from pathlib import Path

from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.file_filter import FileFilter
from local_coding_assistant.repository.file_reindexer import FileReindexer
from local_coding_assistant.repository.repo_map import RepoMapBuilder


def test_full_pipeline_10_files_medium(
    benchmark, benchmark_data_dir: Path, temp_db_path: Path
):
    """Benchmark full pipeline on 10 small files: parse + index + call graph."""
    files = list((benchmark_data_dir / "medium").glob("test_medium.*"))
    i = 10
    files_len = len(files)
    files = files * ((i // files_len) + 1)

    def full_pipeline():
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

        repo_map = RepoMapBuilder(db).build()

        return repo_map

    benchmark(full_pipeline)


def test_full_pipeline_50_files_medium(
    benchmark, benchmark_data_dir: Path, temp_db_path: Path
):
    """Benchmark full pipeline on 50 files (repeating small files)."""
    files = list((benchmark_data_dir / "medium").glob("test_medium.*"))
    i = 50
    files_len = len(files)
    files = files * ((i // files_len) + 1)

    def full_pipeline():
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

        repo_map = RepoMapBuilder(db).build()

        return repo_map

    benchmark(full_pipeline)


def test_full_pipeline_100_files_medium(
    benchmark, benchmark_data_dir: Path, temp_db_path: Path
):
    """Benchmark full pipeline on 100 files (repeating small files)."""
    files = list((benchmark_data_dir / "medium").glob("test_medium.*"))
    i = 100
    files_len = len(files)
    files = files * ((i // files_len) + 1)

    def full_pipeline():
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

        repo_map = RepoMapBuilder(db).build()

        return repo_map

    benchmark(full_pipeline)
