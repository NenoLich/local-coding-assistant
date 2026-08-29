"""Re-indexing strategies for repository context service.

This module provides different re-indexing strategies:
- Startup re-indexing: Detects files modified while assistant was closed
- Agent edit re-indexing: Immediately re-indexes files modified by the AI assistant
- User edit re-indexing: Monitors git-tracked files for changes (handled separately in file_monitoring.py)
"""

import os
from pathlib import Path

from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.file_reindexer import FileReindexer
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("repository.reindexing")


class StartupReindexer:
    """Handles startup re-indexing with extension and size filtering.

    Detects and re-indexes files modified while the assistant was closed.
    """

    def __init__(
        self,
        database: RepositoryDatabase,
        file_reindexer: FileReindexer,
    ) -> None:
        """Initialize the startup re-indexer.

        Args:
            database: Repository database instance.
            file_reindexer: Shared file re-indexer instance.
        """
        self.database = database
        self.file_reindexer = file_reindexer

    def reindex_files(
        self, file_paths: list[Path]
    ) -> tuple[dict[str, int], dict[str, str]]:
        """Check for modified files and re-index them.

        Args:
            file_paths: List of file paths to re-index.

        Returns:
            Tuple of (statistics dictionary, dict of path -> content_hash).
        """
        stats = {"added": 0, "updated": 0, "deleted": 0}

        # Get all files currently in the database
        db_files = {f.path: f for f in self.database.get_all_files()}

        # Check for new and modified files
        files_to_reindex = []
        for file_path in file_paths:
            if not file_path.exists():
                # File was deleted
                if str(file_path) in db_files:
                    self.database.delete_file(str(file_path))
                    stats["deleted"] += 1
                    logger.debug(f"Deleted file from database: {file_path}")
                continue

            try:
                file_mtime = int(file_path.stat().st_mtime)
                db_file = db_files.get(str(file_path))
                # New file or modified file
                if db_file is None or file_mtime > db_file.last_indexed:
                    files_to_reindex.append(file_path)
                    if db_file is None:
                        stats["added"] += 1
                    else:
                        stats["updated"] += 1
            except OSError as e:
                logger.warning(f"Error checking file {file_path}: {e}")
                continue

        # Re-index files in parallel
        file_hashes: dict[str, str] = {}
        if files_to_reindex:
            logger.info(f"Re-indexing {len(files_to_reindex)} files on startup")
            _, file_hashes = self.file_reindexer.parse_and_reindex_files_parallel(
                files_to_reindex,
                max_cumulative_size_kb=10000,
            )
        else:
            logger.info("No files to re-index on startup")

        # Check for deleted files (in DB but not on disk)
        db_paths = set(db_files.keys())
        disk_paths = {str(fp) for fp in file_paths if fp.exists()}
        deleted_files = db_paths - disk_paths

        for file_path in deleted_files:
            self.database.delete_file(file_path)
            stats["deleted"] += 1

        return stats, file_hashes


class AgentEditReindexer:
    """Handles agent edit re-indexing with extension and size filtering.

    Immediately re-indexes files modified by the AI assistant.
    """

    def __init__(
        self,
        file_reindexer: FileReindexer,
    ) -> None:
        """Initialize the agent edit re-indexer.

        Args:
            file_reindexer: Shared file re-indexer instance.
        """
        self.file_reindexer = file_reindexer

    def reindex_file(self, file_path: str | Path) -> bool:
        """Re-index a single file modified by the agent.

        Args:
            file_path: Path to the file to re-index.

        Returns:
            True if re-indexing succeeded, False otherwise.
        """
        result = self.file_reindexer.reindex_file(
            file_path,
            apply_extension_filter=True,
            apply_size_filter=True,
            apply_test_file_filter=True,
        )
        if result:
            logger.info(f"Re-indexed agent-edited file: {file_path}")
        return result


class ReindexManager:
    """Coordinates all re-indexing strategies.

    This class provides a unified interface for different re-indexing strategies
    and manages their lifecycle.
    """

    def __init__(
        self,
        target_project_root: str | Path,
        database: RepositoryDatabase,
        file_reindexer: FileReindexer,
        startup_enabled: bool = True,
        agent_edit_enabled: bool = True,
    ) -> None:
        """Initialize the re-index manager.

        Args:
            target_project_root: Path to the project root directory.
            database: Repository database instance.
            file_reindexer: Shared file re-indexer instance.
            startup_enabled: Whether startup re-indexing is enabled.
            agent_edit_enabled: Whether agent edit re-indexing is enabled.
        """
        self.target_project_root = (
            Path(target_project_root)
            if isinstance(target_project_root, str)
            else target_project_root
        )
        self.database = database
        self.file_reindexer = file_reindexer
        self.startup_enabled = startup_enabled
        self.agent_edit_enabled = agent_edit_enabled

        # Initialize re-indexers
        self.startup_reindexer = None
        if startup_enabled:
            self.startup_reindexer = StartupReindexer(
                database=database,
                file_reindexer=file_reindexer,
            )

        self.agent_edit_reindexer = None
        if agent_edit_enabled:
            self.agent_edit_reindexer = AgentEditReindexer(
                file_reindexer=file_reindexer,
            )

    def reindex_on_startup(
        self, file_paths: list[Path]
    ) -> tuple[dict[str, int], dict[str, str]]:
        """Check for modified files and re-index them on startup.

        Args:
            file_paths: List of file paths to re-index.

        Returns:
            Tuple of (statistics dictionary, dict of path -> content_hash).
        """
        db_path = self.database.db_path
        if db_path.exists():
            os.utime(self.database.db_path, None)

        if not self.startup_enabled or self.startup_reindexer is None:
            logger.info("Startup re-indexing is disabled")
            return {"added": 0, "updated": 0, "deleted": 0}, {}

        return self.startup_reindexer.reindex_files(file_paths)

    def reindex_agent_edit(self, file_path: str | Path) -> bool:
        """Re-index a file modified by the agent.

        Args:
            file_path: Path to the file to re-index.

        Returns:
            True if re-indexing succeeded, False otherwise.
        """
        if not self.agent_edit_enabled or self.agent_edit_reindexer is None:
            logger.info("Agent edit re-indexing is disabled")
            return False

        return self.agent_edit_reindexer.reindex_file(file_path)
