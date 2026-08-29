"""Main service orchestrator for repository context."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_coding_assistant.repository.ast_parser import ASTParser
from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.file_filter import FileFilter
from local_coding_assistant.repository.file_monitoring import (
    FileChangeEvent,
    FileChangesChecker,
    FileChangeType,
    FileMonitoringService,
)
from local_coding_assistant.repository.file_reindexer import FileReindexer
from local_coding_assistant.repository.file_scope import FileScope
from local_coding_assistant.repository.metadata import MetadataExtractor
from local_coding_assistant.repository.models import ProjectInfo, SymbolResult
from local_coding_assistant.repository.reindexing import ReindexManager
from local_coding_assistant.repository.repo_map import (
    MapFormatter,
    RepoMapBuilder,
    RepoMapData,
)
from local_coding_assistant.repository.storage import StorageManager, StorageMode
from local_coding_assistant.repository.symbol_search import (
    SymbolIndexer,
    SymbolSearcher,
)
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.core.protocols import IConfigManager


logger = get_logger("repository.service")


class RepositoryContextService:
    """Main service for providing repository context to the coding agent."""

    def __init__(
        self,
        target_project_root: str | Path,
        config_manager: "IConfigManager",
    ) -> None:
        """Initialize the repository context service.

        Args:
            target_project_root: Path to the project root directory.
            config_manager: Configuration manager interface.
        """
        self.target_project_root = (
            Path(target_project_root)
            if isinstance(target_project_root, str)
            else target_project_root
        )
        self.config_manager = config_manager

        repo_config = self.config_manager.global_config.repository
        self.storage_mode = repo_config.storage.mode

        # Convert string mode to enum
        persistent_db_path = None
        mode_enum = StorageMode.TEMPORARY
        if self.storage_mode == "persistent":
            mode_enum = StorageMode.PERSISTENT
        elif self.storage_mode == "test":
            mode_enum = StorageMode.TEST

            # Initialize storage manager
        self.storage_manager = StorageManager(
            target_project_root=self.target_project_root,
            mode=mode_enum,
            persistent_db_path=persistent_db_path,
            temp_db_dir=repo_config.storage.temp_db_dir,
            path_manager=config_manager.path_manager,
            naming_strategy=repo_config.storage.temp_db_naming_strategy,
        )

        # Initialize AST parser and symbol indexer
        self.ast_parser = ASTParser(language_pack=repo_config.ast_parser.language_pack)

        # Initialize file filter with config parameters
        self.file_filter = FileFilter(
            supported_extensions=set(repo_config.file_filter.supported_extensions),
            max_file_size_kb=repo_config.file_filter.max_file_size_kb,
        )

        # Initialize metadata extractor
        self.metadata_extractor = MetadataExtractor(self.file_filter)

        # Initialize file scope
        tracking_strategy = (
            "git_with_walk_fallback"
            if repo_config.file_monitoring.use_git_tracking
            else "walk_only"
        )
        self.file_scope = FileScope(
            self.target_project_root, tracking_strategy=tracking_strategy
        )

        self.database: RepositoryDatabase | None = None
        self.repo_map_builder: RepoMapBuilder | None = None
        self.symbol_indexer: SymbolIndexer | None = None
        self.file_reindexer: FileReindexer | None = None
        self.reindex_manager: ReindexManager | None = None
        self.file_monitoring_service = None

        self.initialize_file_monitoring_service()

        # Check if database exists
        db_exists = self.storage_manager.db_exists()
        self.db_deps_initialized = False

        if db_exists:
            # Database exists - perform startup initialization
            logger.info("Database exists, performing startup initialization")

            self.initialize_db_dependencies()

            # Start file monitoring if enabled
            try:
                self.start_file_monitoring()
            except Exception as e:
                logger.warning("Failed to start file monitoring", error=str(e))
        else:
            # Database doesn't exist - service will be initialized on first use
            logger.info("No database found, service will initialize on first use")

        StorageManager.cleanup_old_temp_dbs(
            temp_db_dir=repo_config.storage.temp_db_dir,
            max_age_hours=24,
            path_manager=config_manager.path_manager,
        )

    def initialize_file_monitoring_service(self):
        """Initialize file monitoring service. Register re-index callback."""
        repo_config = self.config_manager.global_config.repository
        # Initialize file monitoring service
        if repo_config.file_monitoring.enabled:
            notification_types = []
            if "modified" in repo_config.file_monitoring.notification_types:
                notification_types.append(FileChangeType.MODIFIED)
            if "created" in repo_config.file_monitoring.notification_types:
                notification_types.append(FileChangeType.CREATED)
            if "deleted" in repo_config.file_monitoring.notification_types:
                notification_types.append(FileChangeType.DELETED)
            file_changes_checker = FileChangesChecker()
            self.file_monitoring_service = FileMonitoringService(
                target_project_root=self.target_project_root,
                debounce_window=repo_config.indexing.debounce_window,
                notification_types=notification_types,
                file_scope=self.file_scope,
                file_changes_checker=file_changes_checker,
            )

            # Register re-index callback for file monitoring
            def reindex_callback(event: FileChangeEvent) -> None:
                """Reindex a file changed by user editing."""
                if repo_config.indexing.user_edit_enabled and self.file_reindexer:
                    self.file_reindexer.reindex_file(
                        event.path,
                        apply_extension_filter=True,
                        apply_size_filter=True,
                        apply_test_file_filter=True,
                    )

            self.file_monitoring_service.register_notification_callback(
                reindex_callback
            )

    def should_create_database(self, code_files_to_index_size: int) -> bool:
        """Check if a new database should be created based on cumulative file size.

        Args:
            code_files_to_index_size: Cumulative size of code files to index.

        Returns:
            bool: True if a new database should be created, False otherwise.
        """
        return (
            self.storage_mode != StorageMode.TEMPORARY
            or self.storage_manager.should_create_temp_db(code_files_to_index_size)
        )

    def initialize_db_dependencies(self):
        """
        Initialize database dependencies. Create database if needed. Filter files to index.
        Initialize repo map builder and symbol indexer.
        Initialize file re-indexer and re-index manager.
        Perform initial indexing.
        """
        # Get all git-tracked files
        tracked_files = self.file_scope.get_tracked_files()

        # Filter by extension and size for re-indexing
        code_files_to_index_size, files_to_index = self.file_filter.filter_files(
            list(tracked_files),
            apply_extension_filter=True,
            apply_size_filter=True,
            apply_test_file_filter=True,
        )

        if not self.storage_manager.db_exists() and not self.should_create_database(
            code_files_to_index_size
        ):
            return

        # Initialize database
        db_path = self.storage_manager.get_db_path()
        db = RepositoryDatabase(db_path)
        self.database = db

        # Initialize repo map builder with config parameters
        repo_config = self.config_manager.global_config.repository
        self.repo_map_builder = RepoMapBuilder(
            db,
            max_symbols=repo_config.repo_map.max_symbols,
            include_docstrings=repo_config.repo_map.include_docstrings,
            include_imports=repo_config.repo_map.include_imports,
            included_symbol_types=set(repo_config.repo_map.included_symbol_types),
            relationship_weights=repo_config.call_graph.relationship_weights,
        )
        self.symbol_indexer = SymbolIndexer(db)

        # Initialize file re-indexer (shared utility)
        file_reindexer = FileReindexer(
            database=db,
            file_filter=self.file_filter,
            parallel_workers=repo_config.indexing.startup_parallel_workers,
        )
        self.file_reindexer = file_reindexer

        # Initialize re-index manager
        self.reindex_manager = ReindexManager(
            target_project_root=self.target_project_root,
            database=db,
            file_reindexer=file_reindexer,
            startup_enabled=repo_config.indexing.startup_enabled,
            agent_edit_enabled=repo_config.indexing.agent_edit_enabled,
        )

        # Perform startup re-indexing if enabled
        self.reindex_on_startup(files_to_index)

        self.db_deps_initialized = True
        self.config_manager.register_capability(["repo_map_database"])

    def get_repo_context(self) -> tuple[ProjectInfo, RepoMapData | None]:
        """Get repository context as a formatted string.

        Returns:
            Tuple of ProjectInfo and RepoMapData
        """
        # Build repo map (structured data)
        repo_map_data = None
        if self.repo_map_builder is None:
            logger.debug("Repo map builder not initialized")
        else:
            repo_map_data = self.repo_map_builder.build(
                target_project_root=self.target_project_root,
                convert_to_rel_path_if_possible=True,
            )

        # Extract project metadata
        tracked_files = self.file_scope.get_tracked_files()
        project_info = self.metadata_extractor.extract(
            tracked_files, self.target_project_root, force_refresh=False
        )

        # Use project name from metadata if repo map doesn't have one
        if not project_info.name and repo_map_data and repo_map_data.project_name:
            project_info.name = repo_map_data.project_name

        return project_info, repo_map_data

    @staticmethod
    def build_context_string(  # noqa: C901
        project_info: ProjectInfo,
        repo_map_data: RepoMapData,
        include_ranks: bool = False,
    ) -> str:
        formatter = MapFormatter()
        repo_map_str = formatter.format(repo_map_data, include_ranks)

        context_parts = []
        metadata_parts = []
        if project_info.name:
            metadata_parts.append(f"- Name: {project_info.name}")
        if project_info.description:
            metadata_parts.append(f"- Description: {project_info.description}")
        if repo_map_data.language_distribution:
            lang_strs = [
                f"{lang} ({pct:.1f}%)"
                for lang, pct in sorted(
                    repo_map_data.language_distribution.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
            metadata_parts.append(f"- Languages: {', '.join(lang_strs)}")
        if project_info.language_version:
            metadata_parts.append(
                f"- Language version: {project_info.language_version}"
            )
        if project_info.language_version_edition:
            metadata_parts.append(
                f"- Language version edition: {project_info.language_version_edition}"
            )
        if project_info.package_manager:
            metadata_parts.append(f"- Package Manager: {project_info.package_manager}")
        if project_info.core_frameworks:
            metadata_parts.append(
                f"- Core Frameworks: {', '.join(project_info.core_frameworks)}"
            )
        if project_info.build_backend:
            metadata_parts.append(f"- Build Backend: {project_info.build_backend}")
        if project_info.linter_formatter:
            metadata_parts.append(f"- Formatter: {project_info.linter_formatter}")
        if project_info.test_frameworks:
            metadata_parts.append(f"- Test Frameworks: {project_info.test_frameworks}")
        if project_info.license:
            metadata_parts.append(f"- License: {project_info.license}")
        if project_info.ts_config_target:
            metadata_parts.append(
                f"- TS config target: {project_info.ts_config_target}"
            )
        if project_info.ts_config_module:
            metadata_parts.append(
                f"- TS config module: {project_info.ts_config_module}"
            )
        if project_info.ts_config_module_resolution:
            metadata_parts.append(
                f"- TS config module resolution: {project_info.ts_config_module_resolution}"
            )
        if project_info.ts_config_jsx_mode:
            metadata_parts.append(
                f"- TS config jsx mode: {project_info.ts_config_jsx_mode}"
            )
        if project_info.ts_config_strict_mode:
            metadata_parts.append(
                f"- TS config strict mode: {project_info.ts_config_strict_mode}"
            )
        if project_info.ts_config_path_aliases:
            metadata_parts.append(
                f"- TS config path aliases: {project_info.ts_config_path_aliases}"
            )
        if project_info.is_workspace:
            metadata_parts.append(f"- Is rust workspace: {project_info.is_workspace}")
        if project_info.workspace_members:
            metadata_parts.append(
                f"- Workspace members: {project_info.workspace_members}"
            )
        if project_info.clean_dependencies:
            metadata_parts.append(f"- Dependencies: {project_info.clean_dependencies}")
        if project_info.manifest_files:
            metadata_parts.append(f"- Manifest files: {project_info.manifest_files}")
        if project_info.available_scripts:
            metadata_parts.append(
                f"- Available scripts: {project_info.available_scripts}"
            )

        if metadata_parts:
            context_parts.append("Project Metadata:")
            context_parts.extend(metadata_parts)

        # Add repo map (includes language distribution)
        if repo_map_str:
            context_parts.append(repo_map_str)

        return "\n".join(context_parts)

    def get_project_info(self, force_refresh: bool = False) -> ProjectInfo:
        """Get project metadata.

        Returns:
            ProjectInfo object with project metadata.
        """
        tracked_files = self.file_scope.get_tracked_files()
        project_info = self.metadata_extractor.extract(
            tracked_files, self.target_project_root, force_refresh=force_refresh
        )

        return project_info

    def get_repo_map_data(self) -> RepoMapData | None:
        """Get the repository map as structured data.

        Returns:
            RepoMapData with structured repository map information.
        """
        if self.repo_map_builder is None:
            logger.debug("Repo map builder not initialized")
            return None

        return self.repo_map_builder.build(
            target_project_root=self.target_project_root,
            convert_to_rel_path_if_possible=True,
        )

    def search_symbols(
        self,
        query: str,
        symbol_types: list[str] | None = None,
        file_path: str | None = None,
        limit: int = 20,
    ) -> list[SymbolResult]:
        """Search for symbols across the codebase.

        Args:
            query: Search query for symbol names.
            symbol_types: Optional filter by symbol types (function, class, method, etc.).
            file_path: Optional filter by specific file path.
            limit: Maximum number of results to return.

        Returns:
            List of matching symbols.
        """
        if self.database is None:
            logger.debug("Database not initialized")
            return []

        searcher = SymbolSearcher(self.database)
        return searcher.search_symbols(query, symbol_types, file_path, limit)

    def index_files_batch(
        self,
        file_paths: list[Path],
        max_cumulative_size_kb: int = 10000,
    ) -> int:
        """Index multiple files in batches with chunk-based processing.

        Args:
            file_paths: List of file paths to index.
            max_cumulative_size_kb: Maximum cumulative file size per batch in KB (default 10MB).

        Returns:
            Total number of files indexed.
        """
        if self.file_reindexer is None:
            logger.debug("File reindexer not initialized")
            return 0

        # Filter by extension and size for re-indexing
        _, files_to_index = self.file_filter.filter_files(
            list(file_paths),
            apply_extension_filter=True,
            apply_size_filter=True,
            apply_test_file_filter=True,
        )
        count, _ = self.file_reindexer.parse_and_reindex_files_parallel(
            files_to_index,
            max_cumulative_size_kb=max_cumulative_size_kb,
        )
        return count

    def reindex_on_startup(self, file_paths: list[Path]) -> dict[str, int]:
        """Check for modified files and re-index them on startup.

        Args:
            file_paths: List of file paths to re-index.

        Returns:
            Dictionary with re-indexing statistics (added, updated, deleted).
        """
        stats = {}
        try:
            if self.reindex_manager is None:
                logger.warning("Reindex manager not initialized")
                return stats
            stats, file_hashes = self.reindex_manager.reindex_on_startup(file_paths)
            if (
                stats.get("added", 0)
                + stats.get("updated", 0)
                + stats.get("deleted", 0)
                > 0
            ):
                logger.info(
                    "Startup re-index completed",
                    added=stats.get("added", 0),
                    updated=stats.get("updated", 0),
                    deleted=stats.get("deleted", 0),
                )
                # Update file hashes from reindexed files (hashes already computed during parsing)
                if self.file_monitoring_service and file_hashes:
                    self.file_monitoring_service.update_file_hashes(file_hashes)

        except Exception as e:
            logger.warning("Startup re-index failed", error=str(e))

        return stats

    def reindex_agent_edit(self, file_path: str | Path) -> bool:
        """Re-index a file modified by the agent.

        Args:
            file_path: Path to the file to re-index.

        Returns:
            True if re-indexing succeeded, False otherwise.
        """
        if self.reindex_manager is None:
            logger.warning("Reindex manager not initialized")
            return False

        return self.reindex_manager.reindex_agent_edit(file_path)

    def start_file_monitoring(self) -> None:
        """Start the file monitoring service for git-tracked files."""
        if self.file_monitoring_service:
            self.file_monitoring_service.start()
            self.config_manager.register_capability(["file_monitoring_service"])

    def stop_file_monitoring(self) -> None:
        """Stop the file monitoring service."""
        if self.file_monitoring_service:
            self.file_monitoring_service.stop()
            self.config_manager.unregister_capability(["file_monitoring_service"])

    def register_file_change_callback(self, callback: Any) -> None:
        """Register a callback for file change notifications.

        Args:
            callback: Function to call when a file change event occurs.
        """
        if self.file_monitoring_service:
            self.file_monitoring_service.register_notification_callback(callback)

    def close(self) -> None:
        """Close the service and clean up resources."""
        # Stop file monitoring if running
        if self.file_monitoring_service:
            self.file_monitoring_service.stop()

        if self.database:
            self.database.close()

        # Clean up temporary database if in temporary mode
        if self.storage_mode == "temporary":
            self.storage_manager.cleanup()

    def ensure_initialized(self) -> None:
        """Ensure the service is fully initialized (startup reindex and file monitoring).

        This method performs:
        1. Startup re-indexing if enabled and database creation if needed
        2. File monitoring startup if enabled

        This should be called when the service is first needed (e.g., on user prompt).
        """
        if not self.db_deps_initialized:
            self.initialize_db_dependencies()

        if self.file_monitoring_service and self.file_monitoring_service.is_running():
            return None

        # Start file monitoring if enabled and not already running
        try:
            self.start_file_monitoring()
        except Exception as e:
            logger.warning("Failed to start file monitoring", error=str(e))

        return None

    def __enter__(self) -> "RepositoryContextService":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
