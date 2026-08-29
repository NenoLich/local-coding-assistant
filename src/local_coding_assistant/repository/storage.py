"""Storage manager for database lifecycle (persistent/temp modes)."""

import hashlib
import uuid
from enum import Enum
from pathlib import Path

from local_coding_assistant.config.path_manager import PathManager
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("repository.storage")


class StorageMode(str, Enum):
    """Storage mode for the repository database."""

    PERSISTENT = "persistent"
    TEMPORARY = "temporary"
    TEST = "test"


class StorageManager:
    """Manages database file lifecycle for persistent and temporary modes."""

    DEFAULT_PERSISTENT_DB_PATH = ".locca/repo_context.db"
    DEFAULT_TEMP_DB_DIR = "@data/temp"

    def __init__(
        self,
        target_project_root: str | Path,
        mode: StorageMode = StorageMode.TEMPORARY,
        persistent_db_path: str | Path | None = None,
        temp_db_dir: str | None = None,
        test_db_path: str | Path | None = None,
        session_id: str | None = None,
        path_manager: PathManager | None = None,
        naming_strategy: str = "path_hash",
        max_cum_file_size: int = 1024 * 1024 * 1024 * 10,
    ) -> None:
        """Initialize the storage manager.

        Args:
            target_project_root: Project root path for generating temp db name.
            mode: Storage mode (persistent or temporary).
            persistent_db_path: Path for persistent database (relative to repo root).
                If None, uses DEFAULT_PERSISTENT_DB_PATH.
            temp_db_dir: Directory for temporary databases (supports @ aliases).
                If None, uses DEFAULT_TEMP_DB_DIR.
            session_id: Unique session ID for temporary databases.
                If None, generates based on naming_strategy.
            path_manager: Optional PathManager instance for path resolution.
            naming_strategy: Naming strategy for temp databases ('path_hash' or 'uuid').
            max_cum_file_size: Maximum cumulative file size for temporary database.
        """
        if not path_manager:
            from local_coding_assistant.config.env_manager import get_env_manager

            env_manager = get_env_manager()
            path_manager = getattr(env_manager, "path_manager", None)

        self.path_manager = path_manager or PathManager()
        self.mode = mode
        self.target_project_root = Path(target_project_root)
        self.naming_strategy = naming_strategy
        self.max_cum_file_size = max_cum_file_size

        if mode == StorageMode.PERSISTENT:
            db_path = Path(
                persistent_db_path
                if persistent_db_path
                else self.DEFAULT_PERSISTENT_DB_PATH
            )
            db_path_resolved = self.path_manager.resolve_path(
                db_path, base_dir=self.target_project_root
            )
            self.db_path = db_path_resolved
            if not db_path_resolved.exists():
                temp_db_path = self.get_temporary_db_path(
                    temp_db_dir=temp_db_dir, session_id=session_id
                )
                if temp_db_path.exists():
                    db_path_resolved.parent.mkdir(parents=True, exist_ok=True)
                    self.migrate_to_persistent(
                        persistent_db_path=db_path_resolved, temp_db_path=temp_db_path
                    )

        elif mode == StorageMode.TEMPORARY:
            self.db_path = self.get_temporary_db_path(
                temp_db_dir=temp_db_dir, session_id=session_id
            )

        self._is_initialized = False

        if mode == StorageMode.TEST:
            test_db_path_resolved = (
                test_db_path if test_db_path else self.DEFAULT_TEMP_DB_DIR
            )
            self.db_path = (
                Path(test_db_path_resolved)
                if isinstance(test_db_path_resolved, str)
                else test_db_path_resolved
            )
            self._is_initialized = True

    def get_db_path(self) -> Path:
        """Get the database file path.

        Returns:
            Path to the database file.
        """
        return self.db_path

    def initialize(self) -> None:
        """Initialize the storage (create directories if needed)."""
        if self._is_initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._is_initialized = True

    def should_create_temp_db(self, cum_files_size_to_index: int) -> bool:
        """Check if the cumulative file size exceeds the threshold for creating a new temporary database."""
        return cum_files_size_to_index < self.max_cum_file_size

    def cleanup(self) -> None:
        """Clean up temporary database files.

        For persistent mode, this is a no-op.
        For temporary mode, deletes the database file.
        """
        if self.mode == StorageMode.TEMPORARY and self.db_path.exists():
            self.db_path.unlink()
            self._is_initialized = False

    def get_temporary_db_path(
        self, temp_db_dir: str | None = None, session_id: str | None = None
    ) -> Path:
        """Get temporary database path using provided naming strategy

        Args:
            temp_db_dir: Directory for temporary databases (supports @ aliases).
                If None, uses DEFAULT_TEMP_DB_DIR.
            session_id: Second to target project root folder name part of the db filename if using path_hash naming strategy.

        Returns:
            Path to the temporary database file.
        """
        temp_dir = temp_db_dir if temp_db_dir else self.DEFAULT_TEMP_DB_DIR
        resolved_temp_dir = self.path_manager.resolve_path(temp_dir)
        resolved_temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate session ID based on naming strategy
        if session_id is None:
            if self.naming_strategy == "path_hash":
                session_id = self._generate_path_hash(self.target_project_root)
            else:
                session_id = str(uuid.uuid4())

        # Use folder_name_hash format for path_hash strategy
        if self.naming_strategy == "path_hash":
            folder_name = self.target_project_root.name
            db_filename = f"{folder_name}_{session_id}.db"
        else:
            db_filename = f"repo_{session_id}.db"
        return resolved_temp_dir / db_filename

    def db_exists(self) -> bool:
        """Check if the database file exists.

        Returns:
            True if the database file exists, False otherwise.
        """
        return self.db_path.exists()

    def get_db_size(self) -> int:
        """Get the database file size in bytes.

        Returns:
            File size in bytes, or 0 if file doesn't exist.
        """
        if self.db_path.exists():
            return self.db_path.stat().st_size
        return 0

    @staticmethod
    def _generate_path_hash(target_project_root: Path) -> str:
        """Generate a short hash from the project path.

        Args:
            target_project_root: Path to the project root.

        Returns:
            First 8 characters of SHA256 hash of the absolute path.
        """
        abs_path_str = str(target_project_root.absolute())
        return hashlib.sha256(abs_path_str.encode()).hexdigest()[:8]

    def migrate_to_persistent(
        self,
        persistent_db_path: str | Path | None = None,
        temp_db_path: str | Path | None = None,
    ) -> Path:
        """Migrate temporary database to persistent location.

        Args:
            persistent_db_path: Optional custom persistent db path.
                If None, uses DEFAULT_PERSISTENT_DB_PATH.
            temp_db_path: Optional custom temporary db path.
                If None, uses DEFAULT_TEMP_DB_DIR.

        Returns:
            Path to the new persistent database file.

        Raises:
            FileNotFoundError: If temporary database doesn't exist.
        """
        target_path = Path(
            persistent_db_path
            if persistent_db_path
            else self.DEFAULT_PERSISTENT_DB_PATH
        )
        target_path_resolved = self.path_manager.resolve_path(
            target_path, base_dir=self.target_project_root
        )
        target_path_resolved.parent.mkdir(parents=True, exist_ok=True)

        temp_db_path = (
            Path(temp_db_path) if temp_db_path else self.get_temporary_db_path()
        )

        if not temp_db_path.exists():
            msg = f"Temporary database not found: {temp_db_path}"
            raise FileNotFoundError(msg)

        # Copy the database file
        import shutil

        shutil.copy2(temp_db_path, target_path)
        logger.info(f"Migrated temporary database to {target_path}")

        # Update to persistent mode
        self.mode = StorageMode.PERSISTENT
        self.db_path = target_path

        return target_path

    @staticmethod
    def cleanup_old_temp_dbs(
        temp_db_dir: str | Path | None = None,
        max_age_hours: int = 24,
        path_manager: PathManager | None = None,
    ) -> int:
        """Clean up old temporary database files.

        This method cleans up temporary databases that haven't been accessed
        recently. Works with both path_hash naming ({folder_name}_{hash}.db)
        and UUID naming (repo_{uuid}.db).

        Args:
            temp_db_dir: Directory for temporary databases (supports @ aliases).
                If None, uses DEFAULT_TEMP_DB_DIR.
            max_age_hours: Maximum age in hours for temporary databases.
            path_manager: Optional PathManager instance for path resolution.

        Returns:
            Number of files cleaned up.
        """
        import time

        if path_manager:
            pm = path_manager
        else:
            from local_coding_assistant.config.env_manager import get_env_manager

            env_manager = get_env_manager()
            path_manager = getattr(env_manager, "path_manager", None)
            pm = path_manager or PathManager()

        temp_dir_str = (
            temp_db_dir if temp_db_dir else StorageManager.DEFAULT_TEMP_DB_DIR
        )
        resolved_temp_dir = pm.resolve_path(temp_dir_str)

        if not resolved_temp_dir.exists():
            return 0

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0

        # Match both naming schemes: {folder_name}_{hash}.db and repo_{uuid}.db
        for db_file in resolved_temp_dir.glob("*.db"):
            file_age = current_time - db_file.stat().st_mtime
            if file_age > max_age_seconds:
                db_file.unlink()
                cleaned_count += 1

        return cleaned_count

    @staticmethod
    def list_temp_dbs(
        temp_db_dir: str | Path | None = None,
        path_manager: PathManager | None = None,
    ) -> list[dict[str, str | int | float]]:
        """List all temporary databases with metadata.

        Args:
            temp_db_dir: Directory for temporary databases (supports @ aliases).
                If None, uses DEFAULT_TEMP_DB_DIR.
            path_manager: Optional PathManager instance for path resolution.

        Returns:
            List of dicts with 'path', 'size', and 'last_accessed' keys.
        """
        if path_manager:
            pm = path_manager
        else:
            from local_coding_assistant.config.env_manager import get_env_manager

            env_manager = get_env_manager()
            path_manager = getattr(env_manager, "path_manager", None)
            pm = path_manager or PathManager()

        temp_dir_str = (
            temp_db_dir if temp_db_dir else StorageManager.DEFAULT_TEMP_DB_DIR
        )
        resolved_temp_dir = pm.resolve_path(temp_dir_str)

        if not resolved_temp_dir.exists():
            return []

        db_info: list[dict[str, str | int | float]] = []
        for db_file in resolved_temp_dir.glob("*.db"):
            db_info.append(
                {
                    "path": str(db_file),
                    "size": db_file.stat().st_size,
                    "last_accessed": db_file.stat().st_mtime,
                }
            )

        return db_info
