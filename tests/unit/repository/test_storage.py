"""Unit tests for storage management."""

from pathlib import Path

import pytest

from local_coding_assistant.repository.storage import StorageManager, StorageMode


class TestStorageMode:
    """Test cases for StorageMode enum."""

    def test_storage_mode_values(self) -> None:
        """Test that storage mode enum has correct values."""
        assert StorageMode.PERSISTENT == "persistent"
        assert StorageMode.TEMPORARY == "temporary"


class TestStorageManager:
    """Test cases for StorageManager class."""

    def test_persistent_mode_db_path(self, tmp_path: Path) -> None:
        """Test persistent mode database path."""
        manager = StorageManager(
            target_project_root=tmp_path,
            mode=StorageMode.PERSISTENT,
            persistent_db_path=".locca/repo_context.db",
        )
        assert manager.db_path.relative_to(tmp_path) == Path(".locca/repo_context.db")

    def test_persistent_mode_migrates_from_custom_temp_db_dir(
        self, tmp_path: Path
    ) -> None:
        """Persistent mode should honor a custom temporary DB directory during migration."""
        temp_dir = tmp_path / "custom_temp"
        temp_dir.mkdir()
        temp_db = temp_dir / "repo_1234.db"
        temp_db.write_bytes(b"db-data")

        persistent_db = tmp_path / ".locca" / "repo_context.db"
        manager = StorageManager(
            target_project_root=tmp_path,
            mode=StorageMode.PERSISTENT,
            persistent_db_path=persistent_db,
            temp_db_dir=str(temp_dir),
            naming_strategy="uuid",
            session_id="1234",
        )

        assert manager.db_path == persistent_db
        assert persistent_db.exists()
        assert persistent_db.read_bytes() == b"db-data"
        assert manager.mode == StorageMode.PERSISTENT

    def test_temporary_mode_db_path(self, tmp_path: Path) -> None:
        """Test temporary mode database path with path_hash naming."""
        manager = StorageManager(
            target_project_root=tmp_path,
            mode=StorageMode.TEMPORARY,
            temp_db_dir=str(tmp_path),
            naming_strategy="path_hash",
        )
        # Should be {folder_name}_{hash}.db
        assert manager.db_path.name.startswith(tmp_path.name + "_")
        assert manager.db_path.name.endswith(".db")

    def test_temporary_mode_uuid_naming(self, tmp_path: Path) -> None:
        """Test temporary mode database path with UUID naming."""
        manager = StorageManager(
            target_project_root=tmp_path,
            mode=StorageMode.TEMPORARY,
            temp_db_dir=str(tmp_path),
            naming_strategy="uuid",
        )
        # Should be repo_{uuid}.db
        assert manager.db_path.name.startswith("repo_")
        assert manager.db_path.name.endswith(".db")

    def test_initialize_creates_directory(self, tmp_path: Path) -> None:
        """Test that initialize creates parent directories."""
        db_path = tmp_path / "subdir" / "repo.db"
        manager = StorageManager(
            target_project_root=tmp_path,
            mode=StorageMode.PERSISTENT,
            persistent_db_path=str(db_path),
        )
        manager.initialize()
        assert db_path.parent.exists()

    def test_cleanup_temporary_mode(self, tmp_path: Path) -> None:
        """Test that cleanup removes database in temporary mode."""
        manager = StorageManager(
            target_project_root=tmp_path,
            mode=StorageMode.TEMPORARY,
            temp_db_dir=str(tmp_path),
            naming_strategy="path_hash",
        )
        manager.initialize()
        manager.db_path.touch()
        assert manager.db_path.exists()

        manager.cleanup()
        assert not manager.db_path.exists()

    def test_cleanup_persistent_mode(self, tmp_path: Path) -> None:
        """Test that cleanup is no-op in persistent mode."""
        db_path = tmp_path / "repo.db"
        manager = StorageManager(
            target_project_root=tmp_path,
            mode=StorageMode.PERSISTENT,
            persistent_db_path=str(db_path),
        )
        manager.initialize()
        db_path.touch()
        assert db_path.exists()

        manager.cleanup()
        assert db_path.exists()  # Should still exist

    def test_db_exists(self, tmp_path: Path) -> None:
        """Test db_exists method."""
        manager = StorageManager(
            target_project_root=tmp_path,
            mode=StorageMode.TEMPORARY,
            temp_db_dir=str(tmp_path),
            naming_strategy="path_hash",
        )
        assert not manager.db_exists()
        manager.db_path.touch()
        assert manager.db_exists()

    def test_get_db_size(self, tmp_path: Path) -> None:
        """Test get_db_size method."""
        manager = StorageManager(
            target_project_root=tmp_path,
            mode=StorageMode.TEMPORARY,
            temp_db_dir=str(tmp_path),
            naming_strategy="path_hash",
        )
        assert manager.get_db_size() == 0
        manager.db_path.write_text("test content")
        assert manager.get_db_size() > 0

    def test_cleanup_old_temp_dbs(self, tmp_path: Path) -> None:
        """Test cleaning up old temporary databases."""
        import os
        import time

        # Create an old database
        old_db = tmp_path / "myproject_oldhash.db"
        old_db.touch()
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(old_db, (old_time, old_time))

        # Create a recent database
        recent_db = tmp_path / "myproject_newhash.db"
        recent_db.touch()

        # Create a UUID-style database
        uuid_db = tmp_path / "repo_12345678-1234-1234-1234-123456789abc.db"
        uuid_db.touch()

        cleaned = StorageManager.cleanup_old_temp_dbs(
            temp_db_dir=str(tmp_path),
            max_age_hours=24,
        )
        assert cleaned == 1
        assert not old_db.exists()
        assert recent_db.exists()
        assert uuid_db.exists()

    def test_list_temp_dbs(self, tmp_path: Path) -> None:
        """Test listing temporary databases."""
        # Create some test databases
        db1 = tmp_path / "project1_hash1.db"
        db1.touch()
        db1.write_text("content1")

        db2 = tmp_path / "project2_hash2.db"
        db2.touch()
        db2.write_text("content2")

        db_list = StorageManager.list_temp_dbs(temp_db_dir=str(tmp_path))
        assert len(db_list) == 2
        assert any(db["path"].endswith("project1_hash1.db") for db in db_list)
        assert any(db["path"].endswith("project2_hash2.db") for db in db_list)
        assert all(db["size"] > 0 for db in db_list)
        assert all("last_accessed" in db for db in db_list)
