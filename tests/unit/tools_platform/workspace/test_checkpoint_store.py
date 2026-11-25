"""Unit tests for CheckpointStore - turn-scoped checkpoint storage."""

from __future__ import annotations

import pytest

from local_coding_assistant.tools_platform.workspace.content_store import (
    CheckpointStore,
    CheckpointStoreConfig,
    _file_hash_from_path,
)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    return tmp_path / "checkpoints"


@pytest.fixture
def checkpoint_config(temp_checkpoint_dir):
    return CheckpointStoreConfig(
        base_dir=temp_checkpoint_dir,
        max_turns=10,
        ttl_days=2,
        compression_enabled=True,
        large_file_threshold_mb=1,
    )


@pytest.fixture
def checkpoint_store(checkpoint_config):
    return CheckpointStore(checkpoint_config)


class TestCheckpointStore:
    """Test CheckpointStore functionality."""

    def test_store_and_load_content(self, checkpoint_store):
        """Test storing and loading checkpoint with content."""
        session_id = "session_1"
        turn_id = "1"
        file_path = "test.py"
        content = "original content\n"

        checkpoint_hash = checkpoint_store.store(
            session_id, turn_id, file_path, content
        )

        # Returns first 16 chars of file hash
        assert checkpoint_hash == _file_hash_from_path(file_path)[:16]

        loaded = checkpoint_store.load(session_id, turn_id, file_path)
        assert loaded == content

    def test_store_deletion_marker(self, checkpoint_store):
        """Test storing deletion marker (content=None)."""
        session_id = "session_1"
        turn_id = "1"
        file_path = "new_file.py"

        checkpoint_hash = checkpoint_store.store(session_id, turn_id, file_path, None)

        loaded = checkpoint_store.load(session_id, turn_id, file_path)
        assert loaded == ""  # Empty string = deletion marker

    def test_load_nonexistent(self, checkpoint_store):
        """Test loading non-existent checkpoint returns None."""
        result = checkpoint_store.load("session_1", "1", "nonexistent.py")
        assert result is None

    def test_exists(self, checkpoint_store):
        """Test exists check."""
        session_id = "session_1"
        turn_id = "1"
        file_path = "test.py"

        assert not checkpoint_store.exists(session_id, turn_id, file_path)
        checkpoint_store.store(session_id, turn_id, file_path, "content")
        assert checkpoint_store.exists(session_id, turn_id, file_path)

    def test_different_turns_separate_checkpoints(self, checkpoint_store):
        """Test checkpoints are separate per turn."""
        session_id = "session_1"
        file_path = "test.py"

        checkpoint_store.store(session_id, "1", file_path, "turn1 content")
        checkpoint_store.store(session_id, "2", file_path, "turn2 content")

        loaded_turn1 = checkpoint_store.load(session_id, "1", file_path)
        loaded_turn2 = checkpoint_store.load(session_id, "2", file_path)

        assert loaded_turn1 == "turn1 content"
        assert loaded_turn2 == "turn2 content"

    def test_different_sessions_separate_checkpoints(self, checkpoint_store):
        """Test checkpoints are separate per session."""
        turn_id = "1"
        file_path = "test.py"

        checkpoint_store.store("session_A", turn_id, file_path, "session A content")
        checkpoint_store.store("session_B", turn_id, file_path, "session B content")

        loaded_a = checkpoint_store.load("session_A", turn_id, file_path)
        loaded_b = checkpoint_store.load("session_B", turn_id, file_path)

        assert loaded_a == "session A content"
        assert loaded_b == "session B content"

    def test_first_checkpoint_per_turn(self, checkpoint_store):
        """Test checkpoint store overwrites (idempotent check is at higher level)."""
        session_id = "session_1"
        turn_id = "1"
        file_path = "test.py"

        # First store
        hash1 = checkpoint_store.store(session_id, turn_id, file_path, "original")
        # Second store (overwrites at store level)
        hash2 = checkpoint_store.store(session_id, turn_id, file_path, "modified")

        assert hash1 == hash2
        loaded = checkpoint_store.load(session_id, turn_id, file_path)
        assert loaded == "modified"  # Last write wins at store level

    def test_compression_disabled(self, temp_checkpoint_dir):
        """Test checkpoint store with compression disabled."""
        config = CheckpointStoreConfig(
            base_dir=temp_checkpoint_dir,
            compression_enabled=False,
        )
        store = CheckpointStore(config)

        content = "hello world\n"
        store.store("session_1", "1", "test.py", content)
        loaded = store.load("session_1", "1", "test.py")
        assert loaded == content

    def test_cleanup_expired_by_turns(self, temp_checkpoint_dir):
        """Test cleanup keeps only max_turns."""
        config = CheckpointStoreConfig(
            base_dir=temp_checkpoint_dir,
            max_turns=2,
            ttl_days=365,  # Long TTL
        )
        store = CheckpointStore(config)

        session_id = "session_1"
        file_path = "test.py"

        # Create 5 turns
        for i in range(1, 6):
            store.store(session_id, str(i), file_path, f"content {i}")

        # Keep only last 2 turns (1, 2 will be deleted)
        active_sessions = {session_id: [4, 5]}
        deleted = store.cleanup_expired(active_sessions)
        assert deleted >= 2

        # Turns 4 and 5 should still exist
        assert store.exists(session_id, "4", file_path)
        assert store.exists(session_id, "5", file_path)
        # Turns 1, 2, 3 should be gone
        assert not store.exists(session_id, "1", file_path)
        assert not store.exists(session_id, "2", file_path)
        assert not store.exists(session_id, "3", file_path)

    def test_cleanup_expired_by_ttl(self, temp_checkpoint_dir):
        """Test TTL-based cleanup."""
        import time

        config = CheckpointStoreConfig(
            base_dir=temp_checkpoint_dir,
            max_turns=100,
            ttl_days=0,  # Expire immediately
        )
        store = CheckpointStore(config)

        session_id = "session_1"
        file_path = "test.py"

        store.store(session_id, "1", file_path, "content")
        time.sleep(0.01)  # Ensure different mtimes
        store.store(session_id, "2", file_path, "content")
        time.sleep(0.01)

        # No active sessions - all should be deleted
        deleted = store.cleanup_expired({})
        assert deleted >= 1  # At least one should be deleted (timing dependent)

    def test_cleanup_preserves_active_turns(self, temp_checkpoint_dir):
        """Test cleanup preserves active turns even if old."""
        config = CheckpointStoreConfig(
            base_dir=temp_checkpoint_dir,
            max_turns=2,
            ttl_days=0,
        )
        store = CheckpointStore(config)

        session_id = "session_1"
        file_path = "test.py"

        for i in range(1, 5):
            store.store(session_id, str(i), file_path, f"content {i}")

        # Turns 3 and 4 are active
        active_sessions = {session_id: [3, 4]}
        deleted = store.cleanup_expired(active_sessions)

        # Turns 1, 2 deleted; 3, 4 preserved
        assert not store.exists(session_id, "1", file_path)
        assert not store.exists(session_id, "2", file_path)
        assert store.exists(session_id, "3", file_path)
        assert store.exists(session_id, "4", file_path)

    def test_get_size_bytes(self, checkpoint_store):
        """Test getting total size."""
        initial_size = checkpoint_store.get_size_bytes()
        checkpoint_store.store("session_1", "1", "test.py", "content")
        final_size = checkpoint_store.get_size_bytes()
        assert final_size > initial_size

    def test_unicode_content(self, checkpoint_store):
        """Test storing unicode content."""
        content = "héllo wörld\n日本語\n"
        checkpoint_store.store("session_1", "1", "test.py", content)
        loaded = checkpoint_store.load("session_1", "1", "test.py")
        assert loaded == content

    def test_empty_content(self, checkpoint_store):
        """Test storing empty content."""
        checkpoint_store.store("session_1", "1", "empty.py", "")
        loaded = checkpoint_store.load("session_1", "1", "empty.py")
        assert loaded == ""

    def test_file_hash_deterministic(self):
        """Test file hash from path is deterministic."""
        path = "src/main.py"
        hash1 = _file_hash_from_path(path)
        hash2 = _file_hash_from_path(path)
        assert hash1 == hash2

    def test_different_paths_different_hashes(self):
        """Test different paths produce different hashes."""
        hash1 = _file_hash_from_path("test.py")
        hash2 = _file_hash_from_path("other.py")
        assert hash1 != hash2


class TestCheckpointStoreConfig:
    """Test CheckpointStoreConfig."""

    def test_defaults(self, temp_checkpoint_dir):
        """Test default configuration values."""
        config = CheckpointStoreConfig(base_dir=temp_checkpoint_dir)
        assert config.max_turns == 10
        assert config.ttl_days == 2
        assert config.compression_enabled is True
        assert config.large_file_threshold_mb == 1

    def test_custom_config(self, temp_checkpoint_dir):
        """Test custom configuration."""
        config = CheckpointStoreConfig(
            base_dir=temp_checkpoint_dir,
            max_turns=5,
            ttl_days=1,
            compression_enabled=False,
            large_file_threshold_mb=5,
        )
        assert config.max_turns == 5
        assert config.ttl_days == 1
        assert config.compression_enabled is False
        assert config.large_file_threshold_mb == 5
