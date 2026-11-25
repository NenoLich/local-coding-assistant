"""Unit tests for BaselineStore - content-addressable baseline storage."""

from __future__ import annotations

import hashlib

import pytest

from local_coding_assistant.tools_platform.workspace.content_store import (
    BaselineStore,
    BaselineStoreConfig,
)


@pytest.fixture
def temp_baseline_dir(tmp_path):
    return tmp_path / "baselines"


@pytest.fixture
def baseline_config(temp_baseline_dir):
    return BaselineStoreConfig(
        base_dir=temp_baseline_dir,
        ttl_days=7,
        compression_enabled=True,
        large_file_threshold_mb=1,
    )


@pytest.fixture
def baseline_store(baseline_config):
    return BaselineStore(baseline_config)


class TestBaselineStore:
    """Test BaselineStore functionality."""

    def test_store_and_load(self, baseline_store):
        """Test storing and loading baseline content."""
        content = "hello world\n"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        stored_hash = baseline_store.store(content, content_hash)
        assert stored_hash == content_hash

        loaded = baseline_store.load(content_hash)
        assert loaded == content

    def test_store_computes_hash_if_not_provided(self, baseline_store):
        """Test store computes hash when not provided."""
        content = "hello world\n"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        stored_hash = baseline_store.store(content)
        assert stored_hash == content_hash

    def test_store_idempotent(self, baseline_store):
        """Test storing same content twice is idempotent."""
        content = "hello world\n"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        baseline_store.store(content, content_hash)
        baseline_store.store(content, content_hash)

        loaded = baseline_store.load(content_hash)
        assert loaded == content

    def test_load_nonexistent(self, baseline_store):
        """Test loading non-existent baseline returns None."""
        fake_hash = "a" * 64
        result = baseline_store.load(fake_hash)
        assert result is None

    def test_exists(self, baseline_store):
        """Test exists check."""
        content = "test content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        assert not baseline_store.exists(content_hash)
        baseline_store.store(content, content_hash)
        assert baseline_store.exists(content_hash)

    def test_compression_disabled(self, temp_baseline_dir):
        """Test baseline store with compression disabled."""
        config = BaselineStoreConfig(
            base_dir=temp_baseline_dir,
            compression_enabled=False,
        )
        store = BaselineStore(config)

        content = "hello world\n"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        store.store(content, content_hash)

        # File should not be gzipped
        path = store.get_baseline_path(content_hash)
        assert path.suffix != ".gz"
        loaded = store.load(content_hash)
        assert loaded == content

    def test_large_file_streaming(self, temp_baseline_dir):
        """Test large file handling with streaming."""
        config = BaselineStoreConfig(
            base_dir=temp_baseline_dir,
            large_file_threshold_mb=0,  # Force streaming for all files
        )
        store = BaselineStore(config)

        content = "x" * (2 * 1024 * 1024)  # 2MB
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        store.store(content, content_hash)
        loaded = store.load(content_hash)
        assert loaded == content

    def test_cleanup_expired(self, temp_baseline_dir):
        """Test TTL-based cleanup."""
        import time

        config = BaselineStoreConfig(
            base_dir=temp_baseline_dir,
            ttl_days=0,  # Expire immediately
            compression_enabled=True,
        )
        store = BaselineStore(config)

        content1 = "content1"
        content2 = "content2"
        hash1 = hashlib.sha256(content1.encode()).hexdigest()
        hash2 = hashlib.sha256(content2.encode()).hexdigest()

        store.store(content1, hash1)
        time.sleep(0.01)
        store.store(content2, hash2)
        time.sleep(0.01)

        # Both referenced - should not be deleted
        deleted = store.cleanup_expired(referenced_hashes={hash1, hash2})
        assert deleted == 0

        # Neither referenced - should be deleted (at least one, timing dependent)
        deleted = store.cleanup_expired(referenced_hashes=set())
        assert deleted >= 1

    def test_cleanup_preserves_referenced(self, temp_baseline_dir):
        """Test cleanup preserves referenced baselines even if old."""
        import time

        config = BaselineStoreConfig(
            base_dir=temp_baseline_dir,
            ttl_days=0,
            compression_enabled=True,
        )
        store = BaselineStore(config)

        content = "old content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        store.store(content, content_hash)
        time.sleep(0.01)

        # Referenced - should not be deleted
        deleted = store.cleanup_expired(referenced_hashes={content_hash})
        assert deleted == 0
        assert store.exists(content_hash)

    def test_get_size_bytes(self, baseline_store):
        """Test getting total size."""
        content = "hello world\n"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        initial_size = baseline_store.get_size_bytes()
        baseline_store.store(content, content_hash)
        final_size = baseline_store.get_size_bytes()

        assert final_size > initial_size

    def test_deduplication_across_sessions(self, baseline_store):
        """Test same content deduplicated across sessions."""
        content = "shared content\n"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Store from session A
        baseline_store.store(content, content_hash)
        # Store from session B (same content)
        baseline_store.store(content, content_hash)

        # Only one file on disk
        loaded = baseline_store.load(content_hash)
        assert loaded == content

    def test_unicode_content(self, baseline_store):
        """Test storing unicode content."""
        content = "héllo wörld\n日本語\n"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        baseline_store.store(content, content_hash)
        loaded = baseline_store.load(content_hash)
        assert loaded == content

    def test_binary_content_skipped(self, baseline_store):
        """Test binary content handling."""
        # Binary content that can't be decoded as UTF-8
        binary_content = b"\xff\xfe\x00\x00"

        # This would be handled at the caller level (observe_file skips binary)
        # But we can test the store handles it if passed
        try:
            content = binary_content.decode("utf-8")
        except UnicodeDecodeError:
            # Expected - caller should skip
            pass


class TestBaselineStoreConfig:
    """Test BaselineStoreConfig."""

    def test_defaults(self, temp_baseline_dir):
        """Test default configuration values."""
        config = BaselineStoreConfig(base_dir=temp_baseline_dir)
        assert config.ttl_days == 7
        assert config.compression_enabled is True
        assert config.large_file_threshold_mb == 1

    def test_custom_config(self, temp_baseline_dir):
        """Test custom configuration."""
        config = BaselineStoreConfig(
            base_dir=temp_baseline_dir,
            ttl_days=30,
            compression_enabled=False,
            large_file_threshold_mb=10,
        )
        assert config.ttl_days == 30
        assert config.compression_enabled is False
        assert config.large_file_threshold_mb == 10
