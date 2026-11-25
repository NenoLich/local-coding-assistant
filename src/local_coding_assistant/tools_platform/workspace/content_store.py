"""Content-addressable baseline storage and turn-scoped checkpoint storage.

This module provides:
- BaselineStore: Content-addressable storage for conflict detection baselines
  Keyed by content hash, shared across sessions/branches
- CheckpointStore: Turn-scoped storage for undo/restore operations
  Keyed by session_id + turn_id + file_hash
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from local_coding_assistant.utils.logging import get_logger

logger = get_logger("tools_platform.workspace.content_store")


@dataclass
class BaselineStoreConfig:
    """Configuration for baseline content storage."""

    base_dir: Path
    ttl_days: int = 7
    compression_enabled: bool = True
    large_file_threshold_mb: int = 1


@dataclass
class CheckpointStoreConfig:
    """Configuration for checkpoint storage."""

    base_dir: Path
    max_turns: int = 10
    ttl_days: int = 2
    compression_enabled: bool = True
    large_file_threshold_mb: int = 1


def _content_hash(content: bytes) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content).hexdigest()


def _file_hash_from_path(file_path: str) -> str:
    """Compute deterministic hash from file path for checkpoint naming."""
    return hashlib.sha256(file_path.encode()).hexdigest()


class BaselineStore:
    """Content-addressable baseline storage for conflict detection.

    Baselines are stored by their content hash, enabling deduplication
    across sessions and branches. No database index needed.
    """

    def __init__(self, config: BaselineStoreConfig) -> None:
        self.config = config
        self.config.base_dir.mkdir(parents=True, exist_ok=True)

    def get_baseline_path(self, content_hash: str) -> Path:
        """Get filesystem path for baseline content."""
        # Use first 16 chars of hash for directory sharding
        prefix = content_hash[:16]
        suffix = "content.gz" if self.config.compression_enabled else "content"
        return self.config.base_dir / prefix / suffix

    def delete(self, content_hash: str) -> None:
        """Delete baseline content by hash."""
        path = self.get_baseline_path(content_hash)
        if path.exists():
            try:
                path.unlink(missing_ok=True)
            except Exception:  # noqa: S110 - best-effort content deletion, failures are non-fatal
                pass

    def store(self, content: str, content_hash: str | None = None) -> str:
        """Store baseline content. Returns the content hash.

        If content_hash not provided, computes it. Idempotent - same content
        produces same hash and overwrites same file.
        """
        content_bytes = content.encode("utf-8")
        if content_hash is None:
            content_hash = _content_hash(content_bytes)

        path = self.get_baseline_path(content_hash)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.compression_enabled:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(content)
        else:
            path.write_text(content, encoding="utf-8")

        logger.debug("Stored baseline", hash=content_hash[:16], size=len(content_bytes))
        return content_hash

    def load(self, content_hash: str) -> str | None:
        """Load baseline content by hash. Returns None if not found."""
        path = self.get_baseline_path(content_hash)
        if not path.exists():
            return None

        try:
            if self.config.compression_enabled:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    return f.read()
            return path.read_text(encoding="utf-8")
        except (OSError, gzip.BadGzipFile) as e:
            logger.warning(
                "Failed to load baseline", hash=content_hash[:16], error=str(e)
            )
            return None

    def exists(self, content_hash: str) -> bool:
        """Check if baseline content exists."""
        return self.get_baseline_path(content_hash).exists()

    def cleanup_expired(self, referenced_hashes: set[str]) -> int:
        """Clean up baselines not in referenced_hashes and older than TTL.

        Note: Per design, we do NOT delete unreferenced baselines due to
        potential session_file_versions not being loaded for all sessions.
        Only TTL-based cleanup is performed.

        Returns number of deleted baselines.
        """
        deleted = 0
        cutoff = datetime.now() - timedelta(days=self.config.ttl_days)
        cutoff_ts = cutoff.timestamp()

        if not self.config.base_dir.exists():
            return 0

        for hash_dir in self.config.base_dir.iterdir():
            if not hash_dir.is_dir() or len(hash_dir.name) != 16:
                continue

            content_file = hash_dir / "content.gz"
            if not content_file.exists():
                continue

            try:
                mtime = content_file.stat().st_mtime
                if mtime < cutoff_ts:
                    # Check if still referenced - check if any referenced hash starts with this prefix
                    prefix = hash_dir.name
                    is_referenced = any(h.startswith(prefix) for h in referenced_hashes)
                    if not is_referenced:
                        shutil.rmtree(hash_dir)
                        deleted += 1
                        logger.debug("Deleted expired baseline", hash=prefix)
            except OSError:
                continue

        return deleted

    def get_size_bytes(self) -> int:
        """Get total size of baseline storage in bytes."""
        total = 0
        if self.config.base_dir.exists():
            for root, _dirs, files in os.walk(self.config.base_dir):
                for f in files:
                    total += (Path(root) / f).stat().st_size
        return total


class CheckpointStore:
    """Turn-scoped checkpoint storage for undo/restore operations.

    Checkpoints are stored per session, per turn, per file.
    Keyed by session_id + turn_id + file_hash.
    """

    def __init__(self, config: CheckpointStoreConfig) -> None:
        self.config = config
        self.config.base_dir.mkdir(parents=True, exist_ok=True)

    def _turn_dir(self, session_id: str, turn_id: str) -> Path:
        """Get directory for a turn's checkpoints."""
        return self.config.base_dir / session_id / f"turn_{turn_id}"

    def _checkpoint_path(self, session_id: str, turn_id: str, file_path: str) -> Path:
        """Get filesystem path for a checkpoint."""
        file_hash = _file_hash_from_path(file_path)
        turn_dir = self._turn_dir(session_id, turn_id)
        turn_dir.mkdir(parents=True, exist_ok=True)
        return turn_dir / f"{file_hash[:16]}.ckpt.gz"

    def store(
        self, session_id: str, turn_id: str, file_path: str, content: str | None
    ) -> str:
        """Store checkpoint content. Returns file hash.

        If content is None, stores a deletion marker (empty file).
        """
        file_hash = _file_hash_from_path(file_path)
        path = self._checkpoint_path(session_id, turn_id, file_path)
        content_bytes = None

        if content is None:
            # Deletion marker - empty file
            path.write_text("", encoding="utf-8")
        else:
            content_bytes = content.encode("utf-8")
            if self.config.compression_enabled:
                with gzip.open(path, "wt", encoding="utf-8") as f:
                    f.write(content)
            else:
                path.write_text(content, encoding="utf-8")

        logger.debug(
            "Stored checkpoint",
            session=session_id,
            turn=turn_id,
            file=file_path,
            hash=file_hash[:16],
            size=len(content_bytes) if content and content_bytes else 0,
        )
        return file_hash[:16]

    def load(self, session_id: str, turn_id: str, file_path: str) -> str | None:
        """Load checkpoint content.

        Returns:
            Content string if file existed
            Empty string if deletion marker (file was created)
            None if checkpoint not found
        """
        path = self._checkpoint_path(session_id, turn_id, file_path)
        if not path.exists():
            return None

        try:
            if self.config.compression_enabled:
                # Try to read as gzip first
                try:
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        return f.read()
                except gzip.BadGzipFile:
                    # Not compressed, fall through to text read
                    pass

            # Read as plain text (for non-compressed or fallback)
            content = path.read_text(encoding="utf-8")
            return content
        except OSError as e:
            logger.warning(
                "Failed to load checkpoint",
                session=session_id,
                turn=turn_id,
                file=file_path,
                error=str(e),
            )
            return None

    def exists(self, session_id: str, turn_id: str, file_path: str) -> bool:
        """Check if checkpoint exists."""
        return self._checkpoint_path(session_id, turn_id, file_path).exists()

    def list_turn_files(self, session_id: str, turn_id: str) -> list[str]:
        """List all file paths that have checkpoints in a turn."""
        turn_dir = self._turn_dir(session_id, turn_id)
        if not turn_dir.exists():
            return []

        files = []
        for ckpt_file in turn_dir.iterdir():
            if ckpt_file.suffix == ".ckpt.gz" or ckpt_file.suffix == ".ckpt":
                # We can't easily get original path from hash
                # This would need a manifest file - skip for now
                pass
        return files

    def load_turn_checkpoints(
        self, session_id: str, turn_id: str
    ) -> dict[str, str | None]:
        """Load all checkpoints for a turn.

        Returns dict of file_path -> content (None = deletion marker).
        Note: We can't recover original file paths from hash alone.
        This is a limitation - would need a manifest.
        """
        # For now, return empty - would need manifest to map hash -> path
        # This can be added later if needed for bulk undo
        return {}

    def cleanup_expired(self, active_sessions: dict[str, list[int]]) -> int:
        """Clean up checkpoints for expired turns.

        active_sessions: {session_id: [turn_ids_to_keep]}
        Also applies TTL.

        Returns number of deleted turn directories.
        """
        deleted = 0
        cutoff = datetime.now() - timedelta(days=self.config.ttl_days)
        cutoff_ts = cutoff.timestamp()

        if not self.config.base_dir.exists():
            return 0

        for session_dir in self.config.base_dir.iterdir():
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name
            keep_turns = set(active_sessions.get(session_id, []))

            for turn_dir in session_dir.iterdir():
                if not turn_dir.is_dir() or not turn_dir.name.startswith("turn_"):
                    continue

                try:
                    turn_id_str = turn_dir.name[5:]  # "turn_XXX"
                    turn_id = int(turn_id_str)

                    # Check TTL
                    mtime = turn_dir.stat().st_mtime
                    ttl_expired = mtime < cutoff_ts

                    # Check turn count limit
                    if keep_turns and turn_id not in keep_turns:
                        # Not in keep list
                        shutil.rmtree(turn_dir)
                        deleted += 1
                        logger.debug(
                            "Deleted expired checkpoint turn",
                            session=session_id,
                            turn=turn_id,
                        )
                    elif ttl_expired and (not keep_turns or turn_id not in keep_turns):
                        shutil.rmtree(turn_dir)
                        deleted += 1
                        logger.debug(
                            "Deleted TTL-expired checkpoint turn",
                            session=session_id,
                            turn=turn_id,
                        )
                except (ValueError, OSError):
                    continue

        return deleted

    def get_size_bytes(self) -> int:
        """Get total size of checkpoint storage in bytes."""
        total = 0
        if self.config.base_dir.exists():
            for root, _dirs, files in os.walk(self.config.base_dir):
                for f in files:
                    total += (Path(root) / f).stat().st_size
        return total
