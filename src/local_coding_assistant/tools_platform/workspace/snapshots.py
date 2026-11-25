"""File snapshot management for workspace transactions.

This module provides functionality for creating, storing, and restoring
file snapshots as part of workspace transaction management. Snapshots
are stored outside the target tree to avoid interference with workspace
operations.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from local_coding_assistant.tools_platform.contracts import (
    FileObservation,
    FileSnapshot,
)
from local_coding_assistant.tools_platform.workspace.models import WorkspaceSnapshot
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("tools_platform.workspace.snapshots")


@dataclass
class SnapshotStorageConfig:
    """Configuration for snapshot storage.

    Attributes:
        snapshots_dir: Directory where snapshots are stored
        max_snapshots: Maximum number of snapshots to keep per workspace
        max_snapshot_age_days: Maximum age of snapshots in days
        compression_enabled: Whether to compress snapshot data
    """

    snapshots_dir: Path
    max_snapshots: int = 10
    max_snapshot_age_days: int = 30
    compression_enabled: bool = False


class SnapshotManager:
    """Manages file snapshots for workspace transactions.

    This class handles:
    - Creating snapshots of individual files
    - Creating workspace-wide snapshots
    - Storing snapshots in a dedicated directory
    - Restoring files from snapshots
    - Cleaning up old snapshots
    """

    def __init__(self, config: SnapshotStorageConfig) -> None:
        """Initialize the snapshot manager.

        Args:
            config: Snapshot storage configuration
        """
        self.config = config
        self.config.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _compute_hash(self, content: bytes) -> str:
        """Compute SHA256 hash of content.

        Args:
            content: Content bytes to hash

        Returns:
            Hex digest of SHA256 hash
        """
        return hashlib.sha256(content).hexdigest()

    def _get_snapshot_path(self, snapshot_id: str) -> Path:
        """Get the filesystem path for a snapshot.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            Path to snapshot directory
        """
        return self.config.snapshots_dir / snapshot_id

    def _get_file_snapshot_path(self, snapshot_id: str, file_path: str) -> Path:
        """Get the filesystem path for a file snapshot.

        Args:
            snapshot_id: Snapshot identifier
            file_path: Relative file path

        Returns:
            Path to file snapshot
        """
        # Use a safe filename by replacing path separators
        safe_path = file_path.replace("/", "_").replace("\\", "_")
        return self._get_snapshot_path(snapshot_id) / f"{safe_path}.snapshot"

    def create_file_snapshot(
        self,
        workspace_root: Path,
        file_path: str,
        snapshot_id: str | None = None,
    ) -> FileSnapshot:
        """Create a snapshot of a single file.

        Args:
            workspace_root: Workspace root directory
            file_path: Relative path to file
            snapshot_id: Optional snapshot ID (generated if not provided)

        Returns:
            FileSnapshot with content and metadata

        Raises:
            FileNotFoundError: If file does not exist
            OSError: If file cannot be read
        """
        full_path = workspace_root / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        content_bytes = full_path.read_bytes()
        content = content_bytes.decode("utf-8", errors="replace")

        # Detect newline encoding using splitlines for reliability
        newline_type = None
        lines = content.splitlines(keepends=True)
        if lines:
            first_line_ending = (
                lines[0][-2:]
                if len(lines[0]) >= 2
                else lines[0][-1:]
                if lines[0]
                else ""
            )
            if first_line_ending == "\r\n":
                newline_type = "crlf"
            elif first_line_ending == "\r":
                newline_type = "cr"
            elif first_line_ending == "\n":
                newline_type = "lf"

        # Compute hash
        content_hash = self._compute_hash(content_bytes)

        # Get file metadata
        stat = full_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)

        # Get file permissions (mode) if available
        file_mode = stat.st_mode if hasattr(stat, "st_mode") else None

        # Create observation
        observation = FileObservation(
            path=file_path,
            hash=content_hash,
            size=stat.st_size,
            mtime=mtime,
            inode=str(stat.st_ino) if hasattr(stat, "st_ino") else None,
        )

        # Generate snapshot ID if not provided
        if snapshot_id is None:
            snapshot_id = str(uuid4())

        # Store snapshot on disk
        snapshot_path = self._get_file_snapshot_path(snapshot_id, file_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        # Store content and metadata (Pydantic handles datetime serialization)
        snapshot_data = {
            "content": content,
            "observation": observation.model_dump_json(),
            "file_mode": file_mode,
            "newline_type": newline_type,
        }

        if self.config.compression_enabled:
            # Write compressed data
            with gzip.open(snapshot_path, "wt", encoding="utf-8") as f:
                json.dump(snapshot_data, f)
        else:
            # Write uncompressed data
            snapshot_path.write_text(json.dumps(snapshot_data), encoding="utf-8")

        return FileSnapshot(
            path=file_path,
            content=content,
            observation=observation,
            snapshot_id=snapshot_id,
        )

    def create_deletion_marker(
        self,
        file_path: str,
        snapshot_id: str | None = None,
    ) -> FileSnapshot:
        """Create a deletion marker for a file that was deleted.

        Args:
            file_path: Relative path to file
            snapshot_id: Optional snapshot ID (generated if not provided)

        Returns:
            FileSnapshot with deletion marker
        """
        # Generate snapshot ID if not provided
        if snapshot_id is None:
            snapshot_id = str(uuid4())

        # Store deletion marker on disk
        snapshot_path = self._get_file_snapshot_path(snapshot_id, file_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        # Store deletion marker
        snapshot_data = {
            "deleted": True,
            "path": file_path,
        }

        if self.config.compression_enabled:
            with gzip.open(snapshot_path, "wt", encoding="utf-8") as f:
                json.dump(snapshot_data, f)
        else:
            snapshot_path.write_text(json.dumps(snapshot_data), encoding="utf-8")

        # Return a FileSnapshot with deletion marker
        return FileSnapshot(
            path=file_path,
            content="",
            observation=FileObservation(
                path=file_path,
                hash="",
                size=0,
                mtime=datetime.utcnow(),
                inode=None,
            ),
            snapshot_id=snapshot_id,
        )

    def restore_file_snapshot(
        self,
        workspace_root: Path,
        snapshot_id: str,
        file_path: str,
    ) -> None:
        """Restore a file from a snapshot.

        Args:
            workspace_root: Workspace root directory
            snapshot_id: Snapshot identifier
            file_path: Relative path to file

        Raises:
            FileNotFoundError: If snapshot does not exist
            OSError: If file cannot be written
        """
        snapshot_path = self._get_file_snapshot_path(snapshot_id, file_path)
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_id}/{file_path}")

        snapshot_data = self._load_snapshot_data(snapshot_path)

        # Check if this is a deletion marker
        if snapshot_data.get("deleted"):
            self._delete_marked_file(workspace_root, file_path)
            return

        content = self._restore_content(snapshot_data)

        # Write to workspace
        full_path = workspace_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with full_path.open(mode="w", encoding="utf-8", newline="") as f:
            f.write(content)

        self._restore_file_permissions(full_path, file_path, snapshot_data)

        logger.info(
            "Restored file from snapshot",
            file_path=file_path,
            snapshot_id=snapshot_id,
        )

    def _load_snapshot_data(self, snapshot_path: Path) -> dict[str, Any]:
        """Load snapshot data, handling both compressed and uncompressed formats."""
        try:
            with gzip.open(snapshot_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, gzip.BadGzipFile):
            return json.loads(snapshot_path.read_text(encoding="utf-8"))

    def _delete_marked_file(self, workspace_root: Path, file_path: str) -> None:
        """Delete a file marked as deleted in a snapshot."""
        full_path = workspace_root / file_path
        if full_path.exists():
            full_path.unlink()
            logger.info("Deleted file from deletion marker", file_path=file_path)

    def _restore_content(self, snapshot_data: dict[str, Any]) -> str:
        """Restore content with newline encoding handling."""
        content = snapshot_data["content"]
        newline_type = snapshot_data.get("newline_type")
        if newline_type:
            lines = content.splitlines()
            if newline_type == "crlf":
                content = "\r\n".join(lines)
            elif newline_type == "cr":
                content = "\r".join(lines)
        else:
            lines = content.splitlines(keepends=True)
            if lines and lines[-1].endswith(("\n", "\r")):
                if not content.endswith(("\n", "\r")):
                    content = content + "\n"
        return content

    def _restore_file_permissions(
        self, full_path: Path, file_path: str, snapshot_data: dict[str, Any]
    ) -> None:
        """Restore file permissions from snapshot data."""
        file_mode = snapshot_data.get("file_mode")
        if file_mode is not None:
            try:
                full_path.chmod(file_mode)
                logger.debug("Restored file permissions", file_path=file_path)
            except OSError as e:
                logger.warning(
                    "Failed to restore file permissions",
                    file_path=file_path,
                    error=str(e),
                )

    def create_workspace_snapshot(
        self,
        workspace_id: str,
        workspace_root: Path,
        file_paths: list[str],
        transaction_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot:
        """Create a snapshot of multiple files in the workspace.

        Args:
            workspace_id: Workspace identifier
            workspace_root: Workspace root directory
            file_paths: List of relative file paths to snapshot
            transaction_id: Optional transaction ID
            metadata: Optional metadata

        Returns:
            WorkspaceSnapshot with all file observations
        """
        snapshot_id = str(uuid4())
        file_snapshots: dict[str, FileObservation] = {}

        for file_path in file_paths:
            try:
                file_snapshot = self.create_file_snapshot(
                    workspace_root, file_path, snapshot_id
                )
                file_snapshots[file_path] = file_snapshot.observation
            except (FileNotFoundError, OSError) as e:
                logger.warning(
                    "Failed to snapshot file",
                    file_path=file_path,
                    error=str(e),
                )
                # Continue with other files

        workspace_snapshot = WorkspaceSnapshot(
            snapshot_id=snapshot_id,
            workspace_id=workspace_id,
            file_snapshots=file_snapshots,
            transaction_id=transaction_id,
            metadata=metadata or {},
        )

        # Write metadata.json for cleanup and retrieval
        self._write_snapshot_metadata(snapshot_id, workspace_snapshot)

        return workspace_snapshot

    def _write_snapshot_metadata(
        self, snapshot_id: str, snapshot: WorkspaceSnapshot
    ) -> None:
        """Write snapshot metadata to metadata.json.

        Args:
            snapshot_id: Snapshot identifier
            snapshot: WorkspaceSnapshot to write metadata for
        """
        snapshot_path = self._get_snapshot_path(snapshot_id)
        metadata_file = snapshot_path / "metadata.json"

        metadata = {
            "snapshot_id": snapshot.snapshot_id,
            "workspace_id": snapshot.workspace_id,
            "snapshot_time": snapshot.snapshot_time.isoformat(),
            "transaction_id": snapshot.transaction_id,
            "file_count": len(snapshot.file_snapshots),
            "metadata": snapshot.metadata,
        }

        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.debug("Wrote snapshot metadata", snapshot_id=snapshot_id)

    def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete a snapshot and all its files.

        Args:
            snapshot_id: Snapshot identifier
        """
        snapshot_path = self._get_snapshot_path(snapshot_id)
        if snapshot_path.exists():
            # Remove all files in snapshot directory
            for file_path in snapshot_path.iterdir():
                file_path.unlink()
            # Remove directory
            snapshot_path.rmdir()
            logger.info("Deleted snapshot", snapshot_id=snapshot_id)

    def cleanup_old_snapshots(self, workspace_id: str | None = None) -> int:
        """Clean up old snapshots based on retention policy.

        Args:
            workspace_id: Optional workspace ID to filter by

        Returns:
            Number of snapshots deleted
        """
        deleted_count = 0
        now = datetime.utcnow()

        if not self.config.snapshots_dir.exists():
            return deleted_count

        snapshots_info = self._collect_snapshots_info(now, workspace_id)

        # Sort by modification time (oldest first)
        snapshots_info.sort(key=lambda x: x["mtime"])

        # Delete snapshots exceeding max_snapshots limit
        if len(snapshots_info) > self.config.max_snapshots:
            to_delete = len(snapshots_info) - self.config.max_snapshots
            for snapshot_info in snapshots_info[:to_delete]:
                deleted_count += self._delete_snapshot_safely(
                    snapshot_info["snapshot_id"],
                    "Deleted snapshot exceeding max_snapshots limit",
                )

        # Delete snapshots exceeding max age
        for snapshot_info in snapshots_info:
            if snapshot_info["age_days"] > self.config.max_snapshot_age_days:
                deleted_count += self._delete_snapshot_safely(
                    snapshot_info["snapshot_id"],
                    None,
                )

        return deleted_count

    def _collect_snapshots_info(
        self, now: datetime, workspace_id: str | None
    ) -> list[dict[str, Any]]:
        """Collect snapshot directories with metadata and age."""
        snapshots_info: list[dict[str, Any]] = []
        for snapshot_dir in self.config.snapshots_dir.iterdir():
            if not snapshot_dir.is_dir():
                continue

            stat = snapshot_dir.stat()
            age_days = (now - datetime.fromtimestamp(stat.st_mtime)).days

            metadata_file = snapshot_dir / "metadata.json"
            snapshot_workspace_id = None
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                    snapshot_workspace_id = metadata.get("workspace_id")
                except (OSError, json.JSONDecodeError):
                    pass

            if workspace_id and snapshot_workspace_id != workspace_id:
                continue

            snapshots_info.append(
                {
                    "snapshot_id": snapshot_dir.name,
                    "age_days": age_days,
                    "workspace_id": snapshot_workspace_id,
                    "mtime": stat.st_mtime,
                }
            )
        return snapshots_info

    def _delete_snapshot_safely(
        self, snapshot_id: str, success_message: str | None
    ) -> int:
        """Delete a snapshot, returning 1 on success and 0 on failure."""
        try:
            self.delete_snapshot(snapshot_id)
            if success_message:
                logger.debug(success_message, snapshot_id=snapshot_id)
            return 1
        except OSError as e:
            logger.warning(
                "Failed to delete snapshot",
                snapshot_id=snapshot_id,
                error=str(e),
            )
            return 0

    def get_snapshot_metadata(self, snapshot_id: str) -> dict[str, Any] | None:
        """Get metadata for a snapshot.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            Snapshot metadata or None if not found
        """
        snapshot_path = self._get_snapshot_path(snapshot_id)
        metadata_file = snapshot_path / "metadata.json"

        if not metadata_file.exists():
            return None

        try:
            return json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
