"""Background file monitoring service for repository context service.

This module provides file monitoring for tracked files to detect user edits
and emit system notifications during active agent sessions.

Note: This is separate from re-indexing. File monitoring tracks all tracked
files for notifications, while re-indexing filters by extension and size.
"""

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from local_coding_assistant.repository.file_scope import FileScope
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("repository.file_monitoring")


class FileChangeType(str, Enum):
    """Types of file change events."""

    MODIFIED = "modified"
    CREATED = "created"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileChangeEvent:
    """Event representing a file modification."""

    path: str
    change_type: FileChangeType
    timestamp: float
    is_tracked: bool


class FileChangesChecker:
    """In-memory cache of file content hashes for change detection.

    This class maintains a dictionary mapping file paths to their content hashes.
    It's used as a fast pre-filter to avoid emitting events for files whose
    content hasn't actually changed.
    """

    def __init__(self) -> None:
        """Initialize the file changes checker."""
        self._file_hashes: dict[str, str] = {}

    def _compute_hash(self, file_path: Path) -> str | None:
        """Compute MD5 hash of file content.

        Args:
            file_path: Path to the file.

        Returns:
            Hex digest of MD5 hash, or None if file cannot be read.
        """
        try:
            content = file_path.read_bytes()
            return hashlib.md5(content).hexdigest()  # noqa: S324
        except OSError as e:
            logger.debug(f"Failed to read file for hash computation: {file_path}: {e}")
            return None

    def has_content_changed(self, file_path: Path, change_type: FileChangeType) -> bool:
        """Check if file content has actually changed.

        Args:
            file_path: Path to the file.
            change_type: Type of file change event.

        Returns:
            True if content has changed or cannot be determined, False otherwise.
        """
        # Only check content for MODIFIED and CREATED events
        if change_type not in (FileChangeType.MODIFIED, FileChangeType.CREATED):
            return True

        path_str = str(file_path)

        # If file doesn't exist, consider it changed
        if not file_path.exists():
            return True

        # Compute current hash
        current_hash = self._compute_hash(file_path)
        if current_hash is None:
            # If we can't read the file, assume it changed
            return True

        # Check if hash is in cache
        if path_str not in self._file_hashes:
            # New file to our cache, consider it changed
            self._file_hashes[path_str] = current_hash
            return True

        # Compare with cached hash
        if self._file_hashes[path_str] == current_hash:
            # Content hasn't changed
            return False

        # Content has changed, update cache
        self._file_hashes[path_str] = current_hash
        return True

    def update_hash(self, file_path: str, content_hash: str) -> None:
        """Update the hash for a file in the cache.

        Args:
            file_path: Path to the file.
            content_hash: Content hash to update.
        """
        if not content_hash:
            return

        self._file_hashes[file_path] = content_hash

    def update_file_hashes(self, file_hashes: dict[str, str]) -> None:
        """Update hashes for multiple files.

        This is typically called during startup to populate the cache
        with recently changed files.

        Args:
            file_hashes: File hashes dictionary with file path as a key and content hash as a value.
        """
        for file_path in file_hashes:
            self.update_hash(file_path, file_hashes[file_path])

    def remove_hash(self, file_path: Path) -> None:
        """Remove a file from the hash cache.

        Args:
            file_path: Path to the file.
        """
        self._file_hashes.pop(str(file_path), None)

    def clear(self) -> None:
        """Clear all hashes from the cache."""
        self._file_hashes.clear()


class NotificationEmitter:
    """Emits system notifications for file changes.

    This class provides a callback mechanism for emitting notifications
    when file changes are detected.
    """

    def __init__(self, file_changes_checker: FileChangesChecker | None = None) -> None:
        """Initialize the notification emitter.

        Args:
            file_changes_checker: Optional file changes checker for content filtering.
        """
        self._callbacks: list[Callable[[FileChangeEvent], None]] = []
        self._file_changes_checker = file_changes_checker

    def register_callback(self, callback: Callable[[FileChangeEvent], None]) -> None:
        """Register a callback for file change notifications.

        Args:
            callback: Function to call when a file change event occurs.
        """
        self._callbacks.append(callback)

    def emit(self, event: FileChangeEvent) -> None:
        """Emit a file change notification.

        Args:
            event: The file change event to emit.
        """
        # Check if content actually changed if checker is available
        if self._file_changes_checker is not None:
            file_path = Path(event.path)
            if not self._file_changes_checker.has_content_changed(
                file_path, event.change_type
            ):
                # Content hasn't changed, skip emission
                return

        logger.debug(
            f"File change notification emitted: {event.path} ({event.change_type.value})"
        )

        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Error in notification callback: {e}")


class Debouncer:
    """Debounces file change events to avoid rapid-fire notifications.

    Multiple changes to the same file within the debounce window are
    coalesced into a single event.
    """

    def __init__(self, delay: float = 2.0) -> None:
        """Initialize the debouncer.

        Args:
            delay: Delay in seconds before emitting a debounced event.
        """
        self.delay = delay
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def debounce(self, file_path: str, callback: Callable[[str], None]) -> None:
        """Debounce a file change event.

        Args:
            file_path: Path to the file that changed.
            callback: Function to call after debounce delay.
        """
        with self._lock:
            # Cancel the old timer if it's still waiting
            if file_path in self._timers:
                self._timers[file_path].cancel()

            # Schedule the new timer
            timer = threading.Timer(
                self.delay, self._execute, args=(file_path, callback)
            )
            self._timers[file_path] = timer
            timer.start()

    def _execute(self, file_path: str, callback: Callable[[str], None]) -> None:
        with self._lock:
            if file_path in self._timers:
                del self._timers[file_path]
        # Call the callback outside the lock to prevent potential deadlocks
        callback(file_path)


class TrackedFileHandler(FileSystemEventHandler):
    """File system event handler that filters to tracked files only."""

    def __init__(
        self,
        target_project_root: Path,
        file_scope: FileScope,
        debouncer: Debouncer,
        notification_emitter: NotificationEmitter,
        notification_types: list[FileChangeType] | None = None,
    ) -> None:
        """Initialize the tracked file handler.

        Args:
            target_project_root: Path to the project root directory.
            file_scope: File scope utility for tracking files.
            debouncer: Debouncer for coalescing rapid changes.
            notification_emitter: Emitter for file change notifications.
            notification_types: List of change types to notify on.
        """
        self.target_project_root = target_project_root
        self.file_scope = file_scope
        self.debouncer = debouncer
        self.notification_emitter = notification_emitter
        self.notification_types = notification_types or [
            FileChangeType.MODIFIED,
            FileChangeType.CREATED,
            FileChangeType.DELETED,
        ]
        self._tracked_files: set[Path] | None = None

    def _get_tracked_files(self) -> set[Path]:
        """Get the set of tracked files.

        Returns:
            Set of tracked file paths (absolute).
        """
        return self.file_scope.get_tracked_files()

    def _refresh_tracked_files(self) -> None:
        """Refresh the cached set of tracked files."""
        self._tracked_files = self._get_tracked_files()

    def _is_tracked(self, file_path: str) -> bool:
        """Check if a file is tracked.

        Args:
            file_path: Path to the file (can be absolute or relative).

        Returns:
            True if the file is tracked, False otherwise.
        """
        # Refresh cache if needed
        if self._tracked_files is None:
            self._refresh_tracked_files()

        # Convert to absolute path
        path = Path(file_path)
        if not path.is_absolute():
            path = self.target_project_root / path

        return self._tracked_files is not None and path in self._tracked_files

    def _handle_file_change(
        self, event: FileSystemEvent, change_type: FileChangeType
    ) -> None:
        """Handle a file change event.

        Args:
            event: The file system event.
            change_type: The type of change.
        """
        # Ensure file_path is a string
        file_path = (
            event.src_path
            if isinstance(event.src_path, str)
            else event.src_path.decode("utf-8")
        )

        # Check if this change type should be notified
        if change_type not in self.notification_types:
            return

        # Check if file is tracked
        is_tracked = self._is_tracked(file_path)
        if not is_tracked:
            return

        # Debounce the event
        def emit_event(path: str) -> None:
            change_event = FileChangeEvent(
                path=path,
                change_type=change_type,
                timestamp=time.time(),
                is_tracked=True,
            )
            self.notification_emitter.emit(change_event)

        self.debouncer.debounce(file_path, emit_event)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modified event.

        Args:
            event: The file system event.
        """
        if not event.is_directory:
            self._handle_file_change(event, FileChangeType.MODIFIED)

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file created event.

        Args:
            event: The file system event.
        """
        if not event.is_directory:
            # Refresh tracked files cache when a new file is created
            self._refresh_tracked_files()
            self._handle_file_change(event, FileChangeType.CREATED)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deleted event.

        Args:
            event: The file system event.
        """
        if not event.is_directory:
            self._handle_file_change(event, FileChangeType.DELETED)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file moved event.

        Args:
            event: The file system event.
        """
        if not event.is_directory:
            self._handle_file_change(event, FileChangeType.MOVED)


class FileMonitoringService:
    """Background file monitoring service for tracked files.

    This service monitors tracked files for changes and emits
    notifications during active agent sessions.

    Note: This is separate from re-indexing. File monitoring tracks
    all tracked files for notifications, while re-indexing filters
    by extension and size.
    """

    def __init__(
        self,
        target_project_root: str | Path,
        debounce_window: float = 2.0,
        notification_types: list[FileChangeType] | None = None,
        file_scope: FileScope | None = None,
        file_changes_checker: FileChangesChecker | None = None,
    ) -> None:
        """Initialize the file monitoring service.

        Args:
            target_project_root: Path to the project root directory.
            debounce_window: Delay in seconds for debouncing file changes.
            notification_types: List of change types to notify on.
            file_scope: Optional file scope to filter tracked files.
            file_changes_checker: Optional file changes checker for content filtering.
        """
        self.target_project_root = (
            Path(target_project_root)
            if isinstance(target_project_root, str)
            else target_project_root
        )
        self.debounce_window = debounce_window
        self.notification_types = notification_types or [
            FileChangeType.MODIFIED,
            FileChangeType.CREATED,
            FileChangeType.DELETED,
        ]

        self.debouncer = Debouncer(delay=debounce_window)
        if file_scope is None:
            file_scope = FileScope(self.target_project_root)
        self.file_scope = file_scope
        self.file_changes_checker = file_changes_checker or FileChangesChecker()
        self.notification_emitter = NotificationEmitter(
            file_changes_checker=self.file_changes_checker
        )
        self.observer: Any = None
        self._is_running = False

    def register_notification_callback(
        self, callback: Callable[[FileChangeEvent], None]
    ) -> None:
        """Register a callback for file change notifications.

        Args:
            callback: Function to call when a file change event occurs.
        """
        self.notification_emitter.register_callback(callback)

    def update_file_hashes(self, file_hashes: dict[str, str]) -> None:
        """Update hashes for multiple files.

        This is typically called during startup to populate the cache
        with recently changed files.

        Args:
            file_hashes: File hashes dictionary with file path as a key and content hash as a value.
        """

        self.file_changes_checker.update_file_hashes(file_hashes)

    def start(self) -> None:
        """Start the file monitoring service.

        This starts a background thread that monitors the file system.
        """
        if self._is_running:
            logger.warning("File monitoring service is already running")
            return

        logger.info(f"Starting file monitoring service for {self.target_project_root}")

        # Create event handler

        event_handler = TrackedFileHandler(
            target_project_root=self.target_project_root,
            file_scope=self.file_scope,
            debouncer=self.debouncer,
            notification_emitter=self.notification_emitter,
            notification_types=self.notification_types,
        )

        # Create and start observer
        self.observer = Observer()
        self.observer.schedule(
            event_handler, str(self.target_project_root), recursive=True
        )
        self.observer.start()
        self._is_running = True

        logger.info("File monitoring service started")

    def stop(self) -> None:
        """Stop the file monitoring service."""
        if not self._is_running or self.observer is None:
            logger.warning("File monitoring service is not running")
            return

        logger.info("Stopping file monitoring service")
        self.observer.stop()
        self.observer.join()
        self._is_running = False
        logger.info("File monitoring service stopped")

    def is_running(self) -> bool:
        """Check if the file monitoring service is running.

        Returns:
            True if running, False otherwise.
        """
        return self._is_running

    def get_file_changes_checker(self) -> FileChangesChecker:
        """Get the file changes checker instance.

        Returns:
            The FileChangesChecker instance.
        """
        return self.file_changes_checker
