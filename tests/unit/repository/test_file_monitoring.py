"""Unit tests for file monitoring components."""

import time
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.repository.file_monitoring import (
    Debouncer,
    FileChangeEvent,
    FileChangeType,
    FileMonitoringService,
    NotificationEmitter,
    TrackedFileHandler,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("test_file_monitoring")


class TestNotificationEmitter:
    """Tests for NotificationEmitter."""

    def test_init(self):
        """Test initialization of NotificationEmitter."""
        emitter = NotificationEmitter()
        assert emitter._callbacks == []

    def test_register_callback(self):
        """Test registering a callback."""
        emitter = NotificationEmitter()
        callback = MagicMock()
        emitter.register_callback(callback)
        assert callback in emitter._callbacks

    def test_emit(self):
        """Test emitting a notification."""
        emitter = NotificationEmitter()
        callback = MagicMock()
        emitter.register_callback(callback)

        event = FileChangeEvent(
            path="/test/file.py",
            change_type=FileChangeType.MODIFIED,
            timestamp=time.time(),
            is_tracked=True,
        )
        emitter.emit(event)

        callback.assert_called_once_with(event)

    def test_emit_with_error(self):
        """Test emitting a notification when callback raises an error."""
        emitter = NotificationEmitter()
        callback = MagicMock(side_effect=Exception("Test error"))
        emitter.register_callback(callback)

        event = FileChangeEvent(
            path="/test/file.py",
            change_type=FileChangeType.MODIFIED,
            timestamp=time.time(),
            is_tracked=True,
        )

        # Should not raise exception
        emitter.emit(event)


class TestDebouncer:
    """Tests for Debouncer."""

    def test_init(self):
        """Test initialization of Debouncer."""
        debouncer = Debouncer(delay=1.0)
        assert debouncer.delay == 1.0
        assert debouncer._timers == {}

    def test_debounce(self):
        """Test debouncing a file change."""
        debouncer = Debouncer(delay=0.1)
        callback = MagicMock()
        file_path = "/test/file.py"

        debouncer.debounce(file_path, callback)

        # Wait for debounce delay
        time.sleep(0.2)

        callback.assert_called_once_with(file_path)

    def test_debounce_coalesce(self):
        """Test that rapid changes are coalesced."""
        debouncer = Debouncer(delay=0.2)
        callback = MagicMock()
        file_path = "/test/file.py"

        # Trigger multiple rapid changes
        debouncer.debounce(file_path, callback)
        time.sleep(0.05)
        debouncer.debounce(file_path, callback)
        time.sleep(0.05)
        debouncer.debounce(file_path, callback)

        # Wait for debounce delay (with extra buffer for thread scheduling)
        time.sleep(0.5)

        # Should only be called once
        assert callback.call_count == 1


class TestTrackedFileHandler:
    """Tests for TrackedFileHandler."""

    @pytest.fixture
    def temp_target_project_root(self, tmp_path):
        """Create a temporary project root."""
        (tmp_path / "project").mkdir(parents=True, exist_ok=True)
        return tmp_path / "project"

    @pytest.fixture
    def mock_debouncer(self):
        """Create a mock debouncer."""
        return MagicMock(spec=Debouncer)

    @pytest.fixture
    def mock_emitter(self):
        """Create a mock notification emitter."""
        return MagicMock(spec=NotificationEmitter)

    @pytest.fixture
    def mock_file_scope(self):
        """Create a mock file scope."""
        from local_coding_assistant.repository.file_scope import FileScope

        return MagicMock(spec=FileScope)

    def test_init(
        self, temp_target_project_root, mock_file_scope, mock_debouncer, mock_emitter
    ):
        """Test initialization of TrackedFileHandler."""
        handler = TrackedFileHandler(
            target_project_root=temp_target_project_root,
            file_scope=mock_file_scope,
            debouncer=mock_debouncer,
            notification_emitter=mock_emitter,
        )
        assert handler.target_project_root == temp_target_project_root
        assert handler.file_scope == mock_file_scope
        assert handler.debouncer == mock_debouncer
        assert handler.notification_emitter == mock_emitter

    def test_is_tracked_with_cache(
        self, temp_target_project_root, mock_file_scope, mock_debouncer, mock_emitter
    ):
        """Test tracked check with cached file list."""
        handler = TrackedFileHandler(
            target_project_root=temp_target_project_root,
            file_scope=mock_file_scope,
            debouncer=mock_debouncer,
            notification_emitter=mock_emitter,
        )

        # Create a test file
        test_file = temp_target_project_root / "test.py"
        test_file.write_text("content")

        # Mock the tracked files cache
        handler._tracked_files = {test_file}

        assert handler._is_tracked(str(test_file)) is True
        assert (
            handler._is_tracked(str(temp_target_project_root / "not_tracked.py"))
            is False
        )

    def test_handle_file_change_not_tracked(
        self, temp_target_project_root, mock_file_scope, mock_debouncer, mock_emitter
    ):
        """Test handling file change for non-tracked file."""
        handler = TrackedFileHandler(
            target_project_root=temp_target_project_root,
            file_scope=mock_file_scope,
            debouncer=mock_debouncer,
            notification_emitter=mock_emitter,
        )
        handler._tracked_files = set()  # Empty cache means no tracked files

        mock_event = MagicMock()
        mock_event.src_path = "/test/file.py"

        handler._handle_file_change(mock_event, FileChangeType.MODIFIED)

        # Should not call debouncer
        mock_debouncer.debounce.assert_not_called()


class TestFileMonitoringService:
    """Tests for FileMonitoringService."""

    @pytest.fixture
    def temp_target_project_root(self, tmp_path):
        """Create a temporary project root."""
        (tmp_path / "project").mkdir(parents=True, exist_ok=True)
        return tmp_path / "project"

    def test_init(self, temp_target_project_root):
        """Test initialization of FileMonitoringService."""
        service = FileMonitoringService(
            target_project_root=temp_target_project_root,
            debounce_window=1.0,
        )
        assert service.target_project_root == temp_target_project_root
        assert service.debounce_window == 1.0
        assert service._is_running is False

    def test_register_notification_callback(self, temp_target_project_root):
        """Test registering a notification callback."""
        service = FileMonitoringService(target_project_root=temp_target_project_root)
        callback = MagicMock()
        service.register_notification_callback(callback)
        assert callback in service.notification_emitter._callbacks

    def test_start(self, temp_target_project_root):
        """Test starting the file monitoring service."""
        service = FileMonitoringService(target_project_root=temp_target_project_root)
        service.start()
        assert service._is_running is True
        service.stop()

    def test_stop(self, temp_target_project_root):
        """Test stopping the file monitoring service."""
        service = FileMonitoringService(target_project_root=temp_target_project_root)
        service.start()
        service.stop()
        assert service._is_running is False

    def test_start_already_running(self, temp_target_project_root):
        """Test starting when already running."""
        service = FileMonitoringService(target_project_root=temp_target_project_root)
        service.start()
        service.start()  # Should not raise error
        service.stop()

    def test_stop_not_running(self, temp_target_project_root):
        """Test stopping when not running."""
        service = FileMonitoringService(target_project_root=temp_target_project_root)
        service.stop()  # Should not raise error

    def test_is_running(self, temp_target_project_root):
        """Test checking if service is running."""
        service = FileMonitoringService(target_project_root=temp_target_project_root)
        assert service.is_running() is False
        service.start()
        assert service.is_running() is True
        service.stop()
        assert service.is_running() is False
