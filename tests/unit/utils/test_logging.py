import json
import logging

from unittest.mock import patch, MagicMock

from structlog.testing import capture_logs

from local_coding_assistant.utils.logging import (
    setup_logging,
    get_logger,
    structlog_default_serializer,
    _add_emoji,
    _filter_sensitive_data,
    _truncate_logger_name,
    _add_process_info,
    LEVEL_EMOJI_MAP,
)


def test_add_emoji():
    """Test that emojis are correctly added based on log level."""
    event = {"event": "test"}

    # Test all log levels
    for level, emoji in LEVEL_EMOJI_MAP.items():
        processed = _add_emoji(None, None, {"event": "test", "level": level})
        assert processed["emoji"] == emoji, f"Failed for level: {level}"

    # Test with unknown level
    processed = _add_emoji(None, None, {"event": "test", "level": "unknown"})
    assert processed["emoji"] == "  "  # Default empty emoji


def test_truncate_logger_name():
    """Test that logger names are properly truncated."""
    assert _truncate_logger_name("short.name") == "short.name"
    assert (
        _truncate_logger_name("a.very.long.logger.name.that.exceeds.max.length")
        == "a.very.long.logger..."
    )
    assert _truncate_logger_name("") == ""
    assert _truncate_logger_name(None) is None


def test_add_process_info():
    """Test that process and thread info is added to log entries."""
    with (
        patch("os.getpid", return_value=12345),
        patch("threading.current_thread") as mock_thread,
    ):
        mock_thread.return_value = MagicMock()
        mock_thread.return_value.name = "TestThread"
        mock_thread.return_value.ident = 1234

        event = _add_process_info(None, None, {"event": "test"})
        assert event["pid"] == 12345
        assert "thread" in event  # Don't check exact value as it might be formatted


def test_filter_sensitive_data():
    """Test that sensitive data is properly filtered from logs."""
    # Test direct filtering
    test_data = {
        "username": "user123",
        "password": "s3cr3t",
        "api_key": "key-123",
        "auth_token": "token123",
        "safe_data": "this is safe",
    }

    # Apply the filter directly
    filtered = _filter_sensitive_data(None, None, test_data.copy())

    # Check sensitive fields are redacted
    assert filtered["password"] == "***REDACTED***"
    assert filtered["api_key"] == "***REDACTED***"
    assert filtered["auth_token"] == "***REDACTED***"

    # Check non-sensitive fields remain unchanged
    assert filtered["username"] == "user123"
    assert filtered["safe_data"] == "this is safe"


def test_structlog_default_serializer():
    """Test the default serializer for non-standard types."""
    from datetime import datetime, date, time, timedelta
    from pathlib import Path
    from decimal import Decimal
    from uuid import UUID
    from enum import Enum

    # Test datetime
    dt = datetime(2025, 1, 1, 12, 0, 0)
    assert structlog_default_serializer(dt) == "2025-01-01T12:00:00"

    # Test date
    d = date(2025, 1, 1)
    assert structlog_default_serializer(d) == "2025-01-01"

    # Test time
    t = time(12, 0, 0)
    assert structlog_default_serializer(t) == "12:00:00"

    # Test timedelta
    td = timedelta(days=1, hours=2, minutes=30)
    assert structlog_default_serializer(td) == str(td)

    # Test bytes
    assert structlog_default_serializer(b"hello") == "b'hello'"

    # Test Path
    assert structlog_default_serializer(Path("test.txt")) == "test.txt"

    # Test Decimal
    assert structlog_default_serializer(Decimal("3.14")) == 3.14

    # Test UUID
    test_uuid = "12345678-1234-5678-1234-567812345678"
    assert structlog_default_serializer(UUID(test_uuid)) == test_uuid

    # Test Enum
    class TestEnum(Enum):
        TEST = "test_value"

    assert structlog_default_serializer(TestEnum.TEST) == "test_value"

    # Test set
    assert structlog_default_serializer({1, 2, 3}) == [1, 2, 3]

    # Test dictionary with non-string keys
    assert structlog_default_serializer({1: "one"}) == "{1: 'one'}"

    # Test unserializable object
    class Unserializable:
        def __init__(self):
            self.value = "test"

        def __repr__(self):
            raise RuntimeError("Fail")

    # The serializer should handle the object by converting it to a dict
    assert structlog_default_serializer(Unserializable()) == {"value": "test"}


def test_setup_logging_basic():
    """Test basic logging setup and logger functionality."""
    # Setup logging with debug level
    logger = setup_logging(level=logging.DEBUG)

    # Get a logger and verify it works
    test_logger = get_logger("test.logger")
    assert hasattr(test_logger, "debug")
    assert hasattr(test_logger, "info")
    assert hasattr(test_logger, "warning")
    assert hasattr(test_logger, "error")
    assert hasattr(test_logger, "critical")

    # Test logging at different levels
    with capture_logs() as captured:
        test_logger.debug("Debug message", debug_data="debug")
        test_logger.info("Info message", info_data="info")
        test_logger.warning("Warning message", warn_data="warn")
        test_logger.error("Error message", error_data="error")

    # Verify log entries
    assert len(captured) >= 4
    assert any(e.get("event") == "Info message" for e in captured)
    assert any(e.get("info_data") == "info" for e in captured)


def test_file_logging(tmp_path):
    """Test logging to a file with JSON format."""
    log_file = tmp_path / "test.log"

    # Setup logging with file output
    setup_logging(
        level=logging.INFO,
        log_file=str(log_file),
        json_format=True,
        enable_console=False,
    )

    # Log a test message
    logger = get_logger("file.logger")
    logger.info("File log test", test_data="success")

    # Ensure logs are flushed
    logging.shutdown()

    # Read the log file
    content = log_file.read_text(encoding="utf-8")
    assert content.strip()  # File should not be empty

    # Print content for debugging
    print("\nLog file content:")
    print(content)

    # Parse the log entries
    entries = []
    # The log file contains pretty-printed JSON objects, so we need to parse them correctly
    current_entry = []
    in_json = False

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("{") and line.endswith("}"):
            # Single-line JSON object
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Failed to parse single-line entry: {line}")
                raise
        elif line.startswith("{"):
            # Start of a multi-line JSON object
            current_entry = [line]
            in_json = True
        elif in_json and line.endswith("}"):
            # End of a multi-line JSON object
            current_entry.append(line)
            try:
                entries.append(json.loads("\n".join(current_entry)))
            except json.JSONDecodeError as e:
                print(f"Failed to parse multi-line entry: {current_entry}")
                raise
            in_json = False
            current_entry = []
        elif in_json:
            # Middle of a multi-line JSON object
            current_entry.append(line)

    # Find our test log entry
    entry = next(
        (
            e
            for e in entries
            if isinstance(e, dict) and e.get("event") == "File log test"
        ),
        None,
    )
    assert entry is not None, f"Test log entry not found in: {entries}"
    assert entry.get("event") == "File log test"
    assert entry.get("test_data") == "success"
    assert "timestamp" in entry
    assert "logger" in entry
    assert entry.get("logger") == "file.logger"


def test_third_party_logging():
    """Test that third-party loggers respect the configured levels."""
    # Get current loggers to restore later
    original_levels = {}
    for name in ["test.thirdparty", "test.another", "test.unknown"]:
        logger = logging.getLogger(name)
        original_levels[name] = logger.level

    try:
        # Setup with custom third-party levels
        setup_logging(
            level=logging.WARNING,
            third_party_levels={"test.thirdparty": "DEBUG", "test.another": "ERROR"},
        )

        # Get loggers and verify their levels
        assert logging.getLogger("test.thirdparty").level == logging.DEBUG
        assert logging.getLogger("test.another").level == logging.ERROR

        # The default level for unknown loggers should be NOTSET (0)
        # which means it will inherit from the root logger (WARNING in this case)
        assert logging.getLogger("test.unknown").level == logging.NOTSET
    finally:
        # Restore original levels to avoid affecting other tests
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)


def test_log_rotation(tmp_path):
    """Test log rotation functionality."""
    log_file = tmp_path / "rotating.log"

    # Setup with small max_bytes to trigger rotation
    setup_logging(
        level=logging.INFO,
        log_file=log_file,
        max_bytes=100,  # Very small to trigger rotation
        backup_count=2,
        enable_console=False,
    )

    logger = get_logger("rotation.test")

    # Write enough data to trigger rotation
    for i in range(10):
        logger.info(f"Test message {i}", data="x" * 50)  # Each log is > 50 bytes

    # Check that log files were created
    log_files = list(tmp_path.glob("rotating.log*"))
    assert len(log_files) > 1  # Should have at least the main log and one backup
    assert any(
        ".1" in str(f) for f in log_files
    )  # Should have at least one rotated file


def test_sensitive_data_filtering():
    """Test that sensitive data is properly filtered in logs."""
    # Test direct filtering
    test_data = {
        "username": "user123",
        "password": "s3cr3t",
        "api_key": "key-123",
        "auth_token": "token123",
        "safe_data": "this is safe",
    }

    # Apply the filter directly
    filtered = _filter_sensitive_data(None, None, test_data.copy())

    # Check sensitive fields are redacted
    assert filtered["password"] == "***REDACTED***"
    assert filtered["api_key"] == "***REDACTED***"
    assert filtered["auth_token"] == "***REDACTED***"

    # Check non-sensitive fields remain unchanged
    assert filtered["username"] == "user123"
    assert filtered["safe_data"] == "this is safe"


def test_logging_configuration_fallback():
    """Test that logging falls back to basic config on error."""
    # Get current handlers to restore later
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    original_level = root_logger.level

    try:
        # Patch structlog.configure to raise an exception
        with patch("structlog.configure", side_effect=Exception("Test error")):
            # This should not raise an exception
            logger = setup_logging(level=logging.INFO)

            # Should still be able to log
            logger.info("Test message")

        # The logger should still be functional
        # We'll test by checking if the root logger has handlers
        assert len(logging.root.handlers) > 0, "No handlers found on root logger"

        # Also verify the logger has the correct level
        assert logging.root.level == logging.INFO
    finally:
        # Clean up handlers
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        for h in original_handlers:
            root_logger.addHandler(h)
        root_logger.setLevel(original_level)
