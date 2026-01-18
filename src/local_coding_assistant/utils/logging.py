import datetime as dt
import json
import logging
import sys
from decimal import Decimal
from enum import Enum
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, TextIO
from uuid import UUID

import structlog
from rich.pretty import pretty_repr
from structlog.dev import (
    Column,
    ConsoleRenderer,
    KeyValueColumnFormatter,
    LogLevelColumnFormatter,
)
from structlog.processors import (
    CallsiteParameter,
    CallsiteParameterAdder,
    JSONRenderer,
    StackInfoRenderer,
    TimeStamper,
    UnicodeDecoder,
)
from structlog.stdlib import LoggerFactory

# Type aliases
LogLevel = int | str

# Default log levels for third-party libraries
THIRD_PARTY_LOGGERS = {
    "urllib3": logging.WARNING,
    "urllib3.connectionpool": logging.WARNING,
    "docker": logging.WARNING,
    "docker.auth": logging.WARNING,
    "docker.utils.config": logging.WARNING,
    "asyncio": logging.WARNING,
}

LEVEL_EMOJI_MAP = {
    "debug": "🐞",
    "info": "💡",
    "warning": "⚠️",
    "error": "🔥",
    "critical": "💀",
}
LEVEL_PAD_LENGTH = len("critical")


def pretty_value_formatter(value):
    """Safely format a value for logging, handling Pydantic models and other types."""
    try:
        if hasattr(value, "model_dump"):
            # Handle Pydantic v2 models
            formatted = pretty_repr(value.model_dump(), indent_size=4, expand_all=True)
            return f"\n{formatted}"
        elif hasattr(value, "dict"):
            # Handle Pydantic v1 models
            formatted = pretty_repr(value.dict(), indent_size=4, expand_all=True)
            return f"\n{formatted}"
        elif isinstance(value, (dict, list, tuple, set)):
            # Handle standard containers
            formatted = pretty_repr(value, indent_size=4, expand_all=True)
            return f"\n{formatted}"
        return repr(value)
    except Exception as e:
        return f"<Error formatting value: {e!s}>"


def unescape_newlines(logger, method_name, event_dict):
    # ConsoleRenderer returns a string, not a dict, if it's the last processor
    if isinstance(event_dict, str):
        # Replace the literal string "\n" with an actual newline character
        return event_dict.replace("\\n", "\n")
    return event_dict


def _truncate_logger_name(name: str, max_length: int = 21) -> str:
    """Truncate logger name for better console output."""
    if not name or len(name) <= max_length:
        return name
    return f"{name[: max_length - 3]}..."


def structlog_default_serializer(obj: Any) -> Any:
    """
    Default serializer for structlog that handles various Python types.
    This is used as the 'default' hook for json.dumps.

    Args:
        obj: The object to serialize

    Returns:
        A JSON-serializable representation of the object
    """
    # 1. Handle common objects that convert directly to string
    if isinstance(obj, (Path, UUID, dt.timedelta, bytes, bytearray)):
        return str(obj)

    # 2. Handle Date/Time
    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
        return obj.isoformat()

    # 3. Handle Numeric/Logical types
    if isinstance(obj, Decimal):
        return float(obj)

    if isinstance(obj, Enum):
        return obj.value

    # 4. Handle Collections
    if isinstance(obj, (set, frozenset)):
        return list(obj)

    # 5. Handle Metadata/Classes
    if isinstance(obj, type):
        return f"<class {obj.__module__}.{obj.__name__}>"

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    # Fallback
    try:
        return str(obj)
    except Exception:
        return f"<Unserializable {type(obj).__name__}>"


def _add_emoji(logger, method_name, event_dict):
    level = event_dict.get("level", "").lower()
    event_dict["emoji"] = LEVEL_EMOJI_MAP.get(level, "  ")
    return event_dict


def _get_console_processors() -> list[Any]:
    """Get processors for console output with colors and formatting."""
    styles = ConsoleRenderer.get_default_column_styles(colors=sys.stderr.isatty())

    emoji_formatter = KeyValueColumnFormatter(
        key_style=None,  # This removes "emoji="
        value_style=styles.reset,  # Keeps original emoji color/style
        reset_style=styles.reset,
        value_repr=str,  # Renders as a simple string
    )

    return [
        # Add context variables and standard fields
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        structlog.stdlib.PositionalArgumentsFormatter(),
        TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        CallsiteParameterAdder(
            [
                CallsiteParameter.FILENAME,
                CallsiteParameter.FUNC_NAME,
                CallsiteParameter.LINENO,
            ]
        ),
        # Process the event
        _add_emoji,  # Add emoji based on log level
        _add_truncated_logger_name,
        _add_process_info,
        _filter_sensitive_data,
        # Remove _record and _from_structlog from event_dict
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        # Handle exceptions
        structlog.processors.ExceptionPrettyPrinter(),
        # Render the log
        ConsoleRenderer(
            columns=[
                Column(
                    "timestamp",
                    KeyValueColumnFormatter(
                        key_style=None,
                        value_style=styles.timestamp,
                        reset_style=styles.reset,
                        value_repr=str,
                    ),
                ),
                Column("emoji", emoji_formatter),  # Placed between time and level
                Column(
                    "level",
                    LogLevelColumnFormatter(
                        level_styles=ConsoleRenderer.get_default_level_styles(),
                        reset_style=styles.reset,
                    ),
                ),
                Column(
                    "logger",
                    KeyValueColumnFormatter(
                        key_style=None,
                        value_style=styles.timestamp,
                        reset_style=styles.reset,
                        value_repr=str,
                    ),
                ),
                Column(
                    "event",
                    KeyValueColumnFormatter(
                        key_style=None,
                        value_style=styles.bright,
                        reset_style=styles.reset,
                        value_repr=str,
                    ),
                ),
                Column(
                    "filename",
                    KeyValueColumnFormatter(
                        key_style=styles.timestamp,
                        value_style=styles.timestamp,
                        reset_style=styles.reset,
                        value_repr=str,
                    ),
                ),
                Column(
                    "func_name",
                    KeyValueColumnFormatter(
                        key_style=styles.timestamp,
                        value_style=styles.timestamp,
                        reset_style=styles.reset,
                        value_repr=str,
                    ),
                ),
                Column(
                    "lineno",
                    KeyValueColumnFormatter(
                        key_style=styles.timestamp,
                        value_style=styles.timestamp,
                        reset_style=styles.reset,
                        value_repr=str,
                    ),
                ),
                Column(
                    "pid",
                    KeyValueColumnFormatter(
                        key_style=styles.timestamp,
                        value_style=styles.timestamp,
                        reset_style=styles.reset,
                        value_repr=str,
                    ),
                ),
                Column(
                    "thread",
                    KeyValueColumnFormatter(
                        key_style=styles.timestamp,
                        value_style=styles.timestamp,
                        reset_style=styles.reset,
                        value_repr=str,
                    ),
                ),
                Column(
                    "",
                    KeyValueColumnFormatter(
                        key_style=styles.kv_key,
                        value_style=styles.kv_value,
                        reset_style=styles.reset,
                        value_repr=pretty_value_formatter,
                    ),
                ),
            ],
            colors=sys.stderr.isatty(),
        ),
        unescape_newlines,
    ]


def _get_file_processors(json_format: bool = True) -> list[Any]:
    """Get processors for file output with more detailed information."""
    processors = [
        # Add context variables and standard fields
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,  # Full logger name in files
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        structlog.stdlib.PositionalArgumentsFormatter(),
        # Add structured information
        TimeStamper(fmt="iso"),  # ISO format for better parsing
        _add_process_info,
        # Process the event
        _filter_sensitive_data,
        CallsiteParameterAdder(
            [
                CallsiteParameter.FILENAME,
                CallsiteParameter.FUNC_NAME,
                CallsiteParameter.LINENO,
            ]
        ),
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        # Handle exceptions
        structlog.processors.dict_tracebacks,
    ]

    # Add the appropriate renderer
    if json_format:

        def sort_dict_keys(logger, method_name, event_dict):
            """Sort dictionary keys for consistent JSON output."""
            if isinstance(event_dict, dict):
                return {
                    k: sort_dict_keys(logger, method_name, v)
                    if isinstance(v, dict)
                    else v
                    for k, v in sorted(event_dict.items())
                }
            return event_dict

        processors.extend(
            [
                sort_dict_keys,
                JSONRenderer(
                    serializer=json.dumps,
                    default=structlog_default_serializer,
                    indent=2,
                    sort_keys=True,
                ),
                unescape_newlines,
            ]
        )
    else:
        processors.extend(
            [
                structlog.processors.KeyValueRenderer(
                    key_order=["timestamp", "level", "logger", "event"],
                    drop_missing=True,
                ),
                unescape_newlines,
            ]
        )

    return processors


def _add_truncated_logger_name(logger, method_name, event_dict):
    """Add truncated logger name to the event dict."""
    name = event_dict.get("logger", "root")

    event_dict["logger"] = _truncate_logger_name(name)
    return event_dict


def _add_process_info(logger, method_name, event_dict):
    """Add process/thread information to the event dict."""
    import os
    import threading

    event_dict["pid"] = os.getpid()
    event_dict["thread"] = threading.current_thread().name
    return event_dict


def _filter_sensitive_data(logger, method_name, event_dict):
    """
    Filter out sensitive data from logs, including nested dictionaries and lists.

    Args:
        logger: The logger instance
        method_name: The log method name (e.g., 'info', 'error')
        event_dict: The event dictionary to filter

    Returns:
        The filtered event dictionary with sensitive data redacted
    """
    sensitive_keys = [
        "password",
        "token",
        "secret",
        "key",
        "api_key",
        "auth",
        "credential",
    ]

    def filter_value(value):
        """Recursively filter sensitive data in nested structures."""
        if isinstance(value, dict):
            return {
                k: "***REDACTED***"
                if any(sensitive in k.lower() for sensitive in sensitive_keys)
                else filter_value(v)
                for k, v in value.items()
            }
        elif isinstance(value, (list, tuple)):
            return [filter_value(item) for item in value]
        return value

    # Process the event dictionary
    for key in list(event_dict.keys()):
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            event_dict[key] = "***REDACTED***"
        else:
            event_dict[key] = filter_value(event_dict[key])

    return event_dict


def _create_file_handler(
    log_path: Path,
    level: int,
    json_format: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    time_rotation: str | None = None,
) -> logging.Handler:
    """
    Create a file handler with the specified configuration.

    Args:
        log_path: Path to the log file
        level: Logging level
        json_format: Whether to use JSON format
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        time_rotation: Time-based rotation (e.g., 'midnight', 'D', 'H')

    Returns:
        Configured logging.Handler instance
    """
    try:
        # Ensure the log directory exists
        log_path = Path(log_path).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create the appropriate handler
        if time_rotation:
            handler = TimedRotatingFileHandler(
                str(log_path),
                when=time_rotation,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            handler = RotatingFileHandler(
                str(log_path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )

        handler.setLevel(level)

        # Get the file processors and create formatter
        processors = _get_file_processors(json_format)
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=processors[:-1],  # All except the last (renderer)
            processors=[processors[-1]],  # Just the renderer
        )

        handler.setFormatter(formatter)
        return handler

    except Exception as e:
        # Fallback to console logging if file logging fails
        print(f"Failed to configure file logging: {e}", file=sys.stderr)
        return logging.NullHandler()


def setup_logging(
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
    log_file: str | Path | None = None,
    file_level: int | str | None = None,
    third_party_levels: dict[str, int | str] | None = None,
    json_format: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    time_rotation: str | None = None,
    enable_console: bool = True,
) -> structlog.BoundLogger:
    """
    Configure structured logging for the application using structlog.

    Args:
        level: Root logger level (default: INFO)
        stream: Output stream (default: sys.stderr)
        log_file: Path to the log file. If None, file logging is disabled.
        file_level: Log level for file handler (default: same as console)
        third_party_levels: Custom log levels for third-party libraries
        json_format: Whether to use JSON format for file logs (default: False)
        max_bytes: Max file size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        time_rotation: Time-based rotation (e.g., 'midnight', 'D' for daily, 'H' for hourly)
        enable_console: Whether to enable console logging (default: True)

    Returns:
        Configured structlog logger instance
    "
    """
    try:
        # Convert string log level to int if needed
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Clear existing handlers to avoid duplicate logs
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()

        if enable_console:
            console_handler = logging.StreamHandler(stream or sys.stderr)
            console_handler.setLevel(level)

            # Apply console-specific processors
            console_processors = _get_console_processors()
            console_handler.setFormatter(
                structlog.stdlib.ProcessorFormatter(
                    foreign_pre_chain=console_processors[:-1],
                    processors=[console_processors[-1]],
                )
            )
            root_logger.addHandler(console_handler)

        if log_file:
            log_file_path = Path(log_file).resolve()
            # Ensure level is an integer for the file handler
            final_file_level = file_level or level
            if isinstance(final_file_level, str):
                final_file_level = getattr(
                    logging, final_file_level.upper(), logging.INFO
                )
            file_handler = _create_file_handler(
                log_path=log_file_path,
                level=final_file_level,
                json_format=json_format,
                max_bytes=max_bytes,
                backup_count=backup_count,
                time_rotation=time_rotation,
            )
            if not isinstance(file_handler, logging.NullHandler):
                root_logger.addHandler(file_handler)

        third_party_levels = third_party_levels or THIRD_PARTY_LOGGERS
        for logger_name, logger_level in third_party_levels.items():
            if isinstance(logger_level, str):
                logger_level = getattr(logging, logger_level.upper(), logging.INFO)
            logging.getLogger(logger_name).setLevel(logger_level)

        structlog.configure(
            processors=[
                StackInfoRenderer(),
                UnicodeDecoder(),
                structlog.stdlib.render_to_log_kwargs,
            ],  # Processors are handled by the handlers
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.make_filtering_bound_logger(level),
            cache_logger_on_first_use=True,
        )

        logger = structlog.get_logger()
        logger.info(
            "Logging configured",
            console=enable_console,
            file=str(log_file) if log_file else None,
        )

        return logger

    except Exception as e:
        # Fallback to basic logging if configuration fails
        logging.basicConfig(level=logging.INFO)
        print(f"[ERROR] Failed to configure logging: {e!s}", file=sys.stderr)

        # Return a basic logger that won't fail
        return structlog.wrap_logger(
            structlog.PrintLogger(file=sys.stderr),
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                ConsoleRenderer(
                    colors=sys.stderr.isatty(),
                    level_styles=structlog.dev.ConsoleRenderer.get_default_level_styles(),
                    pad_event=40,  # Pad the event field for better alignment
                ),
            ],
        )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a structlog logger instance with the given name."""
    return structlog.get_logger(name or "")
