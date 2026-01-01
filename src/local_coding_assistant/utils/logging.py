import logging
import sys
from typing import ClassVar, TextIO

# Default log levels for third-party libraries
THIRD_PARTY_LOGGERS = {
    "urllib3": logging.WARNING,
    "urllib3.connectionpool": logging.WARNING,
    "docker": logging.WARNING,
    "docker.auth": logging.WARNING,
    "docker.utils.config": logging.WARNING,
    "asyncio": logging.WARNING,
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors, emojis, and aligned log levels."""

    # Colors for different log levels
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[31;1m",  # Bright Red
    }

    # Emojis for different log levels
    EMOJIS: ClassVar[dict[str, str]] = {
        "DEBUG": "🐞",
        "INFO": "💡",
        "WARNING": "⚠️",
        "ERROR": "🔥",
        "CRITICAL": "💀",
    }

    # Fixed width for log level display
    LEVEL_LENGTH: ClassVar[int] = 8

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        emoji = self.EMOJIS.get(levelname, "")
        color = self.COLORS.get(levelname, "\033[0m")

        # Format the level name with consistent width
        level = f"{color}{levelname:^{self.LEVEL_LENGTH}}\033[0m"

        # Format the logger name to be more compact
        logger_name = record.name
        if len(logger_name) > 20:  # Truncate long logger names
            parts = logger_name.split(".")
            if len(parts) > 2:
                logger_name = f"{parts[0]}.{parts[-1]}"
            if len(logger_name) > 20:
                logger_name = logger_name[:20] + "..."

        # Build the formatted message
        record.levelname = f"{emoji} [{level}]"
        record.name = f"\033[90m{logger_name}\033[0m"  # Dim the logger name

        # Use the parent's formatter to handle the actual formatting
        return super().format(record)


def setup_logging(
    level: int | str = logging.INFO,
    format_str: str = "%(levelname)s %(name)s: %(message)s",
    stream: TextIO | None = None,
    log_file: str | None = None,
    file_level: int | str | None = None,
    third_party_levels: dict[str, int | str] | None = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Root logger level (default: INFO)
        format_str: Log format string
        stream: Output stream (default: sys.stderr)
        log_file: Path to log file (optional)
        file_level: Log level for file handler (default: same as console)
        third_party_levels: Custom log levels for third-party libraries
    """
    # Convert string levels to logging constants
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if file_level and isinstance(file_level, str):
        file_level = getattr(logging, file_level.upper(), level)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(stream or sys.stderr)
    console_handler.setLevel(level)
    formatter = ColoredFormatter(format_str)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level or level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Configure third-party loggers
    third_party_config = THIRD_PARTY_LOGGERS.copy()
    if third_party_levels:
        third_party_config.update(third_party_levels)

    for name, lvl in third_party_config.items():
        if isinstance(lvl, str):
            lvl = getattr(logging, lvl.upper(), logging.WARNING)
        logging.getLogger(name).setLevel(lvl)

    # Set up root logger
    logging.info("Logging configured at level %s", logging.getLevelName(level))


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)
