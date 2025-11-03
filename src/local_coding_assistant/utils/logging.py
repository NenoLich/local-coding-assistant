import logging
import sys
from typing import TextIO

# Emojis per level
EMOJI_MAP = {
    "DEBUG": "🐞",
    "INFO": "💡",
    "WARNING": "⚠️",
    "ERROR": "🔥",
    "CRITICAL": "💀",
}


class EmojiFormatter(logging.Formatter):
    """Custom formatter that adds emojis, subsystem context, and extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        emoji = EMOJI_MAP.get(record.levelname, "")
        log_line = (
            f"{emoji} [{record.levelname:<8}] ({record.name}) {record.getMessage()}"
        )

        # Check for extra fields in the record's __dict__
        # Create a minimal LogRecord to get default attributes
        default_attrs = logging.LogRecord(
            name="",
            level=logging.NOTSET,
            pathname="",
            lineno=0,
            msg="",
            args=(),
            exc_info=None,
        ).__dict__

        extra_attrs = {
            k: v
            for k, v in record.__dict__.items()
            if k not in default_attrs and not k.startswith("_")
        }

        if extra_attrs:
            extra_str = " ".join(f"{k}={v!r}" for k, v in extra_attrs.items())
            log_line = f"{log_line} | {extra_str}"

        return log_line


def setup_logging(level: int = logging.INFO, stream: TextIO | None = None) -> None:
    """Configure root logger with emoji formatter."""
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(EmojiFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


# Helper for subsystems
def get_logger(name: str) -> logging.Logger:
    """Get a subsystem logger."""
    return logging.getLogger(f"{name}")
