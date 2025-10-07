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
    """Custom formatter that adds emojis and subsystem context."""

    def format(self, record: logging.LogRecord) -> str:
        emoji = EMOJI_MAP.get(record.levelname, "")
        module = record.name.split(".")[-1]
        return f"{emoji} [{record.levelname:<8}] ({module}) {record.getMessage()}"


def setup_logging(level: int = logging.INFO, stream: TextIO | None = None) -> None:
    """Configure root logger with emoji formatter."""
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(EmojiFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    logging.getLogger(__name__).debug("Logging initialized.")


# Helper for subsystems
def get_logger(name: str) -> logging.Logger:
    """Get a subsystem logger."""
    return logging.getLogger(f"local_coding_assistant.{name}")
