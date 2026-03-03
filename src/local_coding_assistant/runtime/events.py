"""
Event definitions for streaming execution in the Local Coding Assistant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class EventType(Enum):
    """Types of events emitted during execution."""

    TURN_START = "turn_start"
    FRAME_START = "frame_start"
    LLM_START = "llm_start"
    LLM_CHUNK = "llm_chunk"
    LLM_COMPLETE = "llm_complete"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    FRAME_COMPLETE = "frame_complete"
    TURN_COMPLETE = "turn_complete"
    SESSION_START = "session_start"
    SESSION_RESUME = "session_resume"
    ERROR = "error"


@dataclass
class ExecutionEvent:
    """Event emitted during streaming execution."""

    type: EventType
    session_id: str
    frame_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Ensure data is a copy to prevent external modifications."""
        self.data = self.data.copy()
