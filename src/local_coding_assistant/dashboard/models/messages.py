"""Pydantic models for transcript messages."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MessageEntry(BaseModel):
    """A transcript message, clickable from session/turn/frame detail pages."""

    seq: int
    role: str  # user | assistant | tool | system
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    turn_id: str
    frame_id: str | None = None
    turn_index: int | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageDetail(BaseModel):
    """Full message detail including raw payload."""

    message: MessageEntry
    raw_payload: dict[str, Any] = Field(default_factory=dict)
