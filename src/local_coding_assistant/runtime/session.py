from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single conversational message."""

    role: str  # "user" | "assistant" | "system"
    content: str


class ToolCall(BaseModel):
    """A single tool invocation record."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None


class SessionState(BaseModel):
    """Holds per-run conversational context and tool traces.

    A session stores an identifier, message history, tool call records, and
    optional system prompt/metadata. Use `reset()` to clear state between runs.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Conversation
    history: list[Message] = Field(default_factory=list)
    last_query: str | None = None
    system_prompt: str | None = None

    # Tools
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # Arbitrary metadata for user/session context
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        """Initialize a session with optional system prompt and metadata.

        Keyword Args:
            system_prompt: Optional initial system instruction. If provided, a
                corresponding system `Message` is appended to `history`.
            metadata: Optional dictionary of session metadata (e.g., user id).
        """
        super().__init__(**data)
        # If a system prompt is provided and not already reflected in history, add it.
        if self.system_prompt and not any(m.role == "system" for m in self.history):
            self.history.insert(0, Message(role="system", content=self.system_prompt))

    # ── history helpers ──────────────────────────────────────────────────────
    def add_user_message(self, text: str) -> None:
        self.history.append(Message(role="user", content=text))
        self.last_query = text

    def add_assistant_message(self, text: str) -> None:
        self.history.append(Message(role="assistant", content=text))

    def add_tool_message(
        self, name: str, args: dict[str, Any], result: dict[str, Any] | None = None
    ) -> None:
        self.tool_calls.append(ToolCall(name=name, args=args, result=result))

    # ── accessors ───────────────────────────────────────────────────────────
    @property
    def last_user_message(self) -> str | None:
        """Return the latest user message text if present."""
        for m in reversed(self.history):
            if m.role == "user":
                return m.content
        return None

    # ── lifecycle ───────────────────────────────────────────────────────────
    def reset(self, *, keep_system: bool = True) -> None:
        """Reset the session state for a fresh run.

        Args:
            keep_system: When True, preserves the system prompt (and its
                message in history); otherwise clears everything.
        """
        system_msg: Message | None = None
        if keep_system and self.system_prompt:
            for m in self.history:
                if m.role == "system":
                    system_msg = m
                    break
        self.history.clear()
        self.tool_calls.clear()
        self.last_query = None
        if keep_system and self.system_prompt:
            self.history.append(
                system_msg or Message(role="system", content=self.system_prompt)
            )
