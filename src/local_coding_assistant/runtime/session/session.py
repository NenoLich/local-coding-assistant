from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from local_coding_assistant.runtime.context.profile import ContextProfile

if TYPE_CHECKING:
    from local_coding_assistant.runtime.session import RuntimeTurn


class Message(BaseModel):
    """A single conversational message."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str | list[dict[str, Any]] | None = None
    id: str | None = None  # Optional message identifier for continuation prompts
    tool_call_id: str | None = None  # For tool messages, the tool call ID
    tool_calls: list[dict[str, Any]] | None = (
        None  # For assistant messages, the tool calls
    )
    seq: int | None = Field(default=None, exclude=True)


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
    history_sequence: int = 0
    last_query: str | None = None
    system_prompt: str | None = None

    # Tools
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # Turns cache (turn_id -> RuntimeTurn) to avoid frequent store lookups.
    # Typed as Any on purpose: RuntimeTurn is a dataclass, and a precise
    # annotation would make SessionState's pydantic schema unresolvable
    # (blocking every direct SessionState construction).
    turns: dict[str, Any] = Field(default_factory=dict, exclude=True)

    # Context profile (session-level baseline for context resolution)
    context_profile: ContextProfile

    # Arbitrary metadata for user/session context
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        """Initialize a session with optional system prompt and metadata.

        Keyword Args:
            system_prompt: Optional initial system instruction. If provided, a
                corresponding system `Message` is appended to `history`.
            metadata: Optional dictionary of session metadata (e.g., user id).
        """
        history_sequence_supplied = "history_sequence" in data
        super().__init__(**data)
        # If a system prompt is provided and not already reflected in history, add it.
        if self.system_prompt and not any(m.role == "system" for m in self.history):
            self.history.insert(0, Message(role="system", content=self.system_prompt))
        if not history_sequence_supplied:
            self.history_sequence = max(
                len(self.history),
                max((message.seq or 0 for message in self.history), default=0),
            )

    def set_snapshot_profile(self, profile: ContextProfile) -> None:
        self.context_profile = profile

    # ── history helpers ──────────────────────────────────────────────────────
    def add_user_message(self, text: str) -> None:
        self.append_transcript_message(Message(role="user", content=text))
        self.last_query = text

    def add_assistant_message(
        self, content: str | None = None, tool_call: dict[str, Any] | None = None
    ) -> None:
        if tool_call:
            self.append_transcript_message(
                Message(role="assistant", content=content, tool_calls=[tool_call])
            )
        else:
            self.append_transcript_message(Message(role="assistant", content=content))

    def add_tool_message(self, call_id: str, result: str) -> None:
        """Add a tool execution result message."""
        self.append_transcript_message(
            Message(
                role="tool",
                tool_call_id=call_id,
                content=result,
            )
        )

    def add_system_message(self, content: str) -> None:
        """Add a system message to the history."""
        self.append_transcript_message(Message(role="system", content=content))

    def append_transcript_message(self, message: Message) -> None:
        self.history_sequence += 1
        message.seq = self.history_sequence
        self.history.append(message)

    # ── turn cache helpers ────────────────────────────────────────────────────
    def add_turn(self, turn: RuntimeTurn) -> None:
        """Add a turn to the cache."""
        self.turns[turn.turn_id] = turn

    def get_turn(self, turn_id: str) -> RuntimeTurn | None:
        """Get a turn from cache, returning None if not found."""
        return self.turns.get(turn_id)

    def get_turn_by_index(self, turn_index: int) -> RuntimeTurn | None:
        """Get a turn by its index from cache."""
        for turn in self.turns.values():
            if turn.turn_index == turn_index:
                return turn
        return None

    def get_last_turn(self) -> RuntimeTurn | None:
        """Get the last turn from cache (highest turn_index)."""
        if not self.turns:
            return None
        return max(self.turns.values(), key=lambda t: t.turn_index)

    # ── accessors ───────────────────────────────────────────────────────────
    @property
    def last_user_message(self) -> str | None:
        """Return the latest user message text if present."""
        for m in reversed(self.history):
            if m.role == "user":
                if isinstance(m.content, str):
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
        self.history_sequence = 0
        if keep_system and self.system_prompt:
            self.append_transcript_message(
                system_msg or Message(role="system", content=self.system_prompt)
            )
