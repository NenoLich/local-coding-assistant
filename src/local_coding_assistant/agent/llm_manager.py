"""Agent layer: LLM & reasoning management."""

from collections.abc import Iterable
from typing import Any

from local_coding_assistant.runtime.session import SessionState


class LLMManager:
    """Very small LLM facade used by the Assistant.

    Later you can swap this with a real backend (OpenAI, local model, etc).
    """

    def ask(self, text: str, tools: Iterable[Any] | None = None) -> str:
        """Preserve legacy plain-text interface."""
        tool_count = len(list(tools)) if tools is not None else 0
        return f"[LLMManager] Echo: {text} (tools available: {tool_count})"

    def ask_with_context(
        self,
        session: SessionState,
        *,
        tools: Iterable[Any] | None = None,
        tool_outputs: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str:
        """Accept structured context: conversation history + optional tool outputs.

        For now, this returns a deterministic echo while demonstrating access to context.
        """
        _ = model  # unused placeholder for future selection
        tool_count = len(list(tools)) if tools is not None else 0
        text = session.last_query or ""
        suffix = " with tool outputs" if tool_outputs else ""
        return f"[LLMManager] Echo: {text}{suffix} (tools available: {tool_count})"
