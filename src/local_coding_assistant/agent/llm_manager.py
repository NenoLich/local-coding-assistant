"""Agent layer: LLM & reasoning management."""

from collections.abc import Iterable
from typing import Any


class LLMManager:
    """Very small LLM facade used by the Assistant.

    Later you can swap this with a real backend (OpenAI, local model, etc).
    """

    def ask(self, text: str, tools: Iterable[Any] | None = None) -> str:
        tool_count = len(list(tools)) if tools is not None else 0
        # Placeholder implementation
        return f"[LLMManager] Echo: {text} (tools available: {tool_count})"
