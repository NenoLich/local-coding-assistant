"""Memory provider for context retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.runtime.session import SessionState

logger = get_logger("runtime.memory_provider")


class MemoryProvider:
    """Placeholder memory retrieval component."""

    def fetch(self, *, session: SessionState | None = None) -> list[str]:
        logger.debug("MemoryProvider.fetch called (placeholder)")
        return []


__all__ = ["MemoryProvider"]
