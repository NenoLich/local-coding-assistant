"""Store subscriber - writes runtime events to execution store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_coding_assistant.runtime.events.contracts import RuntimeEvent

if TYPE_CHECKING:
    from local_coding_assistant.storage import IExecutionStore
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.subscribers.store")


class StoreSubscriber:
    """Subscriber that forwards events to the execution store."""

    def __init__(self, store: IExecutionStore) -> None:
        self._store = store

    async def on_event(self, event: RuntimeEvent) -> None:
        """Forward event to execution store."""
        await self._store.process_event(event)

    async def close(self) -> None:
        """Flush remaining events."""
        await self._store.flush()
