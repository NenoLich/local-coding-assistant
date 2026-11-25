"""Event pipeline for fan-out from runtime to subscribers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Protocol

from local_coding_assistant.runtime.events.contracts import RuntimeEvent
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.storage import IExecutionStore

logger = get_logger("runtime.event_pipeline")


class EventSubscriber(Protocol):
    """Protocol for event subscribers."""

    async def on_event(self, event: RuntimeEvent) -> None:
        """Handle a runtime event."""
        ...

    async def close(self) -> None:
        """Cleanup resources."""
        ...


class EventPipeline:
    """Fans out runtime events to registered subscribers.

    Sequence numbers are assigned by the pipeline before fan-out so every
    subscriber observes the same ``event.seq``. Previously subscribers were
    dispatched concurrently and the store adapter assigned ``seq`` in place,
    racing with the dashboard subscriber which read ``seq`` through
    ``model_dump()`` -- producing duplicate zeros and missing sequence numbers
    in the live event timeline.
    """

    def __init__(self, execution_store: IExecutionStore | None = None) -> None:
        self._subscribers: list[EventSubscriber] = []
        self._running = False
        self._execution_store = execution_store
        # Per-session seq counter, initialized from the store's max seq on
        # first use so live events continue the stored numbering.
        self._seq_counters: dict[str, int] = {}

    async def _ensure_seq_counter(self, session_id: str) -> None:
        """Seed the per-session seq counter from persisted events if needed."""
        if session_id not in self._seq_counters and self._execution_store is not None:
            last_seq = await self._execution_store.get_max_seq(session_id)
            self._seq_counters[session_id] = last_seq

    def _next_seq(self, session_id: str) -> int:
        self._seq_counters[session_id] = self._seq_counters.get(session_id, 0) + 1
        return self._seq_counters[session_id]

    def subscribe(self, subscriber: EventSubscriber) -> None:
        """Register a subscriber."""
        self._subscribers.append(subscriber)
        logger.debug("Subscriber registered", subscriber=type(subscriber).__name__)

    def unsubscribe(self, subscriber: EventSubscriber) -> None:
        """Unregister a subscriber."""
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)
            logger.debug(
                "Subscriber unregistered", subscriber=type(subscriber).__name__
            )

    async def emit(self, event: RuntimeEvent) -> None:
        """Emit event to all subscribers concurrently.

        ``event.seq`` is assigned before fan-out so every subscriber observes
        the same sequence number.
        """
        if event.session_id:
            await self._ensure_seq_counter(event.session_id)
            event.seq = self._next_seq(event.session_id)

        if not self._subscribers:
            return

        tasks: list[Awaitable[None]] = [
            self._safe_dispatch(subscriber, event) for subscriber in self._subscribers
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_dispatch(
        self, subscriber: EventSubscriber, event: RuntimeEvent
    ) -> None:
        """Dispatch event to subscriber with error isolation."""
        try:
            await subscriber.on_event(event)
        except Exception as e:
            logger.error(
                "Subscriber error",
                subscriber=type(subscriber).__name__,
                error=str(e),
                exc_info=True,
            )

    async def close(self) -> None:
        """Close all subscribers."""
        tasks = [subscriber.close() for subscriber in self._subscribers]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._subscribers.clear()
        self._running = False

    @property
    def subscribers(self) -> list[EventSubscriber]:
        """Get registered subscribers."""
        return list(self._subscribers)


def create_default_pipeline(
    execution_store: IExecutionStore | None = None,
    enable_otel: bool = False,
) -> EventPipeline:
    """Create event pipeline with default subscribers.

    Args:
        execution_store: Optional execution store for persistence
        enable_otel: Whether to enable OTel/Jaeger subscriber

    Returns:
        Configured EventPipeline
    """
    from local_coding_assistant.runtime.engine.subscribers.store_subscriber import (
        StoreSubscriber,
    )

    pipeline = EventPipeline(execution_store=execution_store)

    # Store subscriber for durable persistence
    if execution_store:
        store_subscriber = StoreSubscriber(execution_store)
        pipeline.subscribe(store_subscriber)

    # OTel subscriber for distributed tracing
    if enable_otel:
        try:
            from local_coding_assistant.runtime.engine.subscribers.otel_subscriber import (
                OTelSubscriber,
            )

            otel_subscriber = OTelSubscriber()
            pipeline.subscribe(otel_subscriber)
        except ImportError:
            logger.warning("OTel subscriber requested but not available")

    return pipeline
