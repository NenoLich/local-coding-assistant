"""Dashboard event subscriber - forwards runtime events to the dashboard via HTTP."""

from __future__ import annotations

from local_coding_assistant.runtime.events.contracts import RuntimeEvent
from local_coding_assistant.runtime.events.pipeline import EventSubscriber
from local_coding_assistant.runtime.integrations.dashboard import (
    DashboardIntegration,
    get_dashboard_integration,
)
from local_coding_assistant.utils.logging import get_logger

log = get_logger("runtime.subscribers.dashboard")


class DashboardEventSubscriber(EventSubscriber):
    """Event subscriber that forwards runtime events to the dashboard via HTTP.

    Registered in the EventPipeline when dashboard integration is enabled.
    Forwards each RuntimeEvent to DashboardIntegration which POSTs it to
    the dashboard's /api/events/ingest endpoint.
    """

    def __init__(self, dashboard_integration: DashboardIntegration | None = None):
        self._integration = dashboard_integration or get_dashboard_integration()

    async def on_event(self, event: RuntimeEvent) -> None:
        """Forward a runtime event to the dashboard."""
        await self._integration.collect_event(event)

    async def close(self) -> None:
        """Cleanup resources."""
        await self._integration.close()
