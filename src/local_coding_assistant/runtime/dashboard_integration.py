"""Integration between RuntimeManager and Dashboard EventCollector."""

from typing import Any

import httpx

from local_coding_assistant.runtime.events import ExecutionEvent
from local_coding_assistant.utils.logging import get_logger

log = get_logger("runtime.dashboard_integration")


class DashboardIntegration:
    """Handles integration between RuntimeManager and Dashboard."""

    def __init__(self, dashboard_url: str = "http://127.0.0.1:8080"):
        self._dashboard_url = dashboard_url
        self._enabled = False
        self._client: httpx.AsyncClient | None = None
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

    def enable(self, dashboard_url: str | None = None):
        """Enable dashboard integration."""
        self._enabled = True
        self._consecutive_failures = 0
        if dashboard_url:
            self._dashboard_url = dashboard_url
        log.info(f"Dashboard integration enabled, URL: {self._dashboard_url}")

    def disable(self):
        """Disable dashboard integration."""
        self._enabled = False
        log.info("Dashboard integration disabled")

    def set_dashboard_url(self, url: str):
        """Set the dashboard URL."""
        self._dashboard_url = url

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx async client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(2.0))
        return self._client

    async def close(self):
        """Close the httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def collect_event(self, event: ExecutionEvent):
        """Send an event to the dashboard via HTTP POST if integration is enabled."""
        if not self._enabled:
            log.debug(
                f"Dashboard integration not enabled, skipping event {event.type.value}"
            )
            return

        # Temporarily disable after repeated failures to reduce noise
        if self._consecutive_failures >= self._max_consecutive_failures:
            return

        try:
            client = self._get_client()

            # Convert event data to JSON-serializable format
            serializable_data = self._make_serializable(event.data)

            # Manually serialize ExecutionEvent to handle enum properly
            event_data = {
                "type": event.type.value,  # Convert enum to string value
                "session_id": event.session_id,
                "frame_id": event.frame_id,
                "data": serializable_data,
                "timestamp": event.timestamp.isoformat(),
            }

            # POST to dashboard's event ingestion endpoint
            url = f"{self._dashboard_url}/api/events/ingest"
            response = await client.post(url, json=event_data)
            if response.status_code != 200:
                log.debug(
                    f"Dashboard event POST failed: {response.status_code} - {response.text}"
                )
            else:
                self._consecutive_failures = 0  # Reset on success

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            # Expected failures when dashboard is not running - log briefly without traceback
            self._consecutive_failures += 1
            if self._consecutive_failures == 1:
                log.debug(
                    f"Dashboard unavailable ({type(e).__name__}), will suppress further errors"
                )
        except Exception as e:
            # Unexpected errors - log with full traceback
            self._consecutive_failures += 1
            log.debug(f"Failed to send event to dashboard: {e}")
            import traceback

            log.debug(traceback.format_exc())

    def _make_serializable(self, obj: Any) -> Any:
        """Convert complex objects to JSON-serializable format."""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, "__dict__"):
            # Convert dataclass or object with __dict__ to dict
            return self._make_serializable(obj.__dict__)
        elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
            # Handle other iterables
            return [self._make_serializable(item) for item in obj]
        else:
            # Fallback: convert to string
            return str(obj)


# Global instance
_dashboard_integration = DashboardIntegration()


def get_dashboard_integration() -> DashboardIntegration:
    """Get the global dashboard integration instance."""
    return _dashboard_integration


async def collect_event_for_dashboard(event: ExecutionEvent):
    """Convenience function to collect an event for the dashboard."""
    await _dashboard_integration.collect_event(event)


def enable_dashboard_integration(dashboard_url: str | None = None):
    """Enable dashboard integration globally."""
    _dashboard_integration.enable(dashboard_url=dashboard_url)


def disable_dashboard_integration():
    """Disable dashboard integration globally."""
    _dashboard_integration.disable()


def set_dashboard_url(url: str):
    """Set the dashboard URL for event posting."""
    _dashboard_integration.set_dashboard_url(url)
