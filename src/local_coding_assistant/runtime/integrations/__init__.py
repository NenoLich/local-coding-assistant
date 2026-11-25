"""Cross-cutting integrations."""

from local_coding_assistant.runtime.integrations.dashboard import (
    DashboardIntegration,
    collect_event_for_dashboard,
    disable_dashboard_integration,
    enable_dashboard_integration,
    get_dashboard_integration,
    set_dashboard_url,
)

__all__ = [
    "DashboardIntegration",
    "collect_event_for_dashboard",
    "disable_dashboard_integration",
    "enable_dashboard_integration",
    "get_dashboard_integration",
    "set_dashboard_url",
]
