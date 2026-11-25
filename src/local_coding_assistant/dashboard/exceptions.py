"""Dashboard exceptions."""


class DashboardError(Exception):
    """Base dashboard exception."""


class LiveMonitorError(DashboardError):
    """Live monitor service error."""


class StoreUnavailableError(DashboardError):
    """Execution store is not available."""


class NotFoundError(DashboardError):
    """Requested resource not found."""
