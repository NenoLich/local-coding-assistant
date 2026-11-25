"""Routes package for the dashboard."""

from local_coding_assistant.dashboard.routes.web import router as web_router
from local_coding_assistant.dashboard.routes.websocket import router as websocket_router

__all__ = ["web_router", "websocket_router"]
