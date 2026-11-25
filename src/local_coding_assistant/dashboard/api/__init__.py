"""API routes package for the dashboard."""

from local_coding_assistant.dashboard.api.debug import router as debug_router
from local_coding_assistant.dashboard.api.events import router as events_router
from local_coding_assistant.dashboard.api.frames import router as frames_router
from local_coding_assistant.dashboard.api.live import router as live_router
from local_coding_assistant.dashboard.api.messages import router as messages_router
from local_coding_assistant.dashboard.api.sessions import router as sessions_router
from local_coding_assistant.dashboard.api.turns import router as turns_router
from local_coding_assistant.dashboard.events.ingest import router as ingest_router

__all__ = [
    "debug_router",
    "events_router",
    "frames_router",
    "ingest_router",
    "live_router",
    "messages_router",
    "sessions_router",
    "turns_router",
]
