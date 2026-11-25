"""Events package for the dashboard."""

from local_coding_assistant.dashboard.events.ingest import router as ingest_router

__all__ = ["ingest_router"]
