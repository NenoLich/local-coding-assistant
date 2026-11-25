"""Dashboard configuration."""

from pydantic import BaseModel, Field


class DashboardConfig(BaseModel):
    """Configuration for the dashboard server."""

    store_db_path: str = Field(
        default="data/execution_store.db",
        description="Path to the execution store database (read-only)",
    )
    host: str = Field(default="127.0.0.1", description="Host to bind to")
    port: int = Field(default=8080, description="Port to listen on")
    live_event_buffer: int = Field(
        default=5000,
        description="Maximum number of live events to keep in memory",
    )
    live_runs_buffer: int = Field(
        default=1000,
        description="Maximum number of live runs to keep in memory",
    )
