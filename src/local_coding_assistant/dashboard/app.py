"""FastAPI application for the redesigned LOCCA dashboard.

The dashboard has two data planes:

1. **Live plane (runs page)**: In-memory state reconstructed from runtime
   events received via HTTP ingestion (/api/events/ingest). Backed by
   LiveMonitor. No store reads during live monitoring.

2. **Historical plane (everything else)**: Read-only queries against the
   execution store via IExecutionStoreReader. No writes.

The dashboard app starts standalone (no LLM, no tool gateway, no runtime
manager). It opens the execution store in read-only mode for historical
queries and exposes an ingest endpoint for live events.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from local_coding_assistant.dashboard.api import (
    debug_router,
    events_router,
    frames_router,
    ingest_router,
    live_router,
    messages_router,
    sessions_router,
    turns_router,
)
from local_coding_assistant.dashboard.config import DashboardConfig
from local_coding_assistant.dashboard.exceptions import NotFoundError
from local_coding_assistant.dashboard.routes import web_router, websocket_router
from local_coding_assistant.dashboard.services.live_monitor import (
    LiveMonitor as LiveMonitorService,
)
from local_coding_assistant.storage.execution_store_reader import ExecutionStoreReader
from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.app")

# Global state
_dashboard_store: ExecutionStoreReader | None = None
_live_monitor: LiveMonitorService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan: opens read-only store, initializes live monitor."""
    global _dashboard_store, _live_monitor

    log.info("Starting dashboard application")

    store_db_path = app.state.store_db_path
    if store_db_path and Path(store_db_path).exists():
        db_path = Path(store_db_path).resolve()
        _dashboard_store = ExecutionStoreReader(db_path)
        await _dashboard_store.init()
        app.state.execution_store = _dashboard_store
        log.info("Dashboard initialized with execution store", db_path=str(db_path))
    else:
        log.warning(
            "Execution store database not found at %s, historical pages unavailable",
            store_db_path,
        )
        app.state.execution_store = None

    # Initialize live monitor (in-memory, always available)
    config = DashboardConfig()
    _live_monitor = LiveMonitorService(
        max_events=config.live_event_buffer,
        max_events_per_frame=200,
        max_runs=config.live_runs_buffer,
    )
    app.state.live_monitor = _live_monitor
    log.info("Live monitor initialized")

    yield

    log.info("Shutting down dashboard application")
    if _dashboard_store is not None:
        await _dashboard_store.close()


def create_app(store_db_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI dashboard application."""
    app = FastAPI(
        title="LOCCA Dashboard",
        description="Execution observability dashboard (live runs + historical store)",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.state.store_db_path = store_db_path

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(web_router)
    app.include_router(websocket_router, prefix="/ws")

    # API routes under /api
    app.include_router(live_router, prefix="/api")
    app.include_router(sessions_router, prefix="/api")
    app.include_router(turns_router, prefix="/api")
    app.include_router(frames_router, prefix="/api")
    app.include_router(events_router, prefix="/api")
    app.include_router(debug_router, prefix="/api")
    app.include_router(messages_router, prefix="/api")
    app.include_router(ingest_router, prefix="/api")

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    else:
        log.warning("Static files directory not found: %s", static_dir)

    # Exception handlers
    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    return app
