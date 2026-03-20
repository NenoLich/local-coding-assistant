"""FastAPI application for the dashboard."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    log.info("Starting dashboard application")
    try:
        yield
    finally:
        log.info("Shutting down dashboard application")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LOCCA Dashboard",
        description="ExecutionFrame observability and analysis dashboard",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For development - restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    from .routes import api, main, websocket

    app.include_router(main.router, tags=["main"])
    app.include_router(api.router, prefix="/api", tags=["api"])
    app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

    # Mount static files if directory exists
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    else:
        log.warning(f"Static files directory not found: {static_dir}")

    return app
