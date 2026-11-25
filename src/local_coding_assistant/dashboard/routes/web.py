"""HTML page routes for the dashboard."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.routes.web")
router = APIRouter()

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request) -> HTMLResponse:
    """Dashboard homepage — redirects to runs page."""
    return templates.TemplateResponse(
        "runs.html", {"request": request, "title": "LOCCA Dashboard"}
    )


@router.get("/runs", response_class=HTMLResponse)
async def runs_page(request: Request) -> HTMLResponse:
    """Runs page — in-memory live monitoring."""
    return templates.TemplateResponse(
        "runs.html", {"request": request, "title": "Runs - LOCCA Dashboard"}
    )


@router.get("/sessions", response_class=HTMLResponse)
async def sessions_page(request: Request) -> HTMLResponse:
    """Historical sessions list page."""
    return templates.TemplateResponse(
        "sessions.html", {"request": request, "title": "Sessions - LOCCA Dashboard"}
    )


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
async def session_detail_page(request: Request, session_id: str) -> HTMLResponse:
    """Session detail page."""
    return templates.TemplateResponse(
        "session_detail.html",
        {
            "request": request,
            "title": f"Session {session_id[:8]} - LOCCA Dashboard",
            "session_id": session_id,
        },
    )


@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail_page(request: Request, run_id: str) -> HTMLResponse:
    """Run detail page."""
    return templates.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "title": f"Run {run_id[:8]} - LOCCA Dashboard",
            "run_id": run_id,
        },
    )


@router.get("/turns/{turn_id}", response_class=HTMLResponse)
async def turn_detail_page(request: Request, turn_id: str) -> HTMLResponse:
    """Turn detail page."""
    return templates.TemplateResponse(
        "turn_detail.html",
        {
            "request": request,
            "title": f"Turn {turn_id[:8]} - LOCCA Dashboard",
            "turn_id": turn_id,
        },
    )


@router.get("/frames/{frame_id}", response_class=HTMLResponse)
async def frame_detail_page(request: Request, frame_id: str) -> HTMLResponse:
    """Frame detail page."""
    return templates.TemplateResponse(
        "frame_detail.html",
        {
            "request": request,
            "title": f"Frame {frame_id[:8]} - LOCCA Dashboard",
            "frame_id": frame_id,
        },
    )


@router.get("/debug", response_class=HTMLResponse)
async def debug_page(request: Request) -> HTMLResponse:
    """Debug tools page."""
    return templates.TemplateResponse(
        "debug.html", {"request": request, "title": "Debug - LOCCA Dashboard"}
    )


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "locca-dashboard"}
