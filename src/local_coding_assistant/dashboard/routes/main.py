"""Main dashboard routes."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.routes.main")
router = APIRouter()

# Get the templates directory relative to this file
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request) -> HTMLResponse:
    """Dashboard homepage."""
    return templates.TemplateResponse(
        "index.html", {"request": request, "title": "LOCCA Dashboard"}
    )


@router.get("/runs", response_class=HTMLResponse)
async def runs_list(request: Request) -> HTMLResponse:
    """Runs list page."""
    return templates.TemplateResponse(
        "runs.html", {"request": request, "title": "Runs - LOCCA Dashboard"}
    )


@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str) -> HTMLResponse:
    """Run detail page."""
    return templates.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "title": f"Run {run_id} - LOCCA Dashboard",
            "run_id": run_id,
        },
    )


@router.get("/frames/{frame_id}", response_class=HTMLResponse)
async def frame_detail(request: Request, frame_id: str) -> HTMLResponse:
    """Frame detail page."""
    return templates.TemplateResponse(
        "frame_detail.html",
        {
            "request": request,
            "title": f"Frame {frame_id} - LOCCA Dashboard",
            "frame_id": frame_id,
        },
    )


@router.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request) -> HTMLResponse:
    """Analytics dashboard page."""
    return templates.TemplateResponse(
        "analytics.html", {"request": request, "title": "Analytics - LOCCA Dashboard"}
    )


@router.get("/live", response_class=HTMLResponse)
async def live_monitoring(request: Request) -> HTMLResponse:
    """Live monitoring page."""
    return templates.TemplateResponse(
        "live.html", {"request": request, "title": "Live Monitoring - LOCCA Dashboard"}
    )


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "locca-dashboard"}
