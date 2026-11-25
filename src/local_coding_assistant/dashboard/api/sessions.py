"""Historical sessions API routes (store-backed, read-only)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from local_coding_assistant.dashboard.api.deps import get_session_service
from local_coding_assistant.dashboard.models.historical import SessionDetail

router = APIRouter()


@router.get("/sessions", response_model=Any)
async def list_sessions(
    request: Request,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, description="Filter by status"),
    date_from: str | None = Query(None, description="Filter from date (ISO format)"),
    date_to: str | None = Query(None, description="Filter to date (ISO format)"),
    model: str | None = Query(None, description="Filter by model name"),
) -> Any:
    """List historical sessions with pagination and filtering."""
    service = get_session_service(request)
    return await service.list_sessions(
        limit=limit,
        offset=offset,
        status=status,
        date_from=date_from,
        date_to=date_to,
        model=model,
    )


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session_detail(
    request: Request,
    session_id: str,
) -> SessionDetail:
    """Get full historical session detail with turns."""
    service = get_session_service(request)
    return await service.get_session_detail(session_id)
