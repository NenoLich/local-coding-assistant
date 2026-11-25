"""Historical events API routes (store-backed, read-only)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from local_coding_assistant.dashboard.api.deps import get_event_service
from local_coding_assistant.dashboard.models.historical import EventTimelineEntry

router = APIRouter()


@router.get("/events/turn/{turn_id}", response_model=list[EventTimelineEntry])
async def get_events_by_turn(
    request: Request,
    turn_id: str,
    event_type: list[str] | None = Query(None, description="Filter by event type"),  # noqa: B008
) -> list[EventTimelineEntry]:
    """Get all events for a turn."""
    service = get_event_service(request)
    return await service.get_events_by_turn(turn_id, event_type=event_type)


@router.get("/events/frame/{frame_id}", response_model=list[EventTimelineEntry])
async def get_events_by_frame(
    request: Request,
    frame_id: str,
    event_type: list[str] | None = Query(None, description="Filter by event type"),  # noqa: B008
) -> list[EventTimelineEntry]:
    """Get all events for a frame."""
    service = get_event_service(request)
    return await service.get_events_by_frame(frame_id, event_type=event_type)


@router.get("/events/payload/{turn_id}", response_model=Any)
async def get_event_payload(
    request: Request,
    turn_id: str,
    event_id: int = Query(..., description="Unique event ID"),
) -> Any:
    """Get the full payload for a specific event (expandable view)."""
    service = get_event_service(request)
    payload = await service.get_event_payload(turn_id, event_id)
    if payload is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Event not found")
    return payload
