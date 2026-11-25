"""Historical frames API routes (store-backed, read-only)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from local_coding_assistant.dashboard.api.deps import get_frame_service
from local_coding_assistant.dashboard.models.historical import FrameDetail

router = APIRouter()


@router.get("/frames/{frame_id}", response_model=FrameDetail)
async def get_frame_detail(
    request: Request,
    frame_id: str,
) -> FrameDetail:
    """Get full historical frame detail with event timeline."""
    service = get_frame_service(request)
    return await service.get_frame_detail(frame_id)
