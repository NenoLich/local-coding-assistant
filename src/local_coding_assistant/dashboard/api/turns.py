"""Historical turns API routes (store-backed, read-only)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from local_coding_assistant.dashboard.api.deps import get_turn_service
from local_coding_assistant.dashboard.models.historical import TurnDetail

router = APIRouter()


@router.get("/turns/{turn_id}", response_model=TurnDetail)
async def get_turn_detail(
    request: Request,
    turn_id: str,
) -> TurnDetail:
    """Get full historical turn detail with frames and messages."""
    service = get_turn_service(request)
    return await service.get_turn_detail(turn_id)
