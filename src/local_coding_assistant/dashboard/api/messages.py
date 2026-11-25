"""Historical messages API routes (store-backed, read-only)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from local_coding_assistant.dashboard.api.deps import get_message_service
from local_coding_assistant.dashboard.models.messages import MessageDetail

router = APIRouter()


@router.get("/messages/session/{session_id}", response_model=Any)
async def list_messages_by_session(
    request: Request,
    session_id: str,
    turn_from: int | None = Query(
        None, description="Filter messages from turn index (inclusive)"
    ),
    turn_to: int | None = Query(
        None, description="Filter messages to turn index (inclusive)"
    ),
    created_from: str | None = Query(
        None, description="Filter messages created from (ISO format)"
    ),
    created_to: str | None = Query(
        None, description="Filter messages created to (ISO format)"
    ),
) -> Any:
    """List all transcript messages for a session with optional filters."""
    service = get_message_service(request)
    return await service.list_messages_by_session(
        session_id=session_id,
        turn_from=turn_from,
        turn_to=turn_to,
        created_from=created_from,
        created_to=created_to,
    )


@router.get("/messages/{session_id}/{seq}", response_model=MessageDetail)
async def get_message_detail(
    request: Request,
    session_id: str,
    seq: int,
) -> MessageDetail:
    """Get full detail for a single message including raw payload."""
    service = get_message_service(request)
    return await service.get_message_detail(session_id, seq)
