"""Debug API routes (store-backed, read-only)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from local_coding_assistant.dashboard.api.deps import get_debug_service

router = APIRouter()


@router.get("/debug/snapshot/{snapshot_id}", response_model=Any)
async def debug_snapshot(
    request: Request,
    snapshot_id: str,
) -> Any:
    """Resolve a context snapshot for debug display."""
    service = get_debug_service(request)
    return await service.resolve_snapshot(snapshot_id)


@router.get("/debug/snapshot/{snapshot_id}/missing-refs", response_model=Any)
async def debug_missing_refs(
    request: Request,
    snapshot_id: str,
) -> Any:
    """Check which context refs of a snapshot still exist in the store."""
    service = get_debug_service(request)
    missing = await service.check_missing_refs(snapshot_id)
    return {
        "snapshot_id": snapshot_id,
        "missing_tools": missing.tool_ids,
        "missing_templates": missing.template_ids,
        "missing_repo_hashes": missing.repo_hashes,
        "missing_memory_ids": missing.memory_ids,
        "is_empty": missing.is_empty,
    }


@router.get("/health", response_model=Any)
async def health(request: Request) -> Any:
    """Health check."""
    store = request.app.state.execution_store
    monitor = request.app.state.live_monitor
    return {
        "status": "healthy",
        "store_available": store is not None,
        "live_monitor_available": monitor is not None,
    }
