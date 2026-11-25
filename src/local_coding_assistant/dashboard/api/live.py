"""Live monitoring API routes.

These endpoints serve the runs page (in-memory live monitoring). Data comes
from the LiveMonitor, which is fed by runtime events via HTTP ingestion.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from local_coding_assistant.dashboard.api.deps import _get_monitor
from local_coding_assistant.dashboard.models.historical import EventTimelineEntry
from local_coding_assistant.dashboard.models.live import LiveRunSummary, LiveStats
from local_coding_assistant.dashboard.services.event_preview import preview_for_event

router = APIRouter()


@router.get("/live/runs", response_model=list[LiveRunSummary])
async def list_live_runs(request: Request) -> list[LiveRunSummary]:
    """List all live runs (turns) currently being monitored."""
    monitor = _get_monitor(request)
    return monitor.get_run_summaries()


@router.get("/live/sessions/{session_id}", response_model=Any)
async def get_live_session(request: Request, session_id: str) -> Any:
    """Get a live session by ID."""
    monitor = _get_monitor(request)
    session = monitor.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Live session not found")
    return session


@router.get("/live/stats", response_model=LiveStats)
async def get_live_stats(request: Request) -> LiveStats:
    """Get live aggregate statistics."""
    monitor = _get_monitor(request)
    return monitor.get_stats()


@router.get("/live/sessions", response_model=Any)
async def list_live_sessions(request: Request) -> Any:
    """List all active (running/paused) live sessions."""
    monitor = _get_monitor(request)
    return monitor.get_active_sessions()


@router.get("/live/runs/{run_id}", response_model=Any)
async def get_live_run(request: Request, run_id: str) -> Any:
    """Get a specific live run (turn) with session info."""
    monitor = _get_monitor(request)
    for session in monitor.sessions.values():
        if run_id in session.turns:
            return {
                "session": session,
                "turn": session.turns[run_id],
            }
    raise HTTPException(status_code=404, detail="Live run not found")


@router.get(
    "/live/events/{run_id}",
    response_model=list[EventTimelineEntry],
)
async def get_live_events(
    request: Request,
    run_id: str,
    event_type: list[str] | None = Query(None, description="Filter by event type"),  # noqa: B008
) -> list[EventTimelineEntry]:
    """Get all events for a live run (turn), shaped like the historical timeline.

    Events are read from the in-memory LiveMonitor (runtime events ingested via
    the DashboardEventSubscriber), not from the execution store. The returned
    entries have the same ``EventTimelineEntry`` shape as the historical
    turn/frame detail timelines, so the run details event timeline renders
    identically to them, preview fields included.
    """
    monitor = _get_monitor(request)
    for session in monitor.sessions.values():
        if run_id in session.turns:
            entries: list[EventTimelineEntry] = []
            for ev in monitor.get_turn_events(run_id):
                etype = ev.type.value
                if event_type and etype not in event_type:
                    continue
                entries.append(
                    EventTimelineEntry(
                        event_id=ev.seq,
                        seq=ev.seq,
                        type=etype,
                        timestamp=ev.timestamp,
                        frame_id=ev.frame_id,
                        preview=preview_for_event(etype, ev.payload),
                        payload=ev.payload,
                        has_payload=bool(ev.payload),
                    )
                )
            entries.sort(key=lambda e: e.seq)
            return entries
    raise HTTPException(status_code=404, detail="Live run not found")
