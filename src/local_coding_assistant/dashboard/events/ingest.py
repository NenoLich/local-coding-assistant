"""HTTP event ingestion for the live monitor.

This endpoint receives RuntimeEvent payloads forwarded by the runtime's
DashboardEventSubscriber (via HTTP POST). It is the wire protocol that
connects the runtime process to the dashboard's live monitoring plane.

Existing wiring is preserved: the runtime POSTs to /api/events/ingest, and
this handler feeds the event into the LiveMonitor and broadcasts it to
WebSocket clients for the live event stream.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from local_coding_assistant.dashboard.routes.websocket import manager
from local_coding_assistant.dashboard.services.live_monitor import LiveMonitor
from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.events.ingest")
router = APIRouter()


def _get_monitor(request: Request) -> LiveMonitor:
    monitor = request.app.state.live_monitor
    if monitor is None:
        raise HTTPException(status_code=503, detail="Live monitor not available")
    return monitor


@router.post("/events/ingest")
async def ingest_event(request: Request, event_data: dict[str, Any]) -> dict[str, str]:
    """Receive a runtime event from the DashboardEventSubscriber and feed it
    into the LiveMonitor for the runs page.

    Expected payload shape (from runtime.integrations.dashboard.DashboardIntegration):
    {
        "type": "<event_type_value>",
        "session_id": "...",
        "frame_id": "...",
        "data": { ...typed event fields... },
        "timestamp": "ISO8601"
    }
    """
    monitor = _get_monitor(request)

    try:
        from local_coding_assistant.runtime.events.contracts import (
            ApprovalDeniedEvent,
            ApprovalGrantedEvent,
            ApprovalRequiredEvent,
            ContinuationDecisionEvent,
            ErrorEvent,
            EventType,
            FrameCompleteEvent,
            FrameStartEvent,
            LLMChunkEvent,
            LLMCompleteEvent,
            LLMStartEvent,
            PartialResponseEvent,
            SessionBranchEvent,
            SessionInterruptEvent,
            SessionResumeEvent,
            SessionStartEvent,
            ToolResultEvent,
            ToolStartEvent,
            TurnCompleteEvent,
            TurnStartEvent,
        )

        event_type_str = event_data.get("type")
        if not event_type_str:
            raise ValueError("Missing 'type' field in event payload")

        event_type = EventType(event_type_str)
        data = event_data.get("data", {}) or {}

        event_kwargs = dict(data)
        event_kwargs["session_id"] = event_data.get("session_id", "")
        event_kwargs["frame_id"] = event_data.get("frame_id")
        if event_data.get("timestamp"):
            from datetime import datetime

            event_kwargs["timestamp"] = datetime.fromisoformat(event_data["timestamp"])

        event_type_to_model = {
            EventType.SESSION_START: SessionStartEvent,
            EventType.SESSION_RESUME: SessionResumeEvent,
            EventType.SESSION_BRANCH: SessionBranchEvent,
            EventType.SESSION_INTERRUPT: SessionInterruptEvent,
            EventType.TURN_START: TurnStartEvent,
            EventType.TURN_COMPLETE: TurnCompleteEvent,
            EventType.FRAME_START: FrameStartEvent,
            EventType.FRAME_COMPLETE: FrameCompleteEvent,
            EventType.LLM_START: LLMStartEvent,
            EventType.LLM_CHUNK: LLMChunkEvent,
            EventType.LLM_COMPLETE: LLMCompleteEvent,
            EventType.TOOL_START: ToolStartEvent,
            EventType.TOOL_RESULT: ToolResultEvent,
            EventType.APPROVAL_REQUIRED: ApprovalRequiredEvent,
            EventType.APPROVAL_GRANTED: ApprovalGrantedEvent,
            EventType.APPROVAL_DENIED: ApprovalDeniedEvent,
            EventType.ERROR: ErrorEvent,
            EventType.PARTIAL_RESPONSE: PartialResponseEvent,
            EventType.CONTINUATION_DECISION: ContinuationDecisionEvent,
        }

        model_cls = event_type_to_model.get(event_type)
        if model_cls is None:
            log.warning("Unknown event type, skipping", event_type=event_type_str)
            return {"status": "skipped", "reason": "unknown event type"}

        event = model_cls(**event_kwargs)
        monitor.ingest_event(event)

        # Broadcast to WebSocket clients for live event stream
        try:
            task = asyncio.create_task(  # fire-and-forget broadcast
                manager.broadcast(
                    {
                        "type": "event",
                        "data": {
                            "seq": event.seq,
                            "type": event.type.value,
                            "timestamp": event.timestamp.isoformat(),
                            "session_id": event.session_id,
                            "turn_id": event.turn_id,
                            "frame_id": event.frame_id,
                            "payload": event.model_dump(
                                exclude={
                                    "type",
                                    "trace_id",
                                    "turn_id",
                                    "frame_id",
                                    "seq",
                                    "timestamp",
                                    "debug_preview",
                                }
                            ),
                        },
                    }
                )
            )
            _ = task
        except Exception as e:
            log.warning("Failed to broadcast event to websocket", error=str(e))

        return {"status": "ok"}

    except ValueError as e:
        log.warning("Failed to ingest event", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        log.warning("Failed to ingest event", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from None
