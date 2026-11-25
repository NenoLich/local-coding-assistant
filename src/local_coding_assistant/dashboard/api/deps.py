"""Dependency injection for dashboard API routes."""

from typing import cast

from fastapi import Request

from local_coding_assistant.dashboard.services.debug_service import DebugService
from local_coding_assistant.dashboard.services.event_service import EventService
from local_coding_assistant.dashboard.services.frame_service import FrameService
from local_coding_assistant.dashboard.services.live_monitor import LiveMonitor
from local_coding_assistant.dashboard.services.message_service import MessageService
from local_coding_assistant.dashboard.services.session_service import SessionService
from local_coding_assistant.dashboard.services.turn_service import TurnService
from local_coding_assistant.storage.execution_store_reader import ExecutionStoreReader


def _get_store(request: Request) -> ExecutionStoreReader:
    store = request.app.state.execution_store
    if store is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Execution store not available")
    return cast(ExecutionStoreReader, store)


def _get_monitor(request: Request) -> LiveMonitor:
    monitor = request.app.state.live_monitor
    if monitor is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Live monitor not available")
    return monitor


def get_session_service(request: Request) -> SessionService:
    return SessionService(_get_store(request))


def get_turn_service(request: Request) -> TurnService:
    return TurnService(_get_store(request))


def get_frame_service(request: Request) -> FrameService:
    return FrameService(_get_store(request))


def get_event_service(request: Request) -> EventService:
    return EventService(_get_store(request))


def get_debug_service(request: Request) -> DebugService:
    return DebugService(_get_store(request))


def get_message_service(request: Request) -> MessageService:
    return MessageService(_get_store(request))
