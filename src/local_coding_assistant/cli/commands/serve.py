"""Start the assistant server for out-of-process approvals and live monitoring."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import typer
import uvicorn
from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Query,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.runtime.events.contracts import (
    ApprovalResponseForm,
    RuntimeEvent,
)
from local_coding_assistant.runtime.manager import RuntimeManager
from local_coding_assistant.runtime.options import RunOptions
from local_coding_assistant.storage import IExecutionStore
from local_coding_assistant.utils.logging import get_logger

app = typer.Typer(name="serve", help="Start the assistant server")
log = get_logger("cli.serve")

# Global server state
_runtime_manager: RuntimeManager | None = None
_store: IExecutionStore | None = None
_server_shutdown = asyncio.Event()


@asynccontextmanager
async def lifespan(server_app: FastAPI):
    """Application lifespan manager for the server."""
    global _runtime_manager, _store

    log.info("Starting server lifespan")
    try:
        yield
    finally:
        log.info("Shutting down server lifespan")
        _server_shutdown.set()


def create_server_app() -> FastAPI:
    """Create the FastAPI server application."""
    server_app = FastAPI(
        title="LOCCA Runtime Server",
        description="Runtime server for out-of-process approvals and live event streaming",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    server_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    server_app.include_router(
        sessions_router, prefix="/api/sessions", tags=["sessions"]
    )
    server_app.include_router(
        approvals_router, prefix="/api/approvals", tags=["approvals"]
    )
    server_app.include_router(events_router, prefix="/api/events", tags=["events"])
    server_app.include_router(health_router, tags=["health"])

    return server_app


# ========== Event Subscriber for Server ==========


class ServerEventSubscriber:
    """Subscriber that forwards runtime events to WebSocket connections for live dashboard updates."""

    def __init__(self):
        self._websockets: dict[str, list[WebSocket]] = {}

    def register_websocket(self, session_id: str, websocket: WebSocket) -> None:
        """Register a WebSocket connection for a session."""
        if session_id not in self._websockets:
            self._websockets[session_id] = []
        self._websockets[session_id].append(websocket)

    def unregister_websocket(self, session_id: str, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection."""
        if session_id in self._websockets:
            try:
                self._websockets[session_id].remove(websocket)
            except ValueError:
                pass
            if not self._websockets[session_id]:
                del self._websockets[session_id]

    async def on_event(self, event: RuntimeEvent) -> None:
        """Process event and broadcast to WebSocket connections."""
        if not event.session_id:
            return

        if event.session_id not in self._websockets:
            return

        message = {
            "type": "event",
            "data": {
                "event_type": event.type.value,
                "session_id": event.session_id,
                "turn_id": event.turn_id,
                "frame_id": event.frame_id,
                "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                "data": event.model_dump(
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

        disconnected = []
        for ws in self._websockets[event.session_id]:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.unregister_websocket(event.session_id, ws)

    async def close(self) -> None:
        """Close all registered WebSocket connections."""
        for ws_list in list(self._websockets.values()):
            for ws in ws_list:
                try:
                    await ws.close()
                except Exception as e:
                    log.warning("Failed to close websocket", error=str(e))
        self._websockets.clear()


server_event_subscriber = ServerEventSubscriber()


# ========== API Routes ==========

sessions_router = APIRouter()
approvals_router = APIRouter()
events_router = APIRouter()
health_router = APIRouter()


def _get_store() -> IExecutionStore:
    """Get execution store from global state."""
    if not _store:
        raise HTTPException(status_code=503, detail="Execution store not available")
    return _store


def _get_runtime_manager() -> RuntimeManager:
    """Get runtime manager from global state."""
    if not _runtime_manager:
        raise HTTPException(status_code=503, detail="Runtime manager not available")
    return _runtime_manager


@health_router.get("/health")
async def health_check():
    """Health check endpoint."""
    data = {"status": "healthy", "service": "locca-runtime-server"}

    # Convert to string and append the newline character
    json_str = json.dumps(data, indent=2, default=str) + "\n"

    return Response(content=json_str, media_type="application/json")


@health_router.get("/health/ready")
async def readiness_check():
    """Readiness check - verifies store and runtime are available."""
    data = {
        "ready": _runtime_manager is not None and _store is not None,
        "runtime_manager": _runtime_manager is not None,
        "store": _store is not None,
    }
    # Convert to string and append the newline character
    json_str = json.dumps(data, indent=2, default=str) + "\n"

    return Response(content=json_str, media_type="application/json")


def _ms_to_iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat()


@sessions_router.get("")
async def list_sessions(
    limit: int = Query(50, ge=1, le=1000),
    status: str | None = Query(None, description="Filter by status"),
):
    """List sessions from durable store, sorted by last updated."""
    store = _get_store()
    sessions = await store.list_sessions(limit=limit, status=status)

    # Sort by updated_at desc (fallback to started_at)
    sessions.sort(
        key=lambda s: s.updated_at_ms
        if s.updated_at_ms is not None
        else s.started_at_ms,
        reverse=True,
    )

    data = [
        {
            "session_id": s.id,
            "user_input": s.user_input,
            "status": s.status,
            "started_at": _ms_to_iso(s.started_at_ms),
            "updated_at": _ms_to_iso(s.updated_at_ms),
            "agent_mode": s.agent_mode,
            "model_name": s.model_name,
            "input_tokens": s.total_input_tokens or 0,
            "output_tokens": s.total_output_tokens or 0,
            "total_turns": s.total_turns or 0,
            "total_frames": s.total_frames or 0,
        }
        for s in sessions
    ]
    json_str = json.dumps(data, indent=2, default=str) + "\n"

    return Response(content=json_str, media_type="application/json")


class SessionResponse(BaseModel):
    """Flat response for session create/resume/approval endpoints."""

    session_id: str
    turn_id: str | None = None
    trace_id: str | None = None
    status: str
    started_at: str | None = None
    updated_at: str | None = None
    final_answer: str | None = None


class CreateSessionRequest(BaseModel):
    """Request model for creating a new session."""

    user_input: str = Field(..., description="User input for the session")
    trace_id: str | None = Field(None, description="Optional trace ID")
    model: str | None = Field(None, description="Model name override")
    agent_mode: str = Field("frame_agent", description="Agent mode")
    tool_call_mode: str | None = Field(None, description="Tool call mode")
    auto_approve: bool | None = Field(None, description="Auto-approve all tool calls")
    metadata_json: str | None = Field(
        None, description="Optional metadata as JSON string"
    )
    stream: bool = Field(False, description="Stream runtime events as SSE")


class ResumeSessionRequest(BaseModel):
    """Request model for resuming a session."""

    user_input: str | None = Field(None, description="User input for new turn")
    turn_id: str | None = Field(None, description="Turn ID to resume from")
    turn_index: int | None = Field(None, description="Turn index to resume from")
    frame_id: str | None = Field(None, description="Frame ID to resume from")
    model: str | None = Field(None, description="Model name override")
    agent_mode: str | None = Field(None, description="Agent mode")
    tool_call_mode: str | None = Field(None, description="Tool call mode")
    auto_approve: bool | None = Field(None, description="Auto-approve all tool calls")
    stream: bool = Field(False, description="Stream runtime events as SSE")


class RespondApprovalRequest(BaseModel):
    """Request model for respond approval."""

    session_id: str = Field(..., description="Session ID to resume")
    turn_id: str = Field(..., description="Turn ID to resume from")
    responses: list[ApprovalResponseForm] = Field(
        default_factory=list, description="List of ApprovalResponseForm"
    )
    stream: bool = Field(False, description="Stream runtime events as SSE")


class InterruptSessionRequest(BaseModel):
    """Request model for interrupting a running session turn."""

    turn_id: str | None = Field(
        None, description="Specific turn to interrupt (defaults to last running turn)"
    )
    stream: bool = Field(False, description="Stream runtime events as SSE")


def _session_response(session, turn) -> SessionResponse:
    """Build a flat session response from a session and its last turn."""
    return SessionResponse(
        session_id=session.id,
        turn_id=turn.turn_id if turn else None,
        trace_id=turn.trace_id if turn else None,
        status=str(turn.status) if turn else "unknown",
        started_at=turn.started_at.isoformat() if turn and turn.started_at else None,
        updated_at=turn.finished_at.isoformat() if turn and turn.finished_at else None,
        final_answer=turn.get_final_answer() if turn else None,
    )


class _RequestEventSubscriber:
    """Transient subscriber that queues runtime events for one HTTP request."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[RuntimeEvent | None] = asyncio.Queue()

    async def on_event(self, event: RuntimeEvent) -> None:
        """Queue an emitted runtime event."""
        await self._queue.put(event)

    async def close(self) -> None:
        """Signal end-of-stream."""
        await self._queue.put(None)

    def event_to_sse(self, event: RuntimeEvent) -> str:
        """Format a runtime event as a Server-Sent Event message."""
        payload = {
            "type": "event",
            "data": {
                "event_type": event.type.value,
                "session_id": event.session_id,
                "turn_id": event.turn_id,
                "frame_id": event.frame_id,
                "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                "data": event.model_dump(
                    exclude={
                        "type",
                        "session_id",
                        "trace_id",
                        "turn_id",
                        "frame_id",
                        "seq",
                        "timestamp",
                        "debug_preview",
                        "adjusted_options",
                        "template_path",
                        "max_tokens_adjustment",
                        "temperature_adjustment",
                        "compact_instruction",
                        "config_snapshot_json",
                        "context_snapshot_json",
                        "context_profile_json",
                        "context_hash",
                        "prompt_hash",
                        "extra_content",
                    }
                ),
            },
        }
        return f"data: {json.dumps(payload, default=str)}\n\n"


async def _sse_session_stream(
    run_coro: Coroutine[Any, Any, Any],
    subscriber: _RequestEventSubscriber,
    pipeline: Any | None,
    turn_getter: Callable[[Any], Any],
) -> AsyncGenerator[str, None]:
    """Stream runtime events as SSE, then yield the final session result."""
    task = asyncio.create_task(run_coro)
    try:
        while not task.done():
            try:
                event = await asyncio.wait_for(subscriber._queue.get(), timeout=0.2)
                if event is None:
                    break
                yield subscriber.event_to_sse(event)
            except TimeoutError:
                continue
        while not subscriber._queue.empty():
            event = subscriber._queue.get_nowait()
            if event is None:
                break
            yield subscriber.event_to_sse(event)
        session = await task
        turn = turn_getter(session)
        result = {
            "type": "session_result",
            "data": _session_response(session, turn).model_dump(),
        }
        yield f"data: {json.dumps(result, indent=2, default=str)}\n\n"
    finally:
        if pipeline:
            pipeline.unsubscribe(subscriber)
        await subscriber.close()


async def _execute_with_stream(
    run_coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    stream: bool,
    turn_getter: Callable[[Any], Any],
    pipeline_getter: Callable[[], Any | None],
) -> SessionResponse | StreamingResponse:
    """Execute a session operation, optionally streaming runtime events via SSE."""
    if not stream:
        session = await run_coro_factory()
        return _session_response(session, turn_getter(session))

    subscriber = _RequestEventSubscriber()
    pipeline = pipeline_getter()
    if pipeline:
        pipeline.subscribe(subscriber)

    return StreamingResponse(
        _sse_session_stream(run_coro_factory(), subscriber, pipeline, turn_getter),
        media_type="text/event-stream",
    )


@sessions_router.post("", response_model=None)
async def create_session(
    request: CreateSessionRequest,
) -> SessionResponse | StreamingResponse:
    """Create a new session and start the first turn."""
    runtime = _get_runtime_manager()

    options = RunOptions(
        model=request.model,
        agent_mode=request.agent_mode,
        tool_call_mode=request.tool_call_mode,
        auto_approve=request.auto_approve,
    )
    runtime.apply_run_options(options)

    return await _execute_with_stream(
        run_coro_factory=lambda: runtime.start_session(
            user_input=request.user_input,
            trace_id=request.trace_id,
            session_metadata={"metadata_json": request.metadata_json}
            if request.metadata_json
            else None,
        ),
        stream=request.stream,
        turn_getter=lambda s: s.get_last_turn(),
        pipeline_getter=lambda: runtime.event_pipeline,
    )


@sessions_router.get("/{session_id}")
async def get_session(session_id: str):
    """Get session details with turns."""
    store = _get_store()
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    turns = await store.list_turns_by_session(session_id)
    turns_data = []
    for turn in turns:
        frame_ids = await store.get_frame_ids_by_turn(turn.id)
        frames_data = []
        for frame_id in frame_ids:
            frame = await store.get_frame(frame_id)
            if frame:
                frames_data.append(
                    {
                        "frame_id": frame.id,
                        "iteration": frame.iteration,
                        "status": frame.status,
                        "started_at": _ms_to_iso(frame.started_at_ms),
                        "finished_at": _ms_to_iso(frame.finished_at_ms),
                        "duration_ms": frame.duration_ms,
                        "input_tokens": frame.input_tokens or 0,
                        "output_tokens": frame.output_tokens or 0,
                        "model_name": frame.model_name,
                        "finish_reason": frame.finish_reason,
                        "error_message": frame.error_message,
                    }
                )

        turns_data.append(
            {
                "turn_id": turn.id,
                "turn_index": turn.turn_index,
                "trace_id": turn.trace_id,
                "user_input": turn.user_input,
                "status": turn.status,
                "started_at": _ms_to_iso(turn.started_at_ms),
                "finished_at": _ms_to_iso(turn.finished_at_ms),
                "input_tokens": turn.total_input_tokens or 0,
                "output_tokens": turn.total_output_tokens or 0,
                "total_frames": turn.total_frames,
                "frames": frames_data,
            }
        )

    data = {
        "session": {
            "id": session.id,
            "user_input": session.user_input,
            "status": session.status,
            "started_at": _ms_to_iso(session.started_at_ms),
            "updated_at": _ms_to_iso(session.updated_at_ms),
            "agent_mode": session.agent_mode,
            "model_name": session.model_name,
            "input_tokens": session.total_input_tokens or 0,
            "output_tokens": session.total_output_tokens or 0,
            "total_turns": session.total_turns,
            "total_frames": session.total_frames,
            "metadata_json": session.metadata_json,
        },
        "turns": turns_data,
    }

    json_str = json.dumps(data, indent=2, default=str) + "\n"

    return Response(content=json_str, media_type="application/json")


@sessions_router.post("/{session_id}/resume", response_model=None)
async def resume_session(
    session_id: str,
    request: ResumeSessionRequest,
) -> SessionResponse | StreamingResponse:
    """Resume a session (continue interrupted turn or start new turn)."""
    runtime = _get_runtime_manager()

    options = RunOptions(
        model=request.model,
        agent_mode=request.agent_mode,
        tool_call_mode=request.tool_call_mode,
        auto_approve=request.auto_approve,
    )
    runtime.apply_run_options(options)

    return await _execute_with_stream(
        run_coro_factory=lambda: (
            runtime.resume_session_with_message(
                session_id=session_id,
                user_input=request.user_input,
                turn_id=request.turn_id,
                turn_index=request.turn_index,
            )
            if request.user_input
            else runtime.resume_session(
                session_id=session_id,
                turn_id=request.turn_id,
                turn_index=request.turn_index,
                frame_id=request.frame_id,
            )
        ),
        stream=request.stream,
        turn_getter=lambda s: s.get_last_turn(),
        pipeline_getter=lambda: runtime.event_pipeline,
    )


@sessions_router.post("/{session_id}/interrupt", response_model=None)
async def interrupt_session_endpoint(
    session_id: str,
    request: InterruptSessionRequest,
) -> SessionResponse | StreamingResponse:
    """Interrupt a running session turn."""
    runtime = _get_runtime_manager()

    return await _execute_with_stream(
        run_coro_factory=lambda: runtime.interrupt_session(
            session_id=session_id,
            turn_id=request.turn_id,
        ),
        stream=request.stream,
        turn_getter=lambda s: s.get_last_turn(),
        pipeline_getter=lambda: runtime.event_pipeline,
    )


@approvals_router.post("/respond_approval", response_model=None)
async def respond_approval(
    request: RespondApprovalRequest,
) -> SessionResponse | StreamingResponse:
    """Respond to an approval request.

    Delegates to RuntimeManager.respond_approval() which handles:
    - Token validation and expiry checks (tokens are in each ApprovalResponseForm)
    - Finding the correct frame to resume
    - Resuming execution from the paused frame

    Client MUST provide session_id and turn_id explicitly.
    Each response form contains its own approval_token.
    """
    runtime = _get_runtime_manager()

    return await _execute_with_stream(
        run_coro_factory=lambda: runtime.respond_approval(
            session_id=request.session_id,
            turn_id=request.turn_id,
            approval_responses=request.responses,
        ),
        stream=request.stream,
        turn_getter=lambda s: s.get_turn(request.turn_id),
        pipeline_getter=lambda: runtime.event_pipeline,
    )


@events_router.websocket("/ws/{session_id}")
async def events_websocket(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time event streaming."""
    await websocket.accept()
    server_event_subscriber.register_websocket(session_id, websocket)

    try:
        # Send initial session status from store
        store = _get_store()
        session = await store.get_session(session_id)
        if session:
            await websocket.send_json(
                {
                    "type": "session_status",
                    "data": {
                        "session_id": session.id,
                        "status": session.status,
                        "started_at": session.started_at_ms,
                        "updated_at": session.updated_at_ms,
                    },
                }
            )

        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_json()
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
            except Exception as e:
                log.warning("WebSocket error", session_id=session_id, error=str(e))
                break
    finally:
        server_event_subscriber.unregister_websocket(session_id, websocket)


# ========== CLI Entrypoint ==========


@app.command()
@safe_entrypoint("cli.serve.start")
def start(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8001, help="Port to listen on"),
    target_project_root: str = typer.Option(
        None, "--target-project-root", "-t", help="Target project root directory"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Start the runtime server for out-of-process approvals and live monitoring."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    async def _main() -> int:
        global _runtime_manager, _store

        async with bootstrap(
            log_level=level, target_project_root=target_project_root
        ) as ctx:
            _runtime_manager = ctx.get("runtime")
            _store = ctx.get("execution_store")

            if _runtime_manager is None:
                typer.echo("Error: Runtime manager not available", err=True)
                return 1

            if _store is None:
                typer.echo("Error: Execution store not available", err=True)
                return 1

            # Subscribe to events for WebSocket broadcasting
            event_pipeline = ctx.get("event_pipeline")
            if event_pipeline:
                event_pipeline.subscribe(server_event_subscriber)

            # Create and run server
            server_app = create_server_app()
            config = uvicorn.Config(
                server_app,
                host=host,
                port=port,
                log_level=log_level.lower(),
                lifespan="on",
            )
            server = uvicorn.Server(config)

            typer.echo(f"Server starting on {host}:{port}")
            try:
                await server.serve()
            except (KeyboardInterrupt, asyncio.CancelledError):
                log.info("Received interrupt signal, shutting down...")
            finally:
                # Close WebSocket subscriber
                await server_event_subscriber.close()

        return 0

    try:
        code = asyncio.run(_main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        code = 0
    raise typer.Exit(code=code)
