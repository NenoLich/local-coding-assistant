"""WebSocket routes for real-time dashboard updates (live monitoring only)."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.routes.websocket")
router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for live monitoring updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        log.info("WebSocket connected", total=len(self.active_connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        log.info("WebSocket disconnected", total=len(self.active_connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self.active_connections:
            return
        payload = json.dumps(message, default=str)
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            await self.disconnect(conn)

    async def broadcast_stats(self) -> None:
        """Broadcast current live stats to all connections."""

        message: dict[str, Any] = {"type": "stats_update", "data": {}}
        await self.broadcast(message)


manager = ConnectionManager()


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    """WebSocket endpoint for live monitoring updates."""
    await manager.connect(websocket)
    try:
        # Send initial stats
        try:
            app = websocket.app
            monitor = app.state.live_monitor
            if monitor:
                stats = monitor.get_stats()
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "stats_update",
                            "data": {
                                "active_sessions": stats.active_sessions,
                                "running_turns": stats.running_turns,
                                "total_events": stats.total_events,
                                "events_per_second": stats.events_per_second,
                                "error_count_last_minute": stats.error_count_last_minute,
                            },
                        },
                        default=str,
                    )
                )
        except Exception as e:
            log.warning("Failed to send initial stats", error=str(e))

        # Keep connection alive; handle ping/pong
        while True:
            try:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_text(
                            json.dumps({"type": "pong"}, default=str)
                        )
                except json.JSONDecodeError:
                    pass
            except WebSocketDisconnect:
                break
    except Exception as e:
        log.warning("WebSocket error", error=str(e))
    finally:
        await manager.disconnect(websocket)
