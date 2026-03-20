"""WebSocket routes for real-time dashboard updates."""

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from local_coding_assistant.runtime.events import ExecutionEvent
from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.routes.websocket")
router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        try:
            log.debug("Accepting WebSocket connection...")
            await websocket.accept()
            log.debug("WebSocket accepted, adding to connections...")

            self.active_connections.append(websocket)
            log.debug(
                f"Added to active_connections, count: {len(self.active_connections)}"
            )

            log.info(
                f"WebSocket connected. Total connections: {len(self.active_connections)}"
            )

            # Send initial data
            log.debug("Sending initial data...")
            await self.send_initial_data(websocket)
            log.debug("Initial data sent successfully")
        except Exception as e:
            log.error(f"Error in WebSocket connect: {e}")
            import traceback

            log.error(traceback.format_exc())
            raise

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        log.info(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def send_initial_data(self, websocket: WebSocket):
        """Send initial dashboard data to a newly connected client."""
        try:
            from local_coding_assistant.dashboard.event_collector import (
                get_event_collector,
            )

            event_collector = get_event_collector()

            # Send current stats
            stats = await event_collector.get_dashboard_stats()
            await websocket.send_text(
                json.dumps({"type": "stats_update", "data": stats})
            )

            # Send recent activity
            activity = await event_collector.get_recent_activity(limit=10)
            await websocket.send_text(
                json.dumps({"type": "activity_update", "data": activity})
            )

            # Send active sessions
            active_sessions = await event_collector.get_active_sessions()
            await websocket.send_text(
                json.dumps({"type": "sessions_update", "data": active_sessions})
            )

        except Exception as e:
            log.error(f"Error sending initial data: {e}")

    async def broadcast_event(self, event: ExecutionEvent):
        """Broadcast an event to all connected clients."""
        if not self.active_connections:
            return

        message = {
            "type": "event_update",
            "data": {
                "event_type": event.type.value,
                "session_id": event.session_id,
                "frame_id": event.frame_id,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data,
            },
        }

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                log.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(connection)

        # Remove disconnected connections
        for conn in disconnected:
            await self.disconnect(conn)

    async def broadcast_stats_update(self):
        """Broadcast statistics update to all WebSocket connections."""
        if not self.active_connections:
            return

        try:
            from local_coding_assistant.dashboard.event_collector import (
                get_event_collector,
            )

            event_collector = get_event_collector()
            stats = await event_collector.get_dashboard_stats()

            message = {"type": "stats_update", "data": stats}

            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    log.warning(f"Failed to send stats update: {e}")
                    disconnected.append(connection)

            # Remove disconnected connections
            for conn in disconnected:
                await self.disconnect(conn)

        except Exception as e:
            log.error(f"Error broadcasting stats update: {e}")

    async def broadcast_sessions_update(self):
        """Broadcast sessions update to all WebSocket connections."""
        if not self.active_connections:
            return

        try:
            from local_coding_assistant.dashboard.event_collector import (
                get_event_collector,
            )

            event_collector = get_event_collector()
            active_sessions = await event_collector.get_active_sessions()

            message = {"type": "sessions_update", "data": active_sessions}

            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    log.warning(f"Failed to send sessions update: {e}")
                    disconnected.append(connection)

            # Remove disconnected connections
            for conn in disconnected:
                await self.disconnect(conn)

        except Exception as e:
            log.error(f"Error broadcasting sessions update: {e}")


# Global connection manager
manager = ConnectionManager()


# Export broadcast functions for use by other modules
async def broadcast_stats_update():
    """Broadcast statistics update to all WebSocket connections."""
    await manager.broadcast_stats_update()


async def broadcast_sessions_update():
    """Broadcast sessions update to all WebSocket connections."""
    await manager.broadcast_sessions_update()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    log.info("WebSocket connection attempt received")

    try:
        await manager.connect(websocket)
        log.info("WebSocket connected successfully")

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Keep connection alive and handle incoming messages
                data = await websocket.receive_text()
                log.debug(f"Received WebSocket message: {data[:100]}...")

                try:
                    message = json.loads(data)

                    # Handle different message types from client
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    elif message.get("type") == "subscribe":
                        # Client can subscribe to specific event types
                        log.info(
                            f"Client subscribed to: {message.get('events', 'all')}"
                        )
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "subscribed",
                                    "events": message.get("events", "all"),
                                }
                            )
                        )
                    elif message.get("type") == "request_sessions_update":
                        # Send active sessions update
                        await manager.broadcast_sessions_update()

                except json.JSONDecodeError:
                    log.warning(f"Invalid JSON received: {data}")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "Invalid JSON format"})
                    )
                except Exception as e:
                    log.error(f"Error handling WebSocket message: {e}")
                    await websocket.send_text(
                        json.dumps(
                            {"type": "error", "message": "Internal server error"}
                        )
                    )

            except WebSocketDisconnect:
                log.info("WebSocket disconnected normally")
                await manager.disconnect(websocket)
                break
            except Exception as e:
                log.error(f"WebSocket error: {e}")
                await manager.disconnect(websocket)
                break

    except Exception as e:
        log.error(f"WebSocket endpoint error: {e}")
        await manager.disconnect(websocket)
