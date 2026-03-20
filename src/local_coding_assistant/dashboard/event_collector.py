"""Event collector for dashboard data aggregation."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

from local_coding_assistant.runtime.events import EventType, ExecutionEvent
from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.event_collector")


class EventCollector:
    """Collects and aggregates ExecutionEvents for dashboard consumption."""

    def __init__(self, max_events: int = 10000, max_recent_activity: int = 100):
        self.max_events = max_events
        self.max_recent_activity = max_recent_activity

        # In-memory storage for events
        self._events: deque[ExecutionEvent] = deque(maxlen=max_events)
        self._sessions: dict[str, dict[str, Any]] = {}
        self._runs: dict[str, dict[str, Any]] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def collect_event(self, event: ExecutionEvent) -> None:
        """Collect and process a single ExecutionEvent."""
        async with self._lock:
            self._events.append(event)
            session_updated = await self._process_event(event)

        # Broadcast outside of lock to avoid deadlock
        await self._broadcast_event(event, session_updated)

    async def collect_events(self, events: list[ExecutionEvent]) -> None:
        """Collect multiple ExecutionEvents."""
        session_updates = []
        async with self._lock:
            for event in events:
                self._events.append(event)
                session_updated = await self._process_event(event)
                session_updates.append(session_updated)

        # Broadcast events outside of lock
        for event, session_updated in zip(events, session_updates, strict=True):
            await self._broadcast_event(event, session_updated)

    async def _process_event(self, event: ExecutionEvent) -> bool:
        """Process an event and update internal state.

        Returns:
            True if the session was updated, False otherwise.
        """
        session_id = event.session_id
        log.info(f"Processing event_type: {event.type}")

        # Initialize session if not exists
        session_updated = self._ensure_session_exists(session_id, event)

        # Update session activity
        self._update_session_activity(session_id, event)

        # Process specific event types
        session_updated = await self._handle_event_type(event, session_updated)

        # Add event to frame if frame exists
        self._add_event_to_frame(event)

        return session_updated

    def _ensure_session_exists(self, session_id: str, event: ExecutionEvent) -> bool:
        """Ensure session exists and return whether it was created."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "session_id": session_id,
                "start_time": event.timestamp,
                "last_activity": event.timestamp,
                "frames": [],
                "status": "running",
                "events_count": 0,
            }
            return True
        return False

    def _update_session_activity(self, session_id: str, event: ExecutionEvent) -> None:
        """Update session activity metrics."""
        session = self._sessions[session_id]
        session["last_activity"] = event.timestamp
        session["events_count"] += 1

    async def _handle_event_type(
        self, event: ExecutionEvent, session_updated: bool
    ) -> bool:
        """Handle specific event types and return whether session was updated."""
        handlers = {
            EventType.SESSION_START: self._handle_session_start,
            EventType.TURN_START: self._handle_turn_start,
            EventType.FRAME_START: self._handle_frame_start,
            EventType.FRAME_COMPLETE: self._handle_frame_complete,
            EventType.LLM_START: self._handle_llm_start,
            EventType.LLM_COMPLETE: self._handle_llm_complete,
            EventType.TOOL_START: self._handle_tool_start,
            EventType.TOOL_RESULT: self._handle_tool_result,
            EventType.TURN_COMPLETE: self._handle_turn_complete,
            EventType.ERROR: self._handle_error,
        }

        handler = handlers.get(event.type)
        if handler:
            return await handler(event, session_updated)

        return session_updated

    async def _handle_turn_start(
        self, event: ExecutionEvent, session_updated: bool
    ) -> bool:
        """Handle turn start event."""
        run_id = event.data.get("run_id")
        if run_id:
            run_data = {
                "run_id": run_id,
                "session_id": event.session_id,
                "start_time": event.timestamp,
                "status": "running",
                "events": [],
            }
            self._runs[run_id] = run_data
            return True
        return session_updated

    async def _handle_session_start(
        self, event: ExecutionEvent, session_updated: bool
    ) -> bool:
        """Handle session start event."""
        session = self._sessions[event.session_id]
        session["status"] = "running"
        session["start_time"] = event.timestamp
        return True

    async def _handle_frame_start(
        self, event: ExecutionEvent, session_updated: bool
    ) -> bool:
        """Handle frame start event."""
        if event.frame_id:
            run_id = event.data.get("run_id")
            frame_data = {
                "frame_id": event.frame_id,
                "run_id": run_id,
                "session_id": event.session_id,
                "start_time": event.timestamp,
                "status": "running",
                "events": [],
            }
            session = self._sessions[event.session_id]
            session["frames"].append(frame_data)
            self._runs[event.frame_id] = frame_data
            return True
        return session_updated

    async def _handle_frame_complete(
        self, event: ExecutionEvent, session_updated: bool
    ) -> bool:
        """Handle frame complete event."""
        if event.frame_id and event.frame_id in self._runs:
            run = self._runs[event.frame_id]
            run["status"] = "completed"
            run["end_time"] = event.timestamp
            run["duration"] = (event.timestamp - run["start_time"]).total_seconds()
        return session_updated

    async def _handle_llm_start(
        self, event: ExecutionEvent, session_updated: bool
    ) -> bool:
        """Handle LLM start event."""
        if event.frame_id and event.frame_id in self._runs:
            self._runs[event.frame_id]["llm_start_time"] = event.timestamp
        return session_updated

    async def _handle_llm_complete(
        self, event: ExecutionEvent, session_updated: bool
    ) -> bool:
        """Handle LLM complete event."""
        if event.frame_id and event.frame_id in self._runs:
            self._extract_model_provider_info(event)
        return session_updated

    def _extract_model_provider_info(self, event: ExecutionEvent) -> None:
        """Extract model and provider information from LLM result."""
        if not (event.data and "result" in event.data):
            return

        result = event.data["result"]
        if isinstance(result, dict):
            model = result.get("model")
            provider = result.get("provider")

            if event.frame_id:
                if model:
                    self._runs[event.frame_id]["model"] = model
                if provider:
                    self._runs[event.frame_id]["provider"] = provider

    async def _handle_tool_start(
        self, event: ExecutionEvent, session_updated: bool
    ) -> bool:
        """Handle tool start event."""
        if not (event.frame_id and event.frame_id in self._runs):
            return session_updated

        run = self._runs[event.frame_id]
        if "tool_calls" not in run:
            run["tool_calls"] = []

        tool_name = self._extract_tool_name(event)
        run["tool_calls"].append(
            {"tool_name": tool_name, "start_time": event.timestamp, "status": "running"}
        )
        return session_updated

    def _extract_tool_name(self, event: ExecutionEvent) -> str:
        """Extract tool name from event data."""
        tool_call = event.data.get("tool_call")
        if hasattr(tool_call, "name"):
            return tool_call.name
        elif isinstance(tool_call, dict):
            return tool_call.get("name", "unknown")
        else:
            return str(tool_call) if tool_call else "unknown"

    async def _handle_tool_result(
        self, event: ExecutionEvent, session_updated: bool
    ) -> bool:
        """Handle tool result event."""
        if not (event.frame_id and event.frame_id in self._runs):
            return session_updated

        run = self._runs[event.frame_id]
        if "tool_calls" in run:
            tool_calls = run["tool_calls"]
            if tool_calls and tool_calls[-1]["status"] == "running":
                self._update_tool_call_status(tool_calls[-1], event)
        return session_updated

    def _update_tool_call_status(
        self, tool_call: dict[str, Any], event: ExecutionEvent
    ) -> None:
        """Update tool call status and duration."""
        tool_call["end_time"] = event.timestamp
        response = event.data.get("response")
        tool_call["status"] = (
            "completed" if (response and response.get("success", True)) else "failed"
        )
        tool_call["duration"] = (
            event.timestamp - tool_call["start_time"]
        ).total_seconds()

    async def _handle_turn_complete(
        self, event: ExecutionEvent, session_updated: bool
    ) -> bool:
        """Handle turn complete event."""
        run_id = event.data.get("run_id")
        if run_id and run_id in self._runs:
            run = self._runs[run_id]
            run["status"] = "completed"
            run["end_time"] = event.timestamp
            run["final_answer"] = event.data.get("final_answer")
            # Calculate duration if start_time exists
            if "start_time" in run:
                run["duration"] = (event.timestamp - run["start_time"]).total_seconds()
            return True

        # Also update session status
        if event.session_id in self._sessions:
            self._sessions[event.session_id]["status"] = "completed"
            return True
        return session_updated

    async def _handle_error(self, event: ExecutionEvent, session_updated: bool) -> bool:
        """Handle error event."""
        session = self._sessions[event.session_id]
        session["status"] = "error"

        if event.frame_id and event.frame_id in self._runs:
            run = self._runs[event.frame_id]
            run["status"] = "error"
            run["error"] = event.data.get("error", "Unknown error")

        return True

    def _add_event_to_frame(self, event: ExecutionEvent) -> None:
        """Add event to frame if frame exists."""
        if event.frame_id and event.frame_id in self._runs:
            self._runs[event.frame_id]["events"].append(event)

        # Also add event to run if it has a run_id in data
        run_id = event.data.get("run_id")
        if run_id and run_id in self._runs:
            self._runs[run_id]["events"].append(event)

    async def get_dashboard_stats(self) -> dict[str, Any]:
        """Get aggregated dashboard statistics."""
        async with self._lock:
            # Filter to only include actual runs, not frames
            actual_runs = [
                run
                for run in self._runs.values()
                if "run_id" in run and "frame_id" not in run
            ]
            total_runs = len(actual_runs)

            # Calculate success rate
            completed_runs = sum(
                1 for run in actual_runs if run.get("status") == "completed"
            )
            error_runs = sum(1 for run in actual_runs if run.get("status") == "error")
            success_rate = (completed_runs / total_runs * 100) if total_runs > 0 else 0

            # Calculate average duration
            completed_with_duration = [
                run
                for run in actual_runs
                if run.get("status") == "completed" and "duration" in run
            ]
            avg_duration = (
                (
                    sum(run["duration"] for run in completed_with_duration)
                    / len(completed_with_duration)
                )
                if completed_with_duration
                else 0
            )

            # Count active sessions (last activity within 5 minutes)
            now = datetime.now(UTC)
            active_threshold = now - timedelta(minutes=5)
            active_sessions = sum(
                1
                for session in self._sessions.values()
                if session["last_activity"] > active_threshold
            )

            return {
                "total_runs": total_runs,
                "success_rate": f"{success_rate:.1f}%",
                "avg_duration": f"{avg_duration:.1f}s",
                "active_sessions": active_sessions,
                "completed_runs": completed_runs,
                "error_runs": error_runs,
            }

    async def get_events_by_session(self, session_id: str) -> list[ExecutionEvent]:
        """Get all events for a specific session."""
        async with self._lock:
            return [event for event in self._events if event.session_id == session_id]

    async def get_events_by_run(self, run_id: str) -> list[ExecutionEvent]:
        """Get all events for a specific run."""
        async with self._lock:
            if run_id not in self._runs:
                return []
            # Return the events stored in the run data
            return self._runs[run_id].get("events", [])

    async def get_events_by_frame(self, frame_id: str) -> list[ExecutionEvent]:
        """Get all events for a specific frame."""
        async with self._lock:
            if frame_id not in self._runs:
                return []
            return self._runs[frame_id].get("events", [])

    async def get_recent_activity_simple(self) -> list[dict[str, Any]]:
        """Get recent activity as a simple list (for backward compatibility)."""
        async with self._lock:
            # Filter to only include actual runs, not frames
            actual_runs = [
                run
                for run in self._runs.items()
                if "run_id" in run[1] and "frame_id" not in run[1]
            ]
            # Get recent runs sorted by start time
            recent_runs = sorted(
                [
                    {
                        "run_id": run_id,
                        "status": run.get("status", "unknown"),
                        "timestamp": run.get("start_time", datetime.now(UTC)).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "duration": run.get("duration", 0),
                    }
                    for run_id, run in actual_runs[-self.max_recent_activity :]
                ],
                key=lambda x: x["timestamp"],
                reverse=True,
            )
            return recent_runs

    async def get_recent_activity(self, limit: int = 50) -> dict[str, Any]:
        """Get recent activity data."""
        async with self._lock:
            # Get recent runs sorted by start time
            recent_runs = sorted(
                [
                    {
                        "run_id": run_id,
                        "status": run.get("status", "unknown"),
                        "timestamp": run.get("start_time", datetime.now(UTC)).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "duration": run.get("duration", 0),
                    }
                    for run_id, run in list(self._runs.items())[-limit:]
                ],
                key=lambda x: x["timestamp"],
                reverse=True,
            )

            return {
                "activities": recent_runs,
                "last_updated": datetime.now(UTC).isoformat(),
            }

    async def get_sessions(self) -> list[dict[str, Any]]:
        """Get all sessions as a list."""
        async with self._lock:
            return list(self._sessions.values())

    async def get_runs_simple(self) -> list[dict[str, Any]]:
        """Get all runs as a simple list (for backward compatibility)."""
        async with self._lock:
            runs_list = []
            for run_id, run in self._runs.items():
                # Only include items that have run_id and don't have frame_id (i.e., actual runs, not frames)
                if "run_id" in run and "frame_id" not in run:
                    runs_list.append(
                        {
                            "run_id": run_id,
                            "session_id": run["session_id"],
                            "status": run.get("status", "unknown"),
                            "start_time": run.get("start_time"),
                            "end_time": run.get("end_time"),
                            "duration": run.get("duration", 0),
                            "events_count": len(run.get("events", [])),
                        }
                    )
            return runs_list

    async def get_runs(self, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """Get paginated runs data."""
        async with self._lock:
            runs_list = []
            for run_id, run in list(self._runs.items())[offset : offset + limit]:
                runs_list.append(
                    {
                        "run_id": run_id,
                        "session_id": run["session_id"],
                        "status": run.get("status", "unknown"),
                        "start_time": run.get("start_time"),
                        "end_time": run.get("end_time"),
                        "duration": run.get("duration", 0),
                        "events_count": len(run.get("events", [])),
                    }
                )

            return {
                "runs": runs_list,
                "total": len(self._runs),
                "offset": offset,
                "limit": limit,
            }

    async def get_run_details(self, run_id: str) -> dict[str, Any] | None:
        """Get detailed information about a specific run."""
        async with self._lock:
            if run_id not in self._runs:
                return None

            run = self._runs[run_id]
            return {
                "run_id": run_id,
                "session_id": run["session_id"],
                "status": run.get("status", "unknown"),
                "start_time": run.get("start_time"),
                "end_time": run.get("end_time"),
                "duration": run.get("duration", 0),
                "model": run.get("model"),
                "provider": run.get("provider"),
                "events": [
                    {
                        "type": event.type.value,
                        "timestamp": event.timestamp,
                        "data": event.data,
                    }
                    for event in run.get("events", [])
                ],
            }

    async def get_paginated_runs(
        self, offset: int = 0, limit: int = 10
    ) -> dict[str, Any]:
        """Get paginated runs data (for test compatibility)."""
        async with self._lock:
            runs_list = []
            # Filter to only include actual runs, not frames
            all_runs = [
                run
                for run in self._runs.items()
                if "run_id" in run[1] and "frame_id" not in run[1]
            ]
            total_runs = len(all_runs)

            for run_id, run in all_runs[offset : offset + limit]:
                runs_list.append(
                    {
                        "run_id": run_id,
                        "session_id": run["session_id"],
                        "status": run.get("status", "unknown"),
                        "start_time": run.get("start_time"),
                        "end_time": run.get("end_time"),
                        "duration": run.get("duration", 0),
                        "events_count": len(run.get("events", [])),
                    }
                )

            return {
                "items": runs_list,
                "total": total_runs,
                "offset": offset,
                "limit": limit,
                "has_next": offset + limit < total_runs,
                "has_prev": offset > 0,
            }

    async def get_analytics_metrics(self, days: int = 30) -> dict[str, Any]:
        """Get analytics metrics for the specified time period."""
        async with self._lock:
            now = datetime.now(UTC)
            cutoff_date = now - timedelta(days=days)

            # Filter runs within the time period
            recent_runs = self._filter_runs_by_date(cutoff_date)

            # Calculate various analytics
            daily_stats = self._calculate_daily_stats(recent_runs)
            tool_usage = self._calculate_tool_usage(recent_runs)
            error_patterns = self._calculate_error_patterns(recent_runs)
            latency_data = self._calculate_latency_data(recent_runs)
            token_usage = self._calculate_token_usage(recent_runs)

            return {
                "daily_stats": daily_stats,
                "tool_usage": dict(
                    sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:10]
                ),
                "error_patterns": error_patterns,
                "latency_data": latency_data,
                "token_usage": token_usage,
                "period_days": days,
                "total_runs_in_period": len(recent_runs),
            }

    def _filter_runs_by_date(self, cutoff_date: datetime) -> list[dict[str, Any]]:
        """Filter runs within the specified date range."""
        return [
            run
            for run in self._runs.values()
            if run.get("start_time") and run["start_time"] >= cutoff_date
        ]

    def _calculate_daily_stats(
        self, runs: list[dict[str, Any]]
    ) -> dict[str, dict[str, int]]:
        """Calculate daily statistics for runs."""
        daily_stats = {}
        for run in runs:
            if run.get("start_time"):
                date_key = run["start_time"].strftime("%Y-%m-%d")
                if date_key not in daily_stats:
                    daily_stats[date_key] = {"total": 0, "completed": 0, "errors": 0}

                daily_stats[date_key]["total"] += 1
                if run.get("status") == "completed":
                    daily_stats[date_key]["completed"] += 1
                elif run.get("status") == "error":
                    daily_stats[date_key]["errors"] += 1

        return daily_stats

    def _calculate_tool_usage(self, runs: list[dict[str, Any]]) -> dict[str, int]:
        """Calculate tool usage statistics."""
        tool_usage = {}
        for run in runs:
            for event in run.get("events", []):
                if hasattr(event, "type") and event.type == EventType.TOOL_RESULT:
                    tool_name = event.data.get("tool", "unknown")
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

        return tool_usage

    def _calculate_error_patterns(self, runs: list[dict[str, Any]]) -> dict[str, int]:
        """Calculate error pattern statistics."""
        error_patterns = {}
        for run in runs:
            for event in run.get("events", []):
                if hasattr(event, "type") and event.type == EventType.ERROR:
                    error_msg = event.data.get("error", "Unknown error")
                    pattern = self._classify_error_pattern(error_msg)
                    error_patterns[pattern] = error_patterns.get(pattern, 0) + 1

        return error_patterns

    def _classify_error_pattern(self, error_msg: str) -> str:
        """Classify error message into pattern categories."""
        error_lower = error_msg.lower()
        if "timeout" in error_lower:
            return "Timeout Errors"
        elif "permission" in error_lower or "access" in error_lower:
            return "Permission Errors"
        elif "file not found" in error_lower or "not found" in error_lower:
            return "File Not Found"
        else:
            return "Other Errors"

    def _calculate_latency_data(
        self, runs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Calculate latency data for runs."""
        latency_data = []
        for run in runs:
            if run.get("duration") and run.get("start_time"):
                latency_data.append(
                    {
                        "date": run["start_time"].strftime("%Y-%m-%d"),
                        "duration": run["duration"],
                    }
                )

        return latency_data

    def _calculate_token_usage(
        self, runs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Calculate token usage data for runs."""
        token_usage = []
        for run in runs:
            if run.get("start_time"):
                tokens = sum(
                    event.data.get("tokens", 0) for event in run.get("events", [])
                )
                token_usage.append(
                    {"date": run["start_time"].strftime("%Y-%m-%d"), "tokens": tokens}
                )

        return token_usage

    async def get_active_sessions(self) -> list[dict[str, Any]]:
        """Get currently active sessions."""
        sessions = []
        async with self._lock:
            for session_id, session in self._sessions.items():
                if session.get("status") == "running":
                    # Find current frame for this session
                    current_frame = None
                    for frame in session.get("frames", []):
                        if frame.get("status") == "running":
                            current_frame = {
                                "frame_id": frame["frame_id"],
                                "start_time": frame["start_time"].isoformat()
                                if frame.get("start_time")
                                else None,
                                "duration": 0,  # Will be calculated on frontend
                            }
                            break

                    sessions.append(
                        {
                            "session_id": session_id,
                            "start_time": session["start_time"].isoformat()
                            if session.get("start_time")
                            else None,
                            "last_activity": session["last_activity"].isoformat()
                            if session.get("last_activity")
                            else None,
                            "events_count": session.get("events_count", 0),
                            "current_frame": current_frame,
                        }
                    )

        return sessions

    async def get_top_failing_tools(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the most frequently failing tools."""
        async with self._lock:
            tool_failures = {}

            for run in self._runs.values():
                if run.get("status") == "error":
                    for event in run.get("events", []):
                        if (
                            hasattr(event, "type")
                            and event.type == EventType.TOOL_RESULT
                        ):
                            tool_name = event.data.get("tool", "unknown")
                            # Check if this tool call resulted in an error
                            if (
                                event.data.get("success") is False
                                or "error" in str(event.data).lower()
                            ):
                                tool_failures[tool_name] = (
                                    tool_failures.get(tool_name, 0) + 1
                                )

            # Sort and return top failing tools
            sorted_failures = sorted(
                tool_failures.items(), key=lambda x: x[1], reverse=True
            )
            return [
                {"tool": tool, "failure_count": count}
                for tool, count in sorted_failures[:limit]
            ]

    async def _broadcast_event(
        self, event: ExecutionEvent, session_updated: bool = False
    ) -> None:
        """Broadcast event to WebSocket connections via ConnectionManager."""
        try:
            # Import here to avoid circular imports
            from local_coding_assistant.dashboard.routes.websocket import manager

            # Use the global manager instance
            await manager.broadcast_event(event)

            # Also broadcast session update if session was updated
            if session_updated:
                await manager.broadcast_sessions_update()

        except ImportError:
            # WebSocket routes not yet imported
            pass
        except Exception as e:
            log.warning(f"Failed to broadcast event: {e}")


# Global instance for the dashboard
_event_collector: EventCollector | None = None


def get_event_collector() -> EventCollector:
    """Get the global EventCollector instance."""
    global _event_collector
    if _event_collector is None:
        _event_collector = EventCollector()
    return _event_collector
