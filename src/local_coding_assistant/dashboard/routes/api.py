"""API routes for dashboard data."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from local_coding_assistant.dashboard.event_collector import get_event_collector
from local_coding_assistant.dashboard.models import (
    ActivityItem,
    DashboardStats,
    FrameDetail,
    FrameSummary,
    PaginatedResponse,
    RecentActivityResponse,
    RunDetail,
    RunsListResponse,
    RunSummary,
)
from local_coding_assistant.runtime.events import EventType, ExecutionEvent
from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.routes.api")
router = APIRouter()

# Module-level Query singletons to avoid B008 linting errors
QUERY_LIMIT_100 = Query(100, ge=1, le=1000)
QUERY_OFFSET_0 = Query(0, ge=0)
QUERY_STATUS_FILTER = Query(None, description="Filter by status")
QUERY_DATE_FROM_FILTER = Query(None, description="Filter from date")
QUERY_DATE_TO_FILTER = Query(None, description="Filter to date")
QUERY_MODEL_FILTER = Query(None, description="Filter by model")
QUERY_PROVIDER_FILTER = Query(None, description="Filter by provider")
QUERY_LIMIT_50 = Query(50, ge=1, le=200)
QUERY_LIMIT_10 = Query(10, ge=1, le=50)
QUERY_DAYS_30 = Query(30, ge=1, le=365, description="Number of days to analyze")
QUERY_EXPORT_FORMAT = Query(None, pattern="^(csv|json)$", description="Export format")
QUERY_EVENT_TYPE_FILTER = Query(None, description="Filter by event type")


@router.post("/events/ingest")
async def ingest_event(event_data: dict[str, Any]) -> dict[str, str]:
    """Receive an event from the agent process and forward to EventCollector."""
    collector = get_event_collector()

    # Parse timestamp
    try:
        timestamp = datetime.fromisoformat(event_data["timestamp"])
    except (ValueError, KeyError):
        timestamp = datetime.now()

    # Create ExecutionEvent from dict
    execution_event = ExecutionEvent(
        type=EventType(event_data["type"]),  # Changed from event_type to type
        session_id=event_data["session_id"],
        frame_id=event_data.get("frame_id"),
        data=event_data.get("data", {}),
        timestamp=timestamp,
    )

    # Collect and broadcast with error handling
    try:
        await collector.collect_event(execution_event)
    except Exception as e:
        log.error(f"Failed to collect event: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to collect event: {e!s}"
        ) from e

    return {"status": "ok", "event_type": execution_event.type.value}


@router.get("/status")
async def api_status() -> dict[str, str]:
    """API status endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats() -> DashboardStats:
    """Get dashboard statistics."""
    collector = get_event_collector()
    stats = await collector.get_dashboard_stats()
    return DashboardStats(**stats)


@router.get("/recent-activity", response_model=RecentActivityResponse)
async def get_recent_activity(limit: int = QUERY_LIMIT_50) -> RecentActivityResponse:
    """Get recent activity data."""
    collector = get_event_collector()
    data = await collector.get_recent_activity(limit=limit)

    activities = [ActivityItem(**activity) for activity in data["activities"]]
    return RecentActivityResponse(
        activities=activities, last_updated=data["last_updated"]
    )


@router.get("/runs", response_model=RunsListResponse)
async def get_runs(
    limit: int = QUERY_LIMIT_100,
    offset: int = QUERY_OFFSET_0,
    status: str | None = QUERY_STATUS_FILTER,
    date_from: datetime | None = QUERY_DATE_FROM_FILTER,
    date_to: datetime | None = QUERY_DATE_TO_FILTER,
    model: str | None = QUERY_MODEL_FILTER,
    provider: str | None = QUERY_PROVIDER_FILTER,
) -> RunsListResponse:
    """Get paginated runs data with optional filtering."""
    collector = get_event_collector()
    data = await collector.get_runs(limit=10000, offset=0)  # Get all runs for filtering

    # Apply filters using helper function
    runs = await _filter_runs(
        collector, data["runs"], status, date_from, date_to, model, provider
    )

    # Apply pagination after filtering
    total_filtered = len(runs)
    paginated_runs = runs[offset : offset + limit]

    # Convert to RunSummary models using helper function
    run_summaries = []
    for run in paginated_runs:
        # Get run details to extract tokens
        run_details = await collector.get_run_details(run["run_id"])
        tokens_used = 0
        if run_details:
            tokens_used = await _extract_tokens_from_events(
                run_details.get("events", [])
            )

        run_summaries.append(
            RunSummary(
                run_id=run["run_id"],
                session_id=run["session_id"],
                status=run["status"],
                start_time=run["start_time"],
                end_time=run.get("end_time"),
                duration=run.get("duration", 0),
                events_count=run["events_count"],
                tokens_used=tokens_used,
            )
        )

    has_next = offset + limit < total_filtered
    has_prev = offset > 0

    return RunsListResponse(
        items=run_summaries,
        total=total_filtered,
        offset=offset,
        limit=limit,
        has_next=has_next,
        has_prev=has_prev,
    )


@router.get("/runs/export")
async def export_runs(
    export_format: str = QUERY_EXPORT_FORMAT,
    date_from: datetime | None = QUERY_DATE_FROM_FILTER,
    date_to: datetime | None = QUERY_DATE_TO_FILTER,
    status: str | None = QUERY_STATUS_FILTER,
    model: str | None = QUERY_MODEL_FILTER,
    provider: str | None = QUERY_PROVIDER_FILTER,
):
    """Export runs data in CSV or JSON format."""
    collector = get_event_collector()

    # Default to CSV if no format specified
    if export_format is None:
        export_format = "csv"

    # Get all runs (without pagination for export)
    data = await collector.get_runs(limit=10000, offset=0)
    runs = data["runs"]

    # Apply filters using helper function
    runs = await _filter_runs(
        collector, runs, status, date_from, date_to, model, provider
    )

    if export_format.lower() == "csv":
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(
            [
                "run_id",
                "session_id",
                "status",
                "start_time",
                "end_time",
                "duration",
                "events_count",
            ]
        )

        # Write data
        for run in runs:
            writer.writerow(
                [
                    run["run_id"],
                    run["session_id"],
                    run.get("status", ""),
                    run.get("start_time", ""),
                    run.get("end_time", ""),
                    run.get("duration", 0),
                    run.get("events_count", 0),
                ]
            )

        output.seek(0)
        from fastapi.responses import Response

        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=runs_export.csv"},
        )

    else:  # JSON format
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content=runs,
            headers={"Content-Disposition": "attachment; filename=runs_export.json"},
        )


@router.get("/runs/export/frames")
async def export_frames(
    export_format: str = Query(
        None, pattern="^(csv|json)$", description="Export format"
    ),
    date_from: datetime | None = QUERY_DATE_FROM_FILTER,
    date_to: datetime | None = QUERY_DATE_TO_FILTER,
    status: str | None = QUERY_STATUS_FILTER,
    model: str | None = QUERY_MODEL_FILTER,
    provider: str | None = QUERY_PROVIDER_FILTER,
):
    """Export frames data in CSV or JSON format."""
    # For now, frames are the same as runs
    return await export_runs(
        export_format=export_format,
        date_from=date_from,
        date_to=date_to,
        status=status,
        model=model,
        provider=provider,
    )


@router.get("/runs/models-providers")
async def get_runs_models_providers() -> dict[str, list[str]]:
    """Get unique models and providers from actual runs."""
    collector = get_event_collector()
    data = await collector.get_runs(limit=10000, offset=0)  # Get all runs
    models = set()
    providers = set()

    for run in data["runs"]:
        # Get model and provider from run details
        run_details = await collector.get_run_details(run["run_id"])
        if run_details:
            # Check if model and provider are stored directly in the run
            if run_details.get("model") and run_details["model"] != "unknown":
                models.add(run_details["model"])
            if run_details.get("provider") and run_details["provider"] != "unknown":
                providers.add(run_details["provider"])

            # Fallback: extract from LLM complete events if not stored
            if not run_details.get("model") or not run_details.get("provider"):
                model, provider = await _extract_model_provider_from_events(
                    run_details.get("events", [])
                )
                if model:
                    models.add(model)
                if provider:
                    providers.add(provider)

    return {"models": sorted(list(models)), "providers": sorted(list(providers))}


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run_details(
    run_id: str, event_type: str | None = QUERY_EVENT_TYPE_FILTER
) -> RunDetail:
    """Get detailed information about a specific run."""
    # Exclude export routes
    if run_id in ["export", "frames"]:
        raise HTTPException(status_code=404, detail="Not found")

    collector = get_event_collector()
    details = await collector.get_run_details(run_id)
    if details is None:
        raise HTTPException(status_code=404, detail="Run not found")

    # Filter events by type if specified
    events = details.get("events", [])
    if event_type:
        events = [event for event in events if event.get("type") == event_type]

    # Extract tokens using helper function
    tokens_used = await _extract_tokens_from_events(events)

    # Extract final answer and error message
    final_answer = None
    error_message = None

    # Check for final_answer at the top level first
    if "final_answer" in details:
        final_answer = details["final_answer"]

    # Also check for error_message at the top level
    if "error_message" in details:
        error_message = details["error_message"]

    # Also check within events
    for event in events:
        if event.get("type") == "frame_complete":
            final_answer = event.get("data", {}).get("answer")
        elif event.get("type") == "error":
            error_message = event.get("data", {}).get("error")

    return RunDetail(
        run_id=details["run_id"],
        session_id=details["session_id"],
        status=details["status"],
        start_time=details["start_time"],
        end_time=details.get("end_time"),
        duration=details.get("duration", 0),
        events_count=len(events),
        tokens_used=tokens_used,
        events=events,
        final_answer=final_answer,
        error_message=error_message,
    )


@router.get("/frames/{frame_id}", response_model=FrameDetail)
async def get_frame_details(frame_id: str) -> FrameDetail:
    """Get detailed information about a specific frame."""
    collector = get_event_collector()

    # First try to get the frame as a run (frames and runs are the same in current implementation)
    run_details = await collector.get_run_details(frame_id)
    if run_details is None:
        raise HTTPException(status_code=404, detail="Frame not found")

    events = run_details.get("events", [])

    # Extract frame actions from FRAME_COMPLETE event
    frame_actions = await _extract_frame_actions(events)

    # Extract basic frame information
    prompt_context, llm_response, actions = await _extract_basic_frame_info(events)

    # Use frame actions if available, otherwise fall back to event-based actions
    final_actions = frame_actions if frame_actions else actions

    # Extract tool calls from events or frame actions
    tool_calls = await _extract_tool_calls_from_events(events, frame_actions)

    return FrameDetail(
        frame_id=run_details["run_id"],
        run_id=run_details["run_id"],  # In current implementation, frame_id == run_id
        status=run_details["status"],
        start_time=run_details["start_time"],
        end_time=run_details.get("end_time"),
        duration=run_details.get("duration", 0),
        action_count=len(final_actions),
        prompt_context=prompt_context,
        llm_response=llm_response,
        tool_calls=tool_calls,
        actions=final_actions,
    )


@router.get("/frames", response_model=PaginatedResponse)
async def get_frames(
    limit: int = QUERY_LIMIT_100,
    offset: int = QUERY_OFFSET_0,
    status: str | None = QUERY_STATUS_FILTER,
) -> PaginatedResponse:
    """Get paginated frames data."""
    # For now, frames are the same as runs
    runs_response = await get_runs(limit=limit, offset=offset, status=status)

    frames = []
    for run in runs_response.items:
        frames.append(
            FrameSummary(
                frame_id=run.run_id,
                run_id=run.run_id,
                status=run.status,
                start_time=run.start_time,
                end_time=run.end_time,
                duration=run.duration,
                action_count=run.events_count,
            )
        )

    return PaginatedResponse(
        items=frames,
        total=runs_response.total,
        offset=runs_response.offset,
        limit=runs_response.limit,
        has_next=runs_response.has_next,
        has_prev=runs_response.has_prev,
    )


@router.get("/analytics/metrics")
async def get_analytics_metrics(days: int = QUERY_DAYS_30) -> dict[str, Any]:
    """Get analytics metrics for the specified time period."""
    collector = get_event_collector()
    metrics = await collector.get_analytics_metrics(days=days)
    return metrics


@router.get("/analytics/failing-tools")
async def get_top_failing_tools(limit: int = QUERY_LIMIT_10) -> list[dict[str, Any]]:
    """Get the most frequently failing tools."""
    collector = get_event_collector()
    failing_tools = await collector.get_top_failing_tools(limit=limit)
    return failing_tools


@router.get("/sessions")
async def get_active_sessions() -> list[dict[str, Any]]:
    """Get currently active sessions."""
    collector = get_event_collector()
    sessions = await collector.get_active_sessions()
    return sessions


@router.get("/sessions/{session_id}/events")
async def get_events_by_session(session_id: str) -> list[dict[str, Any]]:
    """Get events for a specific session."""
    collector = get_event_collector()
    events = await collector.get_events_by_session(session_id)
    return [event.__dict__ for event in events]


async def _extract_frame_actions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract frame actions from events and FRAME_COMPLETE event."""
    frame_actions = []

    for event in events:
        if event.get("type") == "frame_complete":
            frame_data = event.get("data", {}).get("frame")
            if frame_data and isinstance(frame_data, dict):
                frame_actions_list = frame_data.get("actions", [])

                # Convert frame actions to timeline format
                for action in frame_actions_list:
                    if isinstance(action, dict):
                        # Convert action record to timeline format
                        action_type = "unknown"
                        action_kind = action.get("kind")
                        if action_kind == "tool_call":
                            action_type = "tool_call"
                        elif action_kind == "llm_message":
                            action_type = "llm_call"
                        elif action_kind == "observation":
                            action_type = "observation"

                        frame_actions.append(
                            {
                                "type": action_type,
                                "data": {
                                    "name": action.get("name"),
                                    "kind": action_kind,
                                    "started_at": action.get("started_at"),
                                    "finished_at": action.get("finished_at"),
                                    "llm_metrics": action.get("llm_metrics"),
                                    "tool_calls": action.get("tool_calls", []),
                                    "tool_trace": action.get("tool_trace"),
                                },
                                "timestamp": action.get("started_at")
                                or action.get("finished_at"),
                            }
                        )

    return frame_actions


async def _extract_tool_calls_from_events(
    events: list[dict[str, Any]], frame_actions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Extract tool calls from individual events or frame actions."""
    tool_calls = []

    # First pass: extract from individual tool_call events
    for event in events:
        if event.get("type") == "tool_call":
            event_data = event.get("data", {})
            tool_calls.append(
                {
                    "tool": event_data.get("tool"),
                    "args": event_data.get("args"),
                    "result": event_data.get("result"),
                    "timestamp": event.get("timestamp"),
                }
            )

    # If no individual tool_call events, extract from frame actions
    if not tool_calls and frame_actions:
        for action in frame_actions:
            if action["data"]["kind"] == "tool_call" and action["data"]["tool_trace"]:
                tool_trace = action["data"]["tool_trace"]
                tool_calls.append(
                    {
                        "tool": tool_trace.get("tool_name", "unknown"),
                        "args": tool_trace.get("input"),
                        "result": tool_trace.get("output"),
                        "timestamp": action["timestamp"],
                    }
                )

    return tool_calls


async def _extract_basic_frame_info(
    events: list[dict[str, Any]],
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    """Extract basic frame information: prompt context, LLM response, and actions."""
    prompt_context = None
    llm_response = None
    actions = []

    for event in events:
        event_data = event.get("data", {})
        if event.get("type") == "llm_call":
            prompt_context = event_data.get("prompt")
            llm_response = event_data.get("response")
        elif event.get("type") in [
            "tool_start",
            "tool_complete",
            "error",
            "llm_start",
            "llm_complete",
        ]:
            actions.append(
                {
                    "type": event.get("type"),
                    "data": event_data,
                    "timestamp": event.get("timestamp"),
                }
            )

    return prompt_context, llm_response, actions


async def _extract_tokens_from_events(events: list[dict[str, Any]]) -> int:
    """Extract total tokens from frame_complete events."""
    for event in events:
        if event.get("type") == "frame_complete":
            frame_data = event.get("data", {}).get("frame")
            if frame_data and isinstance(frame_data, dict):
                result = frame_data.get("result")
                if result and isinstance(result, dict):
                    total_tokens = result.get("total_tokens")
                    if isinstance(total_tokens, (int, float)):
                        return int(total_tokens)
    return 0


async def _extract_model_provider_from_events(
    events: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    """Extract model and provider from llm_complete events."""
    model = None
    provider = None

    for event in events:
        if event.get("type") == "llm_complete":
            result = event.get("data", {}).get("result")
            if isinstance(result, dict):
                if not model:
                    model_name = result.get("model")
                    if model_name and model_name != "unknown":
                        model = model_name
                if not provider:
                    provider_name = result.get("provider")
                    if provider_name and provider_name != "unknown":
                        provider = provider_name

                if model and provider:
                    break

    return model, provider


async def _filter_runs(
    collector,
    runs: list[dict[str, Any]],
    status: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    """Apply filters to a list of runs."""
    if status:
        runs = [run for run in runs if run.get("status") == status]
    if date_from:
        runs = [
            run
            for run in runs
            if run.get("start_time") and run["start_time"] >= date_from
        ]
    if date_to:
        runs = [
            run
            for run in runs
            if run.get("start_time") and run["start_time"] <= date_to
        ]
    if model:
        runs = [
            run for run in runs if await _run_matches_model_async(collector, run, model)
        ]
    if provider:
        runs = [
            run
            for run in runs
            if await _run_matches_provider_async(collector, run, provider)
        ]

    return runs


async def _run_matches_model_async(collector, run: dict[str, Any], model: str) -> bool:
    """Check if a run matches the specified model filter."""
    run_details = await collector.get_run_details(run["run_id"])
    if not run_details:
        return False

    for event in run_details.get("events", []):
        if event.get("type") == "llm_complete":
            result = event.get("data", {}).get("result")
            if isinstance(result, dict):
                model_name = result.get("model", "")
                if model.lower() in model_name.lower():
                    return True
    return False


async def _run_matches_provider_async(
    collector, run: dict[str, Any], provider: str
) -> bool:
    """Check if a run matches the specified provider filter."""
    run_details = await collector.get_run_details(run["run_id"])
    if not run_details:
        return False

    for event in run_details.get("events", []):
        if event.get("type") == "llm_complete":
            result = event.get("data", {}).get("result")
            if isinstance(result, dict):
                provider_name = result.get("provider", "")
                if provider.lower() in provider_name.lower():
                    return True
    return False
