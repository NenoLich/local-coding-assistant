"""Streaming bridge between the runtime event pipeline and CLI front-ends.

Provides:
- ``QueueSubscriber``: forwards typed runtime events into an ``asyncio`` queue
- ``EventStreamRenderer``: renders events to a rich console and accumulates
  them for report building
- ``build_report``: converts accumulated events into a ``RunReport``
- ``prompt_approvals``: interactive in-process approval prompt
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

import yaml
from rich.console import Console

from local_coding_assistant.runtime.events import TurnStartEvent
from local_coding_assistant.runtime.events.contracts import (
    ApprovalRequestForm,
    ApprovalRequiredEvent,
    ApprovalResponseForm,
    ContinuationDecisionEvent,
    ErrorEvent,
    FrameCompleteEvent,
    FrameStartEvent,
    LLMChunkEvent,
    LLMCompleteEvent,
    LLMStartEvent,
    RuntimeEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnCompleteEvent,
)
from local_coding_assistant.runtime.events.reporting import RunError, RunReport
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("cli.runtime_stream")

_SENTINEL = object()

APPROVAL_YES = "YES_ONCE"
APPROVAL_NO = "NO"


class QueueSubscriber:
    """Event subscriber that forwards events into an ``asyncio`` queue."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    async def on_event(self, event: RuntimeEvent) -> None:
        """Forward a runtime event to the queue."""
        await self._queue.put(event)

    async def close(self) -> None:
        """Signal the end of the stream."""
        await self._queue.put(_SENTINEL)

    async def next_event(self) -> Any:
        """Await the next queued event (or the end-of-stream sentinel)."""
        return await self._queue.get()


class EventStreamRenderer:
    """Renders runtime events to a console and accumulates report data."""

    def __init__(self, console: Console, *, verbose: bool = False) -> None:
        self._console = console
        self._verbose = verbose
        self.events: list[RuntimeEvent] = []
        self.pending_approvals: list[ApprovalRequestForm] = []
        self._handlers: dict[type, Callable[[Any], None]] = {
            LLMChunkEvent: self._handle_chunk,
            LLMCompleteEvent: self._handle_llm_complete,
            LLMStartEvent: self._handle_llm_start,
            ToolStartEvent: self._handle_tool_start,
            ToolResultEvent: self._handle_tool_result,
            FrameStartEvent: self._handle_frame_start,
            FrameCompleteEvent: self._handle_frame_complete,
            ApprovalRequiredEvent: self._handle_approval_required,
            ErrorEvent: self._handle_error,
            ContinuationDecisionEvent: self._handle_continuation,
        }

    def handle(self, event: RuntimeEvent) -> None:
        """Render and accumulate a single event."""
        self.events.append(event)

        handler = self._handlers.get(type(event))
        if handler is not None:
            handler(event)

    def _handle_llm_complete(self, event: LLMCompleteEvent) -> None:
        self._console.print()

    def _handle_llm_start(self, event: LLMStartEvent) -> None:
        if self._verbose:
            self._console.print(
                f"[dim]llm request (model={event.model or 'default'})[/dim]"
            )

    def _handle_chunk(self, event: LLMChunkEvent) -> None:
        if event.content:
            self._console.print(event.content, end="")
        if event.reasoning:
            self._console.print(event.reasoning, style="dim italic", end="")

    def _handle_tool_start(self, event: ToolStartEvent) -> None:
        self._console.print(
            f"\n[bold blue]Executing tool: {event.tool_name}[/bold blue]"
        )
        if event.arguments:
            args = yaml.dump(event.arguments, default_flow_style=False, sort_keys=False)
            self._console.print(args.rstrip(), style="blue")

    def _handle_tool_result(self, event: ToolResultEvent) -> None:
        if event.success:
            result = event.output if event.output is not None else ""
            self._console.print(f"[green]Tool {event.tool_name} completed[/green]")
            if result:
                self._console.print(str(result)[:2000], style="green")
        else:
            message = (event.error or {}).get("message", "unknown error")
            self._console.print(f"[red]Tool {event.tool_name} failed: {message}[/red]")

    def _handle_frame_start(self, event: FrameStartEvent) -> None:
        if self._verbose:
            self._console.print(
                f"\n[bold cyan]frame {event.iteration} "
                f"(snapshot={event.context_snapshot_id})[/bold cyan]"
            )

    def _handle_frame_complete(self, event: FrameCompleteEvent) -> None:
        input_tokens = (
            f", {int(event.total_input_tokens or 0)} input tokens"
            if event.total_input_tokens
            else ""
        )
        output_tokens = (
            f", {int(event.total_output_tokens or 0)} output tokens"
            if event.total_output_tokens
            else ""
        )
        duration = f", {event.duration_ms}ms" if event.duration_ms else ""
        self._console.print(
            f"[dim]frame {event.status}{input_tokens}{output_tokens}{duration}[/dim]"
        )

    def _handle_approval_required(self, event: ApprovalRequiredEvent) -> None:
        self.pending_approvals.extend(event.approval_requests)
        for request in event.approval_requests:
            self._console.print(
                f"\n[bold yellow]Approval required:[/bold yellow] "
                f"{request.tool_name} {json.dumps(request.arguments)}"
            )

    def _handle_error(self, event: ErrorEvent) -> None:
        self._console.print(f"[red]Error: {event.error_message}[/red]")

    def _handle_continuation(self, event: ContinuationDecisionEvent) -> None:
        if self._verbose:
            self._console.print(
                f"[dim]continuation: {event.action} ({event.reason})[/dim]"
            )


def _merge_frame_complete(
    frames: list[dict[str, Any]],
    frames_by_id: dict[str | None, dict[str, Any]],
    event: FrameCompleteEvent,
) -> None:
    """Merge a frame-complete event into its matching frame dict."""
    frame_id = event.frame_id
    if frame_id is None:
        return
    frame = frames_by_id.get(frame_id)
    if frame is None:
        frame = {
            "frame_id": frame_id,
            "iteration": "?",
            "context_snapshot_id": None,
            "result": {},
        }
        frames.append(frame)
        frames_by_id[frame_id] = frame
    frame["result"] = {
        "status": event.status,
        "finish_reason": event.finish_reason,
        "total_input_tokens": event.total_input_tokens or None,
        "total_output_tokens": event.total_output_tokens or None,
        "total_latency_ms": event.duration_ms,
        "error_message": event.error_message,
        "llm_message": event.llm_message,
    }


def build_report(renderer: EventStreamRenderer) -> RunReport | None:  # noqa: C901
    """Build a ``RunReport`` from accumulated events.

    Args:
        renderer: Renderer that collected the turn's events

    Returns:
        RunReport, or None if no terminal event was collected
    """
    if not renderer.events:
        return None

    complete = next(
        (e for e in reversed(renderer.events) if isinstance(e, TurnCompleteEvent)),
        None,
    )
    if complete is None:
        return None

    tool_calls: list[dict[str, Any]] = []
    models_used: list[str] = []
    errors: list[RunError] = []
    frames: list[dict[str, Any]] = []
    frames_by_id: dict[str | None, dict[str, Any]] = {}
    # Track tool start times by call_id
    tool_start_times: dict[str, datetime] = {}
    user_message: str | None = None

    for event in renderer.events:
        if isinstance(event, FrameStartEvent):
            frame: dict[str, Any] = {
                "frame_id": event.frame_id,
                "iteration": event.iteration,
                "context_snapshot_id": event.context_snapshot_id,
                "started_at": event.timestamp.isoformat(),
                "result": {},
                "llm_metrics": None,
                "tool_calls": [],
                "tool_execution_data": [],
            }
            frames.append(frame)
            frames_by_id[event.frame_id] = frame
        elif isinstance(event, FrameCompleteEvent):
            _merge_frame_complete(frames, frames_by_id, event)
        elif isinstance(event, TurnStartEvent):
            user_message = event.user_input
        elif isinstance(event, LLMCompleteEvent):
            if event.model and event.model not in models_used:
                models_used.append(event.model)
            # Associate LLM metrics with the frame
            if event.frame_id and event.frame_id in frames_by_id:
                frame = frames_by_id[event.frame_id]
                llm_metrics = {
                    "prompt_tokens": event.prompt_tokens,
                    "completion_tokens": event.completion_tokens,
                    "reasoning_tokens": event.reasoning_tokens,
                    "input_tokens": event.input_tokens,
                    "output_tokens": event.output_tokens,
                    "cached_tokens": event.cached_tokens,
                    "cache_write_tokens": event.cache_write_tokens,
                    "latency_ms": event.latency_ms,
                    "model_name": event.model,
                    "provider": event.provider,
                }
                # Only set llm_metrics if at least one field has a non-None value
                if any(v is not None for v in llm_metrics.values()):
                    frame["llm_metrics"] = llm_metrics
                # Add tool_calls with full fields from LLM event
                if event.tool_calls:
                    frame["tool_calls"] = event.tool_calls
        elif isinstance(event, ToolStartEvent):
            # Record tool start time
            tool_start_times[event.tool_call_id] = event.timestamp
            # Add tool execution data entry to frame
            if event.frame_id and event.frame_id in frames_by_id:
                frame = frames_by_id[event.frame_id]
                tool_exec_entry = {
                    "tool_id": event.tool_name,
                    "call_id": event.tool_call_id,
                    "status": "running",
                    "started_at": event.timestamp.isoformat(),
                    "finished_at": None,
                    "output": None,
                    "error": None,
                }
                frame["tool_execution_data"].append(tool_exec_entry)
        elif isinstance(event, ToolResultEvent):
            entry: dict[str, Any] = {
                "tool_call_id": event.tool_call_id,
                "name": event.tool_name,
                "arguments": event.arguments or {},
                "success": event.success,
            }
            if event.success:
                entry["result"] = event.output
            else:
                entry["error"] = (event.error or {}).get("message", "unknown error")
                errors.append(
                    RunError(
                        message=str(entry["error"]),
                        kind="tool_error",
                    )
                )
            tool_calls.append(entry)
            # Update tool execution data in frame
            if event.frame_id and event.frame_id in frames_by_id:
                frame = frames_by_id[event.frame_id]
                for tool_exec in frame["tool_execution_data"]:
                    if tool_exec["call_id"] == event.tool_call_id:
                        tool_exec["status"] = "completed" if event.success else "failed"
                        tool_exec["finished_at"] = event.timestamp.isoformat()
                        tool_exec["output"] = event.output
                        if event.error:
                            tool_exec["error"] = event.error
                        break
        elif isinstance(event, ErrorEvent):
            errors.append(RunError(message=event.error_message, kind=event.error_type))

    return RunReport(
        session_id=complete.session_id,
        status=complete.status,
        final_answer=complete.final_answer,
        message=user_message,
        finish_reason=complete.finish_reason,
        models_used=models_used,
        input_tokens_used=complete.total_input_tokens,
        output_tokens_used=complete.total_output_tokens,
        iterations=complete.total_frames or len(frames) or None,
        frames=frames or None,
        tool_calls=None,
        errors=errors,
    )


async def pump_stream(
    subscriber: QueueSubscriber,
    renderer: EventStreamRenderer,
    on_event: Callable[[RuntimeEvent], None] | None = None,
) -> None:
    """Consume queued events until the stream is closed."""
    while True:
        event = await subscriber.next_event()
        if event is _SENTINEL:
            return
        renderer.handle(event)
        if on_event is not None:
            on_event(event)


async def prompt_approvals(
    requests: list[ApprovalRequestForm],
    *,
    input_func: Callable[[str], str] | None = None,
) -> list[ApprovalResponseForm]:
    """Ask the user to approve pending tool calls.

    Denies when stdin is not interactive (non-TTY), which keeps scripted
    runs safe.

    Args:
        requests: Pending approval requests
        input_func: Injectable prompt function (for testing)

    Returns:
        One response form per request
    """
    prompt = input_func or input
    responses: list[ApprovalResponseForm] = []

    for request in requests:
        # Build detailed prompt with rule info
        lines = [
            "[bold yellow]⚠ Approval Required[/bold yellow]",
            f"  Tool: {request.tool_name}",
            f"  Arguments: {request.arguments}",
        ]
        if request.rule_id:
            lines.append(f"  Rule: {request.rule_id}")
        if request.rule_description:
            lines.append(f"  Reason: {request.rule_description}")
        if request.proposed_override:
            lines.append(f"  If approved: {request.proposed_override}")
        lines.append("  Options:")
        lines.append("    [1] YES_ONCE          - Allow this call only")
        lines.append("    [2] YES_ALWAYS_SESSION - Allow this rule this session")
        lines.append("    [3] YES_ALWAYS_PROJECT - Allow this rule permanently")
        lines.append("    [4] NO                - Deny")

        console = Console()
        console.print("\n".join(lines))

        answer = await asyncio.to_thread(prompt, "  Choice [1-4]: ")
        choice = answer.strip()
        response_map = {
            "1": "YES_ONCE",
            "2": "YES_ALWAYS_SESSION",
            "3": "YES_ALWAYS_PROJECT",
            "4": "NO",
            "y": "YES_ONCE",
            "yes": "YES_ONCE",
            "n": "NO",
            "no": "NO",
        }
        response = response_map.get(choice.lower(), "NO")
        responses.append(
            ApprovalResponseForm(
                approval_token=request.approval_token,
                tool_call_id=request.tool_call_id,
                response=response,
            )
        )

    return responses
