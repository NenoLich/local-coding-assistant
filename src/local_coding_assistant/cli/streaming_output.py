"""
Streaming output handling for CLI.
"""

import yaml
from rich.console import Console
from rich.markup import escape

from local_coding_assistant.runtime.events import EventType, ExecutionEvent


def _handle_llm_chunk(event: ExecutionEvent, console: Console) -> None:
    # Print LLM content chunks incrementally
    content = event.data.get("content", "")
    if content:
        console.print(content, end="")
    # Print reasoning chunks in dim italic to differentiate
    reasoning = event.data.get("reasoning", "")
    if reasoning:
        console.print(reasoning, style="dim italic", end="")


def _handle_llm_complete(event: ExecutionEvent, console: Console) -> None:
    # LLM response complete
    console.print()  # New line


def _handle_tool_start(event: ExecutionEvent, console: Console) -> None:
    # Tool execution started
    tool_call = event.data.get("tool_call", {})
    tool_name = tool_call.name if tool_call else "unknown"
    tool_args = tool_call.arguments if tool_call else {}
    args_mention = "with arguments:" if tool_args else ""
    console.print(
        f"\n[bold blue]Executing tool: {tool_name} {args_mention}[/bold blue]"
    )
    if tool_args:
        yaml_output = yaml.dump(tool_args, default_flow_style=False)
        console.print(f"\n[bold blue]{escape(yaml_output)}[/bold blue]")


def _handle_tool_result(event: ExecutionEvent, console: Console) -> None:
    # Tool execution completed
    tool_call = event.data.get("tool_call", {})
    tool_name = tool_call.name if tool_call else "unknown"
    response = event.data.get("response", {})
    if response and response.success:
        result = response.result if response.result else None
        result_mention = "with result:" if result else ""
        console.print(
            f"[green]Tool {tool_name} completed successfully {result_mention}[/green]"
        )
        if result:
            console.print(f"\n[green]{result}[/green]")
    else:
        error = response.error_message if response else "Unknown error"
        console.print(f"[red]Tool {tool_name} failed: {error}[/red]")


def _handle_frame_start(event: ExecutionEvent, console: Console) -> None:
    # New frame/iteration started
    console.print(
        f"\n[bold cyan]Starting iteration {event.frame_id or 'unknown'}[/bold cyan]"
    )


def _handle_frame_complete(event: ExecutionEvent, console: Console) -> None:
    # Frame completed
    console.print(f"[dim]Iteration {event.frame_id or 'unknown'} completed[/dim]")


def _handle_error(event: ExecutionEvent, console: Console) -> None:
    # Error occurred
    error_msg = event.data.get("error", "Unknown error")
    console.print(f"[red]Error: {error_msg}[/red]")


def handle_streaming_event(event: ExecutionEvent, console: Console) -> None:
    """Handle a single streaming event for CLI output."""
    handlers = {
        EventType.LLM_CHUNK: _handle_llm_chunk,
        EventType.LLM_COMPLETE: _handle_llm_complete,
        EventType.TOOL_START: _handle_tool_start,
        EventType.TOOL_RESULT: _handle_tool_result,
        EventType.FRAME_START: _handle_frame_start,
        EventType.FRAME_COMPLETE: _handle_frame_complete,
        EventType.ERROR: _handle_error,
    }
    handler = handlers.get(event.type)
    if handler:
        handler(event, console)
    # TURN_START, TURN_COMPLETE, SESSION_START, SESSION_RESUME can be handled if needed for verbose output


def collect_report_from_events(events: list[ExecutionEvent]) -> dict | None:
    """Collect the final report from TURN_COMPLETE event."""
    for event in reversed(events):
        if event.type == EventType.TURN_COMPLETE:
            return event.data.get("report")
    return None
