"""Run a single LLM or tool request via the runtime orchestrator."""

import asyncio
import json
import logging
from pathlib import Path

import typer
from rich.console import Console

from local_coding_assistant.cli.rendering import (
    coerce_report,
    dump_report,
    render_report,
)
from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.utils.logging import get_logger

app = typer.Typer(name="run", help="Run a single LLM or tool request")
log = get_logger("cli.run")


def _extract_plain_message(result) -> str:
    """Extract the plain text message from a result for CLI output.

    Args:
        result: The result from runtime.orchestrate(), either a dict or RunReport.

    Returns:
        Formatted message string with optional tool calls.
    """
    if isinstance(result, dict):
        message = result.get("message") or result.get("final_answer") or ""
        return message

    report = coerce_report(result)
    message = report.message or report.final_answer or ""

    if report.tool_calls:
        tool_sections = [message, "Tool Calls:"]
        tool_sections.extend(
            json.dumps(tool_call, indent=2) for tool_call in report.tool_calls
        )
        message = "\n\n".join(tool_sections)

    return message


@app.command()
@safe_entrypoint("cli.run.query")
def query(
    text: str = typer.Argument(..., help="The query to run"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    model: str | None = typer.Option(None, help="Model to use for the query"),
    agent_mode: str = typer.Option(
        "no_agent",
        "--agent-mode",
        help="Agent execution mode: 'default_loop' (legacy), 'graph' (LangGraph), 'frame' (ExecutionFrame), or 'no_agent'",
        case_sensitive=False,
    ),
    tool_call_mode: str = typer.Option(
        "reasoning",
        "--tool-call-mode",
        help="Tool calling mode: 'ptc' (Programmatic Tool Calling), 'classic' (standard tool calling), or 'reasoning' (default)",
        case_sensitive=False,
    ),
    sandbox_session: str | None = typer.Option(
        None, "--sandbox-session", help="Session ID for persistent state in sandbox"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    ),
    output_format: str = typer.Option(
        "plain",
        "--format",
        help="Output format: plain, json, frame, summary",
        case_sensitive=False,
    ),
    trace: str | None = typer.Option(
        None,
        "--trace",
        help="Write RunReport JSON trace to a file",
    ),
) -> None:
    """Execute a single query using `RuntimeManager` and print the result.

    This command configures logging, boots the app context, and delegates to
    the runtime to handle the session, LLM interaction, and tool orchestration.
    """
    format_lower = output_format.lower()

    typer.echo(f"Running query: {text}", err=True)

    typer.echo(f"Using agent mode: {agent_mode}", err=True)

    effective_log_level = log_level
    if format_lower != "plain" and not verbose and log_level.upper() == "INFO":
        effective_log_level = "WARNING"

    # Map provided level string to logging.* constant (default INFO)
    level = getattr(logging, effective_log_level.upper(), logging.INFO)

    ctx = bootstrap(log_level=level)
    runtime = ctx["runtime"]

    if format_lower == "plain" or verbose:
        log.info(
            "Executing query with model=%s, agent_mode=%s",
            model or "default",
            agent_mode or "default",
        )

    # Check if runtime is available
    if runtime is None:
        typer.echo("Error: Runtime manager not available (LLM initialization failed)")
        raise typer.Exit(code=1)

    result = asyncio.run(
        runtime.orchestrate(
            text,
            agent_mode=agent_mode,
            model=model,
            tool_call_mode=tool_call_mode,
            sandbox_session=sandbox_session,
        )
    )

    if trace is not None:
        trace_payload = dump_report(result)
        trace_path = Path(trace)
        trace_path.write_text(
            json.dumps(trace_payload, indent=2, ensure_ascii=True, default=str)
        )

    if format_lower == "plain":
        message = _extract_plain_message(result)
        typer.echo("\nResponse:")
        typer.echo(message)
        return

    console = Console()
    render_report(
        coerce_report(result), _format=format_lower, verbose=verbose, console=console
    )
