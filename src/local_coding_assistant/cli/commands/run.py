"""Run a single LLM or tool request via the runtime.

Streams typed runtime events (LLM chunks, tool calls, frame boundaries) to
the console, handles in-process approval prompts, and prints the final
report in the requested format.
"""

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
from local_coding_assistant.cli.run_flow import RunOptions, execute_query
from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.utils.logging import get_logger

app = typer.Typer(name="run", help="Run a single LLM or tool request")
log = get_logger("cli.run")

_SUCCESS_STATUSES = ("completed", "success")


@app.command()
@safe_entrypoint("cli.run.query")
def query(
    text: str,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    model: str = typer.Option(None, "--model", "-m", help="Override model name"),
    agent_mode: str = typer.Option("no_agent", "--agent-mode", help="Agent mode"),
    tool_call_mode: str = typer.Option(
        None, "--tool-call-mode", help="Tool call mode (e.g. 'ptc' for sandbox)"
    ),
    sandbox_session: str = typer.Option(
        None, "--sandbox-session", help="Sandbox session name"
    ),
    target_project_root: str = typer.Option(
        None, "--target-project-root", "-t", help="Target project root directory"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    output_format: str = typer.Option(
        "plain", "--format", help="Output format: plain, yaml, json"
    ),
    session_id: str = typer.Option(
        None,
        "--session-id",
        help=(
            "Continue an existing durable session (creates a new turn). "
            "Use `locca sessions resume` to re-enter an interrupted turn "
            "or branch from a past one"
        ),
    ),
    trace: str = typer.Option(None, "--trace", help="Dump JSON trace to file"),
    auto_approve: bool = typer.Option(
        None, "--auto-approve", help="Auto-approve all tool calls"
    ),
) -> None:
    """Run a query with streaming runtime events."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    async def _main() -> int:
        console = Console()
        async with bootstrap(
            log_level=level, target_project_root=target_project_root
        ) as ctx:
            runtime = ctx.get("runtime")
            if runtime is None:
                typer.echo(
                    "Error: Runtime manager not available (LLM initialization failed)",
                    err=True,
                )
                return 1

            options = RunOptions(
                model=model,
                agent_mode=agent_mode,
                tool_call_mode=tool_call_mode,
                sandbox_session=sandbox_session,
                session_id=session_id,
                verbose=verbose,
                auto_approve=auto_approve,
            )
            report, session = await execute_query(runtime, text, options, console)

            if report is None:
                typer.echo("Error: No terminal report received from runtime", err=True)
                return 1

            if trace is not None:
                txpath = Path(trace)
                txpath.write_text(
                    json.dumps(
                        dump_report(report), indent=2, ensure_ascii=True, default=str
                    )
                )

            if verbose:
                typer.echo(f"session={session.id}")

            fmt = output_format.lower()
            if fmt == "plain":
                message = report.final_answer or ""
                typer.echo("\nResponse:")
                typer.echo(message)
                return 0 if report.status in _SUCCESS_STATUSES else 1

            render_report(
                coerce_report(report), _format=fmt, verbose=verbose, console=console
            )
        return 0 if report.status in _SUCCESS_STATUSES else 1

    raise typer.Exit(code=asyncio.run(_main()))
