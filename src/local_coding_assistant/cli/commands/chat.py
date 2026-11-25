"""Interactive multi-turn chat session in a single process.

Each input creates a new turn in the same session (resident in memory),
reusing the same RuntimeManager entrypoints as ``locca run``. Tool
approvals pause the turn and are prompted inline.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

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

app = typer.Typer(name="chat", help="Interactive multi-turn chat session")
log = get_logger("cli.chat")

_EXIT_COMMANDS = {"/exit", "/quit", "/q"}
_SUCCESS_STATUSES = ("completed", "success")


@app.command()
@safe_entrypoint("cli.chat.start")
def start(
    session_id: str = typer.Option(
        None,
        "--session-id",
        help="Continue an existing durable session instead of creating one",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    model: str = typer.Option(None, "--model", "-m", help="Override model name"),
    agent_mode: str = typer.Option("no_agent", "--agent-mode", help="Agent mode"),
    tool_call_mode: str = typer.Option(
        None, "--tool-call-mode", help="Tool call mode (e.g. 'ptc' for sandbox)"
    ),
    target_project_root: str = typer.Option(
        None, "--target-project-root", "-t", help="Target project root directory"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    auto_approve: bool = typer.Option(
        None, "--auto-approve", help="Auto-approve all tool calls"
    ),
    trace: str = typer.Option(None, "--trace", help="Dump JSON trace to file"),
    output_format: str = typer.Option(
        "plain", "--format", help="Output format: plain, yaml, json"
    ),
) -> None:
    """Start an interactive chat session (one turn per input)."""
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

            typer.echo("LOCCA chat. Type /exit to finish.")
            current_session = session_id
            fmt = output_format.lower()
            turn = 0
            trace_entries: list[dict[str, Any]] = []

            while True:
                text = await asyncio.to_thread(
                    console.input, "[bold green]you>[/bold green] "
                )
                if not text.strip():
                    continue
                if text.strip().lower() in _EXIT_COMMANDS:
                    break

                options = RunOptions(
                    session_id=current_session,
                    model=model,
                    agent_mode=agent_mode,
                    tool_call_mode=tool_call_mode,
                    verbose=verbose,
                    auto_approve=auto_approve,
                )
                try:
                    report, session = await execute_query(
                        runtime, text, options, console
                    )
                except Exception as e:
                    log.error("Turn failed", error=str(e), exc_info=True)
                    typer.echo(f"[red]Turn failed: {e}[/red]", err=True)
                    continue

                current_session = session.id
                turn += 1
                typer.echo(f"[dim]session={session.id}[/dim]")

                if report is not None:
                    if trace is not None:
                        trace_entries.append(
                            {
                                "session_id": session.id,
                                "turn": turn,
                                "report": dump_report(report),
                            }
                        )
                        txpath = Path(trace)
                        txpath.write_text(
                            json.dumps(
                                trace_entries,
                                indent=2,
                                ensure_ascii=True,
                                default=str,
                            )
                        )

                    if fmt == "plain":
                        message = report.final_answer or ""
                        typer.echo("\nResponse:")
                        typer.echo(message)
                        return 0 if report.status in _SUCCESS_STATUSES else 1

                    render_report(
                        coerce_report(report),
                        _format=fmt,
                        verbose=verbose,
                        console=console,
                    )
        return 0

    raise typer.Exit(code=asyncio.run(_main()))
