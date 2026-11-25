"""List sessions/turns stored in the durable execution store."""

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
from local_coding_assistant.storage.execution_store_reader import ExecutionStoreReader
from local_coding_assistant.utils.logging import get_logger, setup_logging

app = typer.Typer(name="sessions", help="List stored sessions and turns")
log = get_logger("cli.sessions")


@app.command("list")
@safe_entrypoint("cli.sessions.list")
def list_sessions(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum sessions to list"),
    status: str = typer.Option(
        None, "--status", help="Filter by status (idle|running)"
    ),
    store_db_path: str = typer.Option(
        "data/execution_store.db",
        "--store-db-path",
        help="Path to the execution store database",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """List recent sessions from the durable store."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    setup_logging(level=level)

    async def _main() -> int:
        db_path = Path(store_db_path)
        if not db_path.exists():
            typer.echo(f"Error: Execution store not found: {db_path}", err=True)
            return 1

        store = ExecutionStoreReader(db_path)
        await store.init()
        try:
            sessions = await store.list_sessions(limit=limit, status=status)
            if not sessions:
                typer.echo("No sessions found")
                return 0

            for session in sessions:
                turns = await store.list_turns_by_session(session.id)
                typer.echo(
                    f"{session.id}  status={session.status}  turns={len(turns)}  "
                    f"input_tokens={session.total_input_tokens}  "
                    f"output_tokens={session.total_output_tokens}  "
                    f"model={session.model_name or '-'}"
                )
                typer.echo(f"    input: {(session.user_input or '')[:100]}")
        finally:
            await store.close()

        return 0

    raise typer.Exit(code=asyncio.run(_main()))


@app.command("resume")
@safe_entrypoint("cli.sessions.resume")
def resume(
    session_id: str = typer.Argument(..., help="Session to resume"),
    text: str = typer.Argument(
        None,
        help=(
            "New user message (creates a new turn). Omit to continue the "
            "interrupted turn where it stopped (crash recovery)"
        ),
    ),
    turn: str = typer.Option(
        None, "--turn", help="Turn ID to resume, or to branch from"
    ),
    index: int = typer.Option(
        None, "--index", help="Turn index to resume, or to branch from"
    ),
    frame: str = typer.Option(
        None,
        "--frame",
        help="Frame ID within the turn (no-message resume only)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    model: str = typer.Option(None, "--model", "-m", help="Override model name"),
    agent_mode: str = typer.Option(None, "--agent-mode", help="Agent mode"),
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
    trace: str = typer.Option(None, "--trace", help="Dump JSON trace to file"),
    auto_approve: bool = typer.Option(
        None, "--auto-approve", help="Auto-approve all tool calls"
    ),
) -> None:
    """Resume a session where it stopped.

    With a TEXT argument, creates a new turn with that message. Without,
    re-executes the interrupted turn from where it paused - use this after
    a crash or Ctrl+C (`locca sessions list --status running` finds
    interrupted sessions).

    --turn/--index select any turn: the last one continues in place, a past
    one branches a new session at that turn. --frame picks a frame inside
    the turn (no-message resume).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    async def _main() -> int:
        async with bootstrap(
            log_level=level, target_project_root=target_project_root
        ) as ctx:
            console = Console()
            runtime = ctx.get("runtime")
            if runtime is None:
                typer.echo(
                    "Error: Runtime manager not available (LLM initialization failed)",
                    err=True,
                )
                return 1

            options = RunOptions(
                session_id=session_id,
                resume_turn_id=turn,
                resume_turn_index=index,
                resume_frame_id=frame,
                model=model,
                agent_mode=agent_mode,
                tool_call_mode=tool_call_mode,
                sandbox_session=sandbox_session,
                verbose=verbose,
                auto_approve=auto_approve,
            )

            report, session = await execute_query(runtime, text, options, console)

            if report is None:
                typer.echo("Error: No terminal report received from runtime", err=True)
                return 1

            forked = session.id != session_id
            if forked:
                # A past turn was selected: the runtime branched a new
                # session from it and execution continues there.
                typer.echo(f"branched from {session_id}")
            typer.echo(f"resumed session={session.id}")
            last_turn = session.get_last_turn()
            if last_turn is not None:
                typer.echo(f"turn={last_turn.turn_id} status={last_turn.status}")

            if report is not None and trace is not None:
                txpath = Path(trace)
                txpath.write_text(
                    json.dumps(
                        dump_report(report), indent=2, ensure_ascii=True, default=str
                    )
                )

            fmt = output_format.lower()
            if fmt == "plain":
                message = report.final_answer or ""
                typer.echo("\nResponse:")
                typer.echo(message)
            else:
                render_report(
                    coerce_report(report), _format=fmt, verbose=verbose, console=console
                )

        return 0

    raise typer.Exit(code=asyncio.run(_main()))
