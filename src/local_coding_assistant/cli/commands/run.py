"""Run a single LLM or tool request via the runtime orchestrator."""

import asyncio
import logging

import typer

from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.utils.logging import get_logger

app = typer.Typer(name="run", help="Run a single LLM or tool request")
log = get_logger("cli.run")


@app.command()
@safe_entrypoint("cli.run.query")
def query(
    text: str = typer.Argument(..., help="The query to run"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    model: str | None = typer.Option(None, help="Model to use for the query"),
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
) -> None:
    """Execute a single query using `RuntimeManager` and print the result.

    This command configures logging, boots the app context, and delegates to
    the runtime to handle the session, LLM interaction, and tool orchestration.
    """
    typer.echo(f"Running query: {text}")
    if verbose:
        typer.echo("Verbose mode enabled")
    if model:
        typer.echo(f"Using model: {model}")

    # Map provided level string to logging.* constant (default INFO)
    level = getattr(logging, log_level.upper(), logging.INFO)

    ctx = bootstrap(log_level=level)
    runtime = ctx["runtime"]

    log.info("Executing query with model=%s", model or "default")

    # Check if runtime is available
    if runtime is None:
        typer.echo("Error: Runtime manager not available (LLM initialization failed)")
        raise typer.Exit(code=1)

    result = asyncio.run(
        runtime.orchestrate(
            text,
            model=model,
            tool_call_mode=tool_call_mode,
            sandbox_session=sandbox_session,
        )
    )

    # Print the assistant message (preserves existing tests that check for LLM echo)
    typer.echo("\nResponse:")
    typer.echo(result["message"])
