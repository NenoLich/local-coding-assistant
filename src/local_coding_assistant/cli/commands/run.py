"""Run a single LLM or tool request via the runtime orchestrator."""

import logging

import typer

from local_coding_assistant.core import bootstrap
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.utils.logging import get_logger

app = typer.Typer(name="run", help="Run a single LLM or tool request")
log = get_logger(__name__)


@app.command()
@safe_entrypoint("cli.run.query")
def query(
    text: str = typer.Argument(..., help="The query to run"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    model: str | None = typer.Option(None, help="Model to use for the query"),
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
    result = runtime.orchestrate(text, model=model)

    # Print the assistant message (preserves existing tests that check for LLM echo)
    typer.echo("\nResponse:")
    typer.echo(result["message"])
