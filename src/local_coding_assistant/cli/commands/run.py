"""Run a single LLM or tool request."""

import logging

import typer

from local_coding_assistant.core import bootstrap
from local_coding_assistant.core.assistant import Assistant

app = typer.Typer(name="run", help="Run a single LLM or tool request")


@app.command()
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
    """Execute a single query against the assistant."""
    typer.echo(f"Running query: {text}")
    if verbose:
        typer.echo("Verbose mode enabled")
    if model:
        typer.echo(f"Using model: {model}")

    # Map provided level string to logging.* constant (default INFO)
    level = getattr(logging, log_level.upper(), logging.INFO)

    ctx = bootstrap(log_level=level)
    assistant = Assistant(ctx)
    response = assistant.run_query(text, verbose)

    typer.echo(f"\nResponse:\n{response}")
