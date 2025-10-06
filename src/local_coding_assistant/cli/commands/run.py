"""Run a single LLM or tool request."""

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
) -> None:
    """Execute a single query against the assistant."""
    typer.echo(f"Running query: {text}")
    if verbose:
        typer.echo("Verbose mode enabled")
    if model:
        typer.echo(f"Using model: {model}")

    ctx = bootstrap()
    assistant = Assistant(ctx)
    response = assistant.run_query(text, verbose)

    typer.echo(f"\nResponse:\n{response}")
