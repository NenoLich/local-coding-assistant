"""Start the assistant server."""

import logging

import typer

from local_coding_assistant.core import bootstrap
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.utils.logging import get_logger

app = typer.Typer(name="serve", help="Start the assistant server")
log = get_logger(__name__)


@app.command()
@safe_entrypoint("cli.serve.start")
def start(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging level (e.g., DEBUG, INFO, WARNING)"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose error output"),
) -> None:
    """Start the assistant server (placeholder)."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    ctx = bootstrap(log_level=level)

    typer.echo(f"Starting server on {host}:{port}")
    if reload:
        typer.echo("Auto-reload enabled")

    # Placeholder server start; log that runtime exists
    runtime = ctx.get("runtime")
    log.info("Runtime component: %s", type(runtime).__name__ if runtime else "missing")
    # In a real app, this is where you'd start an ASGI/HTTP server
