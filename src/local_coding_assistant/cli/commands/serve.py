"""Start the assistant server."""

import typer

app = typer.Typer(name="serve", help="Start the assistant server")


@app.command()
def start(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Start the assistant server."""
    typer.echo(f"Starting server on {host}:{port}")
    if reload:
        typer.echo("Auto-reload enabled")
    # TODO: Implement server startup
