"""Configure system settings."""

import typer

app = typer.Typer(name="config", help="Configure system settings")


@app.command("get")
def get_config(
    key: str | None = typer.Argument(None, help="Configuration key to get"),
) -> None:
    """Get configuration value(s)."""
    if key:
        typer.echo(f"Getting config: {key}")
    else:
        typer.echo("All configuration:")


@app.command("set")
def set_config(
    key: str = typer.Argument(..., help="Configuration key to set"),
    value: str = typer.Argument(..., help="Value to set"),
) -> None:
    """Set a configuration value."""
    typer.echo(f"Setting {key} = {value}")
