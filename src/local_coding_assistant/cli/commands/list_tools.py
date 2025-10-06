"""List available tools."""

import typer

app = typer.Typer(name="list-tools", help="List available tools")


@app.command("list")
def list_available(
    category: str | None = typer.Option(None, help="Filter tools by category"),
    json: bool = typer.Option(False, help="Output as JSON"),
) -> None:
    """List all available tools."""
    if json:
        typer.echo('{"tools": []}')  # TODO: Implement actual tool listing
    else:
        typer.echo("Available tools:")
        if category:
            typer.echo(f"Category: {category}")
        # TODO: Implement formatted tool listing
