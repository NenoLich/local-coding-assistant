"""List available tools."""

import json
import logging

import typer

from local_coding_assistant.core import bootstrap
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.utils.logging import get_logger

app = typer.Typer(name="list-tools", help="List available tools")
log = get_logger("cli.list_tools")


def _serialize_tool(tool: object) -> dict[str, str]:
    name = getattr(tool, "name", None)
    if name is None:
        name = getattr(tool, "__name__", None) or tool.__class__.__name__
    return {"name": str(name), "type": tool.__class__.__name__}


@app.command("list")
@safe_entrypoint("cli.list_tools")
def list_available(
    category: str | None = typer.Option(
        "All", "--cat", help="Filter tools by category (unused)"
    ),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging level (e.g., DEBUG, INFO, WARNING)"
    ),
) -> None:
    """List all available tools."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    ctx = bootstrap(log_level=level)

    tools = ctx.get("tools")
    if tools is None:
        log.error("Tool registry missing from context")
        typer.echo("No tools available")
        return

    items = list(tools)

    if json_out:
        typer.echo(
            json.dumps(
                {"count": len(items), "tools": [_serialize_tool(t) for t in items]}
            )
        )
        return

    typer.echo("Available tools:")
    if category:
        typer.echo(f"Category filter requested: {category} (not implemented)")
    if not items:
        typer.echo("- (none)")
    else:
        for t in items:
            meta = _serialize_tool(t)
            typer.echo(f"- {meta['name']} ({meta['type']})")
