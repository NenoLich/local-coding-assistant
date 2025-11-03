"""List available tools."""

import json
import logging
from typing import Literal

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
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = typer.Option(
        "INFO", "--log-level", help="Logging level (e.g., DEBUG, INFO, WARNING)"
    ),
) -> None:
    """List all available tools."""
    # Convert log_level to string and validate it's one of the allowed values
    log_level_str = log_level.default if hasattr(log_level, "default") else log_level
    log_level_str = str(log_level_str).upper()

    # Ensure the log level is one of the allowed values
    allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level_str not in allowed_levels:
        log_level_str = "INFO"  # Default to INFO if invalid

    level = getattr(logging, log_level_str, logging.INFO)
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
