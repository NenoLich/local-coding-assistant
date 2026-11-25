"""
Tool management commands for the CLI.

This module provides commands to list and run tools through the
tools_platform ToolGateway, which is the single tool execution path.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import typer
import yaml
from rich.console import Console
from rich.table import Table

from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.core.exceptions import CLIError
from local_coding_assistant.tools_platform.contracts import ToolCallRequest
from local_coding_assistant.tools_platform.gateway import ToolGateway
from local_coding_assistant.utils.logging import get_logger

app = typer.Typer(name="tool", help="Manage tools")
log = get_logger("cli.tool")
console = Console()


class ToolCLIError(CLIError):
    """Raised for recoverable CLI tool command errors."""


@dataclass
class ToolCLIContext:
    """Shared context for tool CLI commands."""

    level: int
    app_context: Any
    tool_gateway: ToolGateway

    @classmethod
    def create(
        cls,
        log_level: str,
        *,
        config_file: str | None = None,
        sandbox: bool = False,
    ) -> "ToolCLIContext":
        """Bootstrap the application and return a ready-to-use context.

        Args:
            log_level: Logging level as a string (e.g., "INFO", "DEBUG")
            config_file: Optional path to a configuration file
            sandbox: Whether to enable sandbox execution

        Returns:
            An initialized ToolCLIContext instance

        Raises:
            ToolCLIError: If the tool gateway cannot be initialized
        """
        level = _resolve_log_level(log_level)

        # Bootstrap the application
        ctx = bootstrap(config_path=config_file, log_level=level)

        config_manager = ctx.get("config")
        if config_manager is None:
            raise ToolCLIError("Config manager not available in context")

        # If sandbox is requested, set the session override
        if sandbox:
            config_manager.set_session_overrides({"sandbox.enabled": True})

        tool_gateway = ctx.get("tool_gateway")
        if tool_gateway is None:
            raise ToolCLIError("Tool gateway not available (initialization failed)")

        return cls(level=level, app_context=ctx, tool_gateway=tool_gateway)


# Type variable for enum validation
T = TypeVar("T", bound=Enum)


def _truncate_text(text: str, limit: int = 100) -> str:
    """Trim text to the provided limit with ellipsis."""

    text = text or ""
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _render_tool_table(rows: list[dict[str, Any]]) -> None:
    """Render tool information using a Rich table."""

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Risk")
    table.add_column("Backends")
    table.add_column("Description", overflow="fold")

    for row in rows:
        table.add_row(
            row["name"],
            row.get("version", "-"),
            row.get("risk_class", "-"),
            ", ".join(row.get("supported_backends", [])),
            _truncate_text(row.get("description", "")),
        )

    console.print(table)


def _resolve_log_level(log_level: str) -> int:
    """Resolve log level string into a logging constant."""
    return getattr(logging, str(log_level).upper(), logging.INFO)


def _collect_tool_rows(
    tool_gateway: ToolGateway, available_only: bool
) -> list[dict[str, Any]]:
    """Retrieve and normalize tool entries from the gateway registry.

    Args:
        tool_gateway: The tool gateway instance
        available_only: If True, only include exposed tools

    Returns:
        List of tool display dicts, sorted by tool name
    """
    if available_only:
        definitions = tool_gateway.get_exposed_tools()
    else:
        definitions = tool_gateway.get_exposed_tools()

    rows = []
    for definition in definitions:
        rows.append(
            {
                "name": definition.id,
                "version": definition.version,
                "risk_class": definition.risk_class.value
                if hasattr(definition.risk_class, "value")
                else str(definition.risk_class),
                "supported_backends": [
                    b.value if hasattr(b, "value") else str(b)
                    for b in definition.supported_backends
                ],
                "description": definition.description,
            }
        )

    return sorted(rows, key=lambda _row: _row["name"].lower())


async def _run_tool(
    tool_gateway: ToolGateway,
    tool_id: str,
    args: dict,
    session_id: str = "default",
) -> Any:
    """Execute a tool through the gateway and return the result.

    Args:
        tool_gateway: The tool gateway instance
        tool_id: ID of the tool to run
        args: Dictionary of arguments to pass to the tool
        session_id: Session ID for the tool call

    Returns:
        The tool's execution result

    Raises:
        ToolCLIError: If there's an error executing the tool
    """
    # Check tool exists and resolve its definition
    definitions = tool_gateway.get_exposed_tools()
    definition = next((d for d in definitions if d.id == tool_id), None)
    if definition is None:
        raise ToolCLIError(f"Tool '{tool_id}' not found or not exposed")

    call_args: dict[str, Any] = dict(args or {})
    positional: list[Any] = []
    if "input" in call_args:
        positional.append(call_args.pop("input"))
    wrapped_args = call_args.pop("args", None)
    if wrapped_args is not None:
        positional.extend(
            wrapped_args if isinstance(wrapped_args, list) else [wrapped_args]
        )
    if positional:
        properties = list(
            (getattr(definition, "input_schema", None) or {})
            .get("properties", {})
            .keys()
        )
        for index, value in enumerate(positional):
            if index < len(properties):
                call_args[properties[index]] = value
            else:
                call_args.setdefault("args", []).append(value)

    call_id = str(uuid.uuid4())
    request = ToolCallRequest(
        call_id=call_id,
        session_id=session_id,
        frame_id=session_id,
        tool_id=tool_id,
        raw_arguments=call_args or {},
        workspace_id=tool_gateway.workspace_manager.workspace_id,
    )

    typer.echo(f"Executing tool: {tool_id}")

    try:
        result = await tool_gateway.execute(request)
        if result.error:
            raise ToolCLIError(result.error.message)
        return result.output
    except Exception as e:
        raise ToolCLIError(f"Error executing tool: {e}") from e


@app.command(name="list")
@safe_entrypoint("cli.tool.list")
def list_tools(
    available_only: bool = typer.Option(
        False,
        "--available-only",
        help="Show only tools that are available for execution",
    ),
    config_file: str | None = typer.Option(
        None, "--config-file", help="Configuration file to read from"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    ),
):
    """List all available tools with their metadata."""

    try:
        context = ToolCLIContext.create(log_level, config_file=config_file)
        rows = _collect_tool_rows(context.tool_gateway, available_only)

        if not rows:
            suffix = " (matching the filter)" if available_only else ""
            typer.echo(f"No tools found{suffix}")
            return

        _render_tool_table(rows)

    except ToolCLIError as exc:
        log.error(str(exc))
        raise typer.Exit(1) from exc
    except Exception as exc:  # pragma: no cover - unforeseen runtime issues
        log.error(
            "Failed to list tools: %s",
            exc,
            exc_info=log.is_enabled_for(logging.DEBUG)
            if hasattr(log, "is_enabled_for")
            else True,
        )
        raise typer.Exit(1) from exc


@app.command()
@safe_entrypoint("cli.tool.run")
def run(
    tool_id: str = typer.Argument(..., help="ID of the tool to run"),
    args: list[str] = typer.Argument(  # noqa: B008
        None, help="Arguments to pass to the tool"
    ),
    sandbox: bool = typer.Option(
        False, "--sandbox", "-s", help="Execute tool in sandbox environment"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    ),
) -> None:
    """Execute a tool with the given arguments.

    Examples:
        # Run a tool directly
        locca tool run my_tool arg1=value1 arg2=value2

        # Run a tool in a sandbox
        locca tool run --sandbox my_tool arg1=value1 arg2=value2
    """
    # Create the context with sandbox enabled if requested
    ctx = ToolCLIContext.create(log_level, sandbox=sandbox)
    tool_gateway = ctx.tool_gateway

    try:
        # Parse the arguments for the single tool
        parsed_args = _parse_cli_args(args or [])

        # Run the tool through the gateway
        session_id = str(uuid.uuid4()) if sandbox else "default"
        result = asyncio.run(
            _run_tool(tool_gateway, tool_id, parsed_args, session_id=session_id)
        )

        # Print the result if there is one
        if result is not None:
            console.print("\n[bold]Tool execution result:[/bold]")
            # If the result is a string, print it directly, otherwise dump as YAML
            if isinstance(result, str):
                console.print(result)
            else:
                console.print(
                    yaml.dump(result, default_flow_style=False, sort_keys=False)
                )
        else:
            console.print(
                "\n[green]✓ Tool executed successfully but returned no result[/green]"
            )

    except ToolCLIError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except Exception as exc:  # pragma: no cover - unforeseen runtime issues
        log.error(
            "Failed to execute tool: %s",
            exc,
            exc_info=log.is_enabled_for(logging.DEBUG)
            if hasattr(log, "is_enabled_for")
            else True,
        )
        console.print(f"\n[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


def _parse_cli_args(args: list[str]) -> dict[str, Any]:
    """Convert raw CLI arguments into a structured payload.

    Handles simple JSON payload shorthands and key=value pairs.
    """
    if not args:
        return {}

    json_payload = _try_parse_json_blob(args[0])
    if json_payload is not None:
        return json_payload

    named: dict[str, list[Any]] = {}
    positional: list[Any] = []

    for token in args:
        key, raw_value = _split_assignment(token)
        if key is None:
            positional.append(_coerce_cli_value(raw_value))
            continue

        value = _coerce_cli_value(raw_value)
        named.setdefault(key, []).append(value)

    payload: dict[str, Any] = {
        key: _collapse_named_values(values) for key, values in named.items()
    }

    if positional:
        if len(positional) == 1 and not named:
            payload.setdefault("input", positional[0])
        else:
            payload.setdefault("args", positional)

    return payload


def _try_parse_json_blob(token: str) -> dict[str, Any] | None:
    """Return a full payload when the token contains a JSON object or array."""
    text = token.strip()
    if not text:
        return None

    if not (
        (text.startswith("{") and text.endswith("}"))
        or (text.startswith("[") and text.endswith("]"))
    ):
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"args": parsed}

    return None


def _split_assignment(token: str) -> tuple[str | None, str]:
    """Split ``key=value`` tokens while keeping positional arguments intact."""
    if "=" not in token:
        return None, token

    key, value = token.split("=", 1)
    key = key.strip()
    if not key:
        return None, token

    return key, value


def _coerce_cli_value(raw: str) -> Any:
    """Best-effort conversion from CLI string tokens to Python values."""
    text = raw.strip()
    if not text:
        return text

    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    ):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None

    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            continue

    return text


def _collapse_named_values(values: list[Any]) -> Any:
    """Fold repeated named values to match CLI expectations."""
    if not values:
        return None

    merged: list[Any] = []
    for value in values:
        if isinstance(value, list | tuple):
            merged.extend(value)
        else:
            merged.append(value)

    if len(merged) == 1 and not isinstance(merged[0], list | dict):
        return merged[0]

    return merged
