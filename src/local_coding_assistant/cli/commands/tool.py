"""
Tool management commands for the CLI.

This module provides commands to manage tools through configuration files
and the bootstrap system, following the same patterns as other CLI commands.
"""

import ast
import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import typer
import yaml
from rich.console import Console
from rich.table import Table

from local_coding_assistant.config import EnvManager
from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.core.exceptions import CLIError
from local_coding_assistant.tools.types import (
    ToolCategory,
    ToolPermission,
    ToolTag,
)
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
    tool_manager: Any

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
            ToolCLIError: If the tool manager cannot be initialized
        """
        level = _resolve_log_level(log_level)

        # Bootstrap the application
        ctx = bootstrap(config_path=config_file, log_level=level)
        tool_manager = ctx.get("tools")
        if tool_manager is None:
            raise ToolCLIError("Tool manager not available (initialization failed)")

        config_manager = ctx.get("config")
        # If sandbox is requested, set the session override
        if sandbox:
            if config_manager is None:
                raise ToolCLIError("Config manager not available in context")

            # Set the sandbox enabled flag using session overrides
            config_manager.set_session_overrides({"sandbox.enabled": True})

        return cls(level=level, app_context=ctx, tool_manager=tool_manager)


# Type variable for enum validation
T = TypeVar("T", bound=Enum)


def get_enum_values(enum_class: type[Enum]) -> list[Any]:
    """Get all possible values for an enum class."""
    return [e.value for e in enum_class]


# Module-level constants for help text
TOOL_CATEGORIES = ", ".join(get_enum_values(ToolCategory))
TOOL_PERMISSIONS = ", ".join(get_enum_values(ToolPermission))
TOOL_TAGS = ", ".join(get_enum_values(ToolTag))


@dataclass
class ToolDisplayRow:
    """Renderable representation of a tool entry."""

    name: str
    category: str
    enabled: bool
    available: bool
    source: str
    description: str


def _truncate_text(text: str, limit: int = 100) -> str:
    """Trim text to the provided limit with ellipsis."""

    text = text or ""
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _normalize_tool_info(tool_info: Any) -> ToolDisplayRow:
    """Convert ToolInfo-like objects into display rows."""

    category = getattr(tool_info, "category", "")
    if hasattr(category, "value"):
        category = category.value

    source = getattr(tool_info, "source", "external")
    if hasattr(source, "value"):
        source = source.value

    return ToolDisplayRow(
        name=str(getattr(tool_info, "name", "")),
        category=str(category or ""),
        enabled=bool(getattr(tool_info, "enabled", True)),
        available=bool(getattr(tool_info, "available", True)),
        source=str(source or "external"),
        description=str(getattr(tool_info, "description", "")),
    )


def _collect_tool_rows(tool_manager: Any, available_only: bool) -> list[ToolDisplayRow]:
    """Retrieve and normalize tool entries from the manager.

    Args:
        tool_manager: The tool manager instance
        available_only: If True, only include tools that are available for execution

    Returns:
        List of tool display rows, sorted by tool name
    """
    rows = []
    for info in tool_manager.list_tools(available_only=available_only):
        row = _normalize_tool_info(info)
        rows.append(row)

    return sorted(rows, key=lambda _row: _row.name.lower())


def _render_tool_table(rows: list[ToolDisplayRow]) -> None:
    """Render tool information using a Rich table."""

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Category")
    table.add_column("Enabled")
    table.add_column("Available")
    table.add_column("Source")
    table.add_column("Description", overflow="fold")

    for row in rows:
        table.add_row(
            row.name,
            row.category or "-",
            "✓" if row.enabled else "✗",
            "✓" if row.available else "✗",
            row.source or "-",
            _truncate_text(row.description),
        )

    console.print(table)


def _validate_mcp_tool_config(
    endpoint: str | None, provider: str | None, module: str | None, path: str | None
) -> None:
    """Validate MCP tool configuration."""
    if not endpoint or not provider:
        raise ToolCLIError("Both --endpoint and --provider are required for MCP tools")
    if module or path:
        raise ToolCLIError("Cannot mix --module/--path with --endpoint/--provider")


def _validate_regular_tool_config(
    module: str | None, path: str | None, tool_class: str | None
) -> None:
    """Validate regular tool configuration."""
    if not (module or path):
        raise ToolCLIError("Either --module or --path is required for regular tools")
    if module and path:
        raise ToolCLIError("Cannot provide both --module and --path")
    if not tool_class:
        raise ToolCLIError("--tool-class is required for regular tools")


def _build_tool_config(
    *,
    tool_id: str,
    name: str | None,
    description: str,
    enabled: bool,
    category: str,
    module: str | None,
    path: str | None,
    endpoint: str | None,
    provider: str | None,
    permissions: list[str] | None,
    tags: list[str] | None,
    tool_class: str | None = None,
) -> dict[str, Any]:
    """Construct a validated tool configuration payload."""
    is_mcp = endpoint is not None or provider is not None

    # Validate configuration based on tool type
    if is_mcp:
        _validate_mcp_tool_config(endpoint, provider, module, path)
    else:
        _validate_regular_tool_config(module, path, tool_class)

    # Build base configuration
    tool_config: dict[str, Any] = {
        "id": tool_id,
        "name": name or tool_id,
        "description": description,
        "enabled": enabled,
        "source": "mcp" if is_mcp else "external",
        "category": category,
    }

    # Add optional fields if provided
    if permissions:
        tool_config["permissions"] = permissions
    if tags:
        tool_config["tags"] = tags

    # Add type-specific configuration
    if is_mcp:
        tool_config.update({"endpoint": endpoint, "provider": provider})
    else:
        if module:
            tool_config["module"] = module
        elif path:
            tool_config["path"] = path
        if tool_class:
            tool_config["tool_class"] = tool_class

    # Validate the complete configuration
    is_valid, errors = validate_tool_config(tool_config)
    if not is_valid:
        raise ToolCLIError("; ".join(errors))

    return tool_config


def _persist_tool_config(
    tool_config: dict[str, Any], *, config_file: str | None
) -> str:
    """Write tool configuration updates to disk and return applied action."""

    config_path = _get_config_path(config_file)
    _, action = _save_config(
        config_path=config_path,
        config={"tools": [tool_config]},
        tool_id=tool_config["id"],
    )
    return action


def _verify_tool_loaded(
    tool_manager: Any, tool_id: str, tool_name: str, *, action: str
) -> None:
    """Check that a tool is available after configuration updates."""

    try:
        tool_manager.get_tool(tool_id)
        typer.echo(
            f"✅ Successfully {action.lower()} tool '{tool_name}' with ID '{tool_id}'"
        )
    except Exception as exc:
        error_msg = str(exc)
        typer.echo(
            (
                "⚠️  Warning: Tool "
                f"'{tool_name}' was {action.lower()} but could not be reloaded: {error_msg}"
            ),
            err=True,
        )


def _parse_tool_args(args: list[str]) -> dict[str, Any]:
    """Convert raw CLI arguments into a structured payload.

    Responsibilities kept on the CLI side:
    - Understand simple JSON payload shorthands (a single dict/list argument)
    - Split ``key=value`` pairs from positional arguments
    - Coerce scalar values to basic Python types (numbers, booleans, lists/dicts)
    - Preserve repeated named parameters without collapsing positional context

    Schema-aware shaping of the payload is delegated to
    :class:`ToolInputTransformer`, which runs inside the tool manager.
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

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
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


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {seconds:.1f}s"


def _format_timestamp(timestamp: datetime | None) -> str:
    """Format a timestamp for display."""
    if not timestamp:
        return "Never"
    now = datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    delta = now - timestamp

    if delta < timedelta(minutes=1):
        return "Just now"
    if delta < timedelta(hours=1):
        mins = int(delta.total_seconds() / 60)
        return f"{mins}m ago"
    if delta < timedelta(days=1):
        hours = int(delta.total_seconds() / 3600)
        return f"{hours}h ago"
    return timestamp.strftime("%Y-%m-%d %H:%M")


def _display_tool_stats(tool_manager: Any, tool_name: str, _console: Console) -> None:
    """Display statistics for a specific tool.

    Args:
        tool_manager: The tool manager instance
        tool_name: Name of the tool to display stats for
        _console: Rich console instance for output
    """
    try:
        stats = tool_manager.get_execution_stats(tool_name)

        if not stats:
            _console.print(
                f"[yellow]No statistics available for tool: {tool_name}[/yellow]"
            )
            return

        _console.print("\n[bold]Execution Statistics:[/bold]")
        _console.print("-" * 30)

        # Basic stats
        _console.print(f"[bold]Tool:[/bold] {tool_name}")
        _console.print(
            f"[bold]Total Executions:[/bold] {stats.get('total_executions', 0)}"
        )
        _console.print(f"[bold]Success Rate:[/bold] {stats.get('success_rate', 0):.1%}")
        _console.print(
            f"[bold]Average Duration:[/bold] {_format_duration(stats.get('avg_duration', 0))}"
        )
        _console.print(
            f"[bold]First Execution:[/bold] {_format_timestamp(stats.get('first_execution'))}"
        )
        _console.print(
            f"[bold]Last Execution:[/bold] {_format_timestamp(stats.get('last_execution'))}"
        )

        # Metrics summary if available
        metrics = stats.get("metrics_summary", {})
        if metrics:
            _console.print("\n[bold]Metrics:[/bold]")
            for name, value in metrics.items():
                _console.print(f"  {name}: {value}")

    except Exception as e:
        _console.print(f"[yellow]Failed to get statistics: {e}[/yellow]")


def _display_system_stats(tool_manager: Any, _console: Console) -> None:
    """Display system-wide statistics.

    Args:
        tool_manager: The tool manager instance
        _console: Rich console instance for output
    """
    try:
        stats = tool_manager.get_system_stats()
        if not stats:
            _console.print("[yellow]No system statistics available[/yellow]")
            return

        _console.print("\n[bold]System Statistics:[/bold]")
        _console.print("-" * 30)

        _console.print(
            f"[bold]Total Executions:[/bold] {stats.get('total_executions', 0)}"
        )
        _console.print(
            f"[bold]Total Duration:[/bold] {_format_duration(stats.get('total_duration', 0))}"
        )
        _console.print(
            f"[bold]Average Duration:[/bold] {_format_duration(stats.get('avg_duration', 0))}"
        )
        _console.print(
            f"[bold]First Execution:[/bold] {_format_timestamp(stats.get('first_execution'))}"
        )
        _console.print(
            f"[bold]Last Execution:[/bold] {_format_timestamp(stats.get('last_execution'))}"
        )

        # Metrics summary if available
        metrics = stats.get("metrics_summary", {})
        if metrics:
            _console.print("\n[bold]System Metrics:[/bold]")
            for name, value in metrics.items():
                _console.print(f"  {name}: {value}")

    except Exception as e:
        _console.print(f"[yellow]Failed to get system statistics: {e}[/yellow]")


def parse_kv_pairs(s: str) -> dict[str, Any]:
    """Parse key=value pairs, handling JSON values properly."""
    result = {}
    # This regex matches key=value where value can be JSON
    pattern = r'([a-zA-Z0-9_]+)=({.*?}|\[.*?\]|"[^"]*"|[^,]+)(?=,|$)'
    for match in re.finditer(pattern, s):
        key = match.group(1)
        val = match.group(2).strip()
        # Try to parse as JSON first
        try:
            result[key] = json.loads(val)
        except json.JSONDecodeError:
            # If not JSON, handle as string (removing quotes if present)
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            result[key] = val
    return result


def _parse_tool_specs(specs: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse tool specifications from a string.

    Args:
        specs: String in format "tool1:arg1=val1,arg2=val2;tool2:arg1=val1"

    Returns:
        List of tuples (tool_id, args_list)

    Raises:
        ToolCLIError: If the specs format is invalid
    """
    tools = []
    if not specs.strip():
        return tools

    for tool_spec in specs.split(";"):
        tool_spec = tool_spec.strip()
        if not tool_spec:
            continue

        # Split tool name and args
        if ":" in tool_spec:
            parts = tool_spec.split(":", 1)
            if len(parts) != 2 or not parts[0].strip():
                raise ValueError(
                    "Invalid tool specification format. Expected 'tool:arg1=val1,arg2=val2'"
                )

            tool_name, args_str = parts
            tool_name = tool_name.strip()

            # Validate that args_str is not empty or contains invalid format
            if args_str.strip() and not any(c in args_str for c in "=, "):
                raise ValueError(
                    "Invalid argument format. Expected 'arg1=val1,arg2=val2'"
                )

            args = parse_kv_pairs(args_str)
            tools.append((tool_name, args))
        else:
            # Tool with no arguments
            tools.append((tool_spec, {}))

    return tools


async def _run_tools(
    tool_manager: Any,
    tool_specs: list[tuple[str, dict]],
    sandbox: bool = False,
    parallel: bool = False,
) -> list[tuple[str, bool, str]]:
    """Run multiple tools and return their results.

    Args:
        tool_manager: The tool manager instance
        tool_specs: List of (tool_id, args_dict) tuples where args_dict is a dictionary
                   of argument names to values
        sandbox: Whether to run tools in a sandbox
        parallel: Whether to run tools in parallel

    Returns:
        List of (tool_id, success, result) tuples
    """
    results = []

    if parallel:

        async def run_single(
            tool_id: str, args: dict, session_id: str
        ) -> tuple[str, bool, str]:
            try:
                _result = await _run_tool(
                    tool_manager, tool_id, args, sandbox, session_id=session_id
                )
                return tool_id, True, str(_result)
            except Exception as err:
                return tool_id, False, str(err)

        # Generate a unique session ID for each tool execution in parallel mode
        session_ids = [str(uuid.uuid4()) for _ in tool_specs]

        # Run all tools in parallel with their own session IDs
        tasks = [
            run_single(tool_id, args, session_id)
            for (tool_id, args), session_id in zip(tool_specs, session_ids, strict=True)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    else:
        # For sequential execution, use a single session ID for all tools
        session_id = str(uuid.uuid4()) if sandbox else ""

        # Run tools sequentially with the same session ID
        for tool_id, args in tool_specs:
            try:
                result = await _run_tool(
                    tool_manager, tool_id, args or {}, sandbox, session_id=session_id
                )
                results.append((tool_id, True, str(result)))
            except Exception as e:
                results.append((tool_id, False, str(e)))

    return results


async def _run_tool(
    tool_manager: Any,
    tool_id: str,
    args: dict,
    sandbox: bool = False,
    session_id: str = "default",
) -> Any:
    """Execute a tool and return the result.

    Args:
        tool_manager: The tool manager instance
        tool_id: ID of the tool to run
        args: Dictionary of arguments to pass to the tool
        sandbox: Whether to run the tool in a sandbox
        session_id: Optional session ID for sandbox execution. If None, a default will be used.
                   In parallel execution, a unique session ID should be provided for each tool.

    Returns:
        The tool's execution result

    Raises:
        ToolCLIError: If there's an error executing the tool
    """
    try:
        tool = tool_manager.get_tool(tool_id)
    except Exception as exc:
        raise ToolCLIError(str(exc)) from exc

    # Args are already parsed, use them directly
    payload = args or {}

    # Get tool info for display
    tool_info = None
    if hasattr(tool_manager, "get_tool_info"):
        try:
            tool_info = tool_manager.get_tool_info(tool_id)
        except Exception:  # pragma: no cover - optional hook
            tool_info = None

    display_name = (
        getattr(tool_info, "name", None) or getattr(tool, "name", None) or tool_id
    )

    typer.echo(f"Executing tool: {display_name}")
    if sandbox:
        typer.echo(f"Running in sandbox mode (Session: {session_id or 'default'})")

    try:
        if sandbox:
            # Execute in sandbox
            if not hasattr(tool_manager, "execute_tool_in_sandbox"):
                raise ToolCLIError("Sandbox execution is not available")
            # Pass the session_id to the sandbox execution
            return await tool_manager.execute_tool_in_sandbox(
                tool_id,
                payload,
                session_id=session_id,  # Pass the session_id through
            )
        else:
            # Execute directly
            result = await tool_manager.arun_tool(tool_id, payload)
            if result.get("result"):
                return result.get("result")
            else:
                return result

    except Exception as e:
        raise ToolCLIError(f"Error executing tool: {e}") from e


def _remove_tool_from_config(tool_id: str, *, config_file: str | None) -> Path:
    """Remove tool entry from configuration file."""

    config_path = _get_config_path(config_file)

    try:
        config = _load_config(config_path)
    except FileNotFoundError as exc:
        raise ToolCLIError(str(exc)) from exc

    tools = config.get("tools", [])
    remaining = [tool for tool in tools if tool.get("id") != tool_id]

    if len(remaining) == len(tools):
        raise ToolCLIError(f"Tool with ID '{tool_id}' not found in configuration")

    config["tools"] = remaining
    _save_config(config_path, config)
    return config_path


def validate_enum_value[T: Enum](value: str, enum_class: type[T], field_name: str) -> T:
    """Validate a single enum value.

    Args:
        value: The value to validate
        enum_class: The enum class to validate against
        field_name: Name of the field for error messages

    Returns:
        The validated enum value of type T

    Raises:
        ValueError: If the value is not a valid enum value
    """
    try:
        return enum_class(value)
    except ValueError as err:
        valid_values = get_enum_values(enum_class)
        raise ValueError(
            f"Invalid {field_name} '{value}'. Valid values are:\n"
            f"  {', '.join(valid_values)}"
        ) from err


def validate_enum_list[T: Enum](
    values: list[str] | None, enum_class: type[T], field_name: str
) -> list[T]:
    """Validate a list of enum values.

    Args:
        values: List of values to validate
        enum_class: The enum class to validate against
        field_name: Name of the field for error messages

    Returns:
        List of validated enum values

    Raises:
        ValueError: If any value is not a valid enum value
    """
    if not values:
        return []

    validated = []
    for v in values:
        try:
            validated.append(validate_enum_value(v, enum_class, field_name))
        except ValueError as e:
            # Add more context to the error message
            raise ValueError(f"Error in {field_name} '{v}': {e!s}") from e
    return validated


def validate_tool_field(
    config: dict,
    field_name: str,
    enum_class: type[Enum] | None = None,
    required: bool = False,
    default: Any = None,
    is_list: bool = False,
) -> tuple[Any, list[str]]:
    """Validate a single tool configuration field.

    Args:
        config: The tool configuration dictionary
        field_name: Name of the field to validate
        enum_class: Optional enum class for validation
        required: Whether the field is required
        default: Default value if field is not present
        is_list: Whether the field should be a list

    Returns:
        Tuple of (validated_value, error_messages)
    """
    errors = []
    value = config.get(field_name, default)

    # Check required fields
    if required and value is None:
        errors.append(f"Missing required field: {field_name}")
        return None, errors

    # Skip further validation if value is None
    if value is None:
        return None, errors

    # Validate list type if needed
    if is_list and not isinstance(value, list):
        errors.append(f"{field_name} must be a list, got {type(value).__name__}")
        return None, errors

    # Validate enum values
    if enum_class:
        try:
            if is_list:
                value = validate_enum_list(value, enum_class, field_name)
            else:
                value = validate_enum_value(value, enum_class, field_name)
        except ValueError as e:
            errors.append(str(e))

    return value, errors


def _extract_value(param_value: Any, default: Any = None) -> Any:
    """Extract the actual value, handling typer objects."""
    if hasattr(param_value, "default"):
        return param_value.default
    return param_value if param_value is not None else default


def _get_config_path(
    config_file: str | None = None, env_manager: EnvManager | None = None
) -> Path:
    """Get the configuration file path using PathManager.

    Args:
        config_file: Custom config file path if provided
        env_manager: Optional EnvManager instance (will create one if not provided)

    Returns:
        Path to the tools configuration file
    """
    if config_file:
        return Path(config_file)

    # Create or use provided env_manager
    env_manager = env_manager or EnvManager.create(load_env=True)

    # Use PathManager to resolve the config path based on environment
    return env_manager.path_manager.resolve_path("@config/tools.local.yaml")


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load and parse the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the parsed configuration

    Raises:
        yaml.YAMLError: If there's an error parsing the YAML file
        FileNotFoundError: If the config file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            log.error(f"Error parsing YAML file {config_path}: {e}")
            raise


def _update_tool_in_config(
    existing_config: dict, new_config: dict, tool_id: str
) -> tuple[dict, str]:
    """Update an existing tool configuration or add a new one.

    Args:
        existing_config: The current configuration
        new_config: New configuration to merge
        tool_id: ID of the tool to update

    Returns:
        Tuple of (updated_config, action) where action is "Added" or "Updated"
    """
    try:
        # Check if tool exists and update it
        for tool in existing_config["tools"]:
            if tool.get("id") == tool_id:
                if new_config["tools"]:  # Only update if we have new config
                    tool.update(new_config["tools"][0])
                return existing_config, "Updated"

        # If tool didn't exist, add it if we have new config
        if new_config["tools"]:
            existing_config["tools"].append(new_config["tools"][0])

        return existing_config, "Added"
    except Exception as e:
        log.warning(f"Could not update existing config: {e}")
        return new_config, "Added"


def _save_config_to_file(config_path: Path, config: dict) -> None:
    """Save configuration to file with proper error handling."""
    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    except yaml.YAMLError as e:
        log.error(f"YAML serialization error: {e}")
        raise
    except Exception as e:
        log.error(f"Failed to write config to {config_path}: {e}")
        raise


def _save_config(
    config_path: Path,
    config: dict | None = None,
    tool_id: str | None = None,
) -> tuple[dict, str]:
    """Save tool configuration to file.

    Args:
        config_path: Path to the config file
        config: Tool configuration to save
        tool_id: Optional ID of the tool being saved

    Returns:
        Tuple of (updated_config, action) where action is "Added" or "Updated"
    """
    try:
        # Ensure config directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle empty config case
        if config is None:
            config = {"tools": []}
            _save_config_to_file(config_path, config)
            return config, "Initialized"

        # Ensure tools key exists
        if "tools" not in config:
            config["tools"] = []

        # If no tool_id, just save the config as-is
        if not tool_id:
            _save_config_to_file(config_path, config)
            return config, "Saved"

        # Handle tool update case
        try:
            existing_config = _load_config(config_path)
            if "tools" not in existing_config:
                existing_config["tools"] = []

            updated_config, action = _update_tool_in_config(
                existing_config, config, tool_id
            )
            _save_config_to_file(config_path, updated_config)
            return updated_config, action

        except FileNotFoundError:
            # No existing config, save as new
            _save_config_to_file(config_path, config)
            return config, "Added"

    except Exception as e:
        error_msg = f"❌ Failed to save configuration to {config_path}: {e!s}"
        log.error(error_msg)
        raise typer.Exit(code=1) from e


def validate_tool_config(tool_config: dict) -> tuple[bool, list[str]]:
    """Validate a tool configuration dictionary.

    Args:
        tool_config: The tool configuration to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Validate required fields
    if "id" not in tool_config:
        errors.append("Missing required field: id")
    if "description" not in tool_config:
        errors.append("Missing required field: description")

    # Validate tool_class for non-MCP tools
    if tool_config.get("source") != "mcp" and "tool_class" not in tool_config:
        errors.append("Missing required field: tool_class")

    # Validate enum fields
    _category, category_errors = validate_tool_field(
        tool_config, "category", ToolCategory, required=False
    )
    errors.extend(category_errors)

    _permissions, perm_errors = validate_tool_field(
        tool_config, "permissions", ToolPermission, is_list=True
    )
    errors.extend(perm_errors)

    _tags, tag_errors = validate_tool_field(tool_config, "tags", ToolTag, is_list=True)
    errors.extend(tag_errors)

    return len(errors) == 0, errors


def validate_configuration_file(config_path: Path) -> tuple[bool, list[str]]:
    """Validate a tool configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Tuple of (is_valid, error_messages)
    """
    if not config_path.exists():
        return False, [f"Configuration file not found: {config_path}"]

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        return False, [f"Invalid YAML: {e!s}"]

    if not isinstance(config, dict):
        return False, ["Configuration must be a dictionary"]

    if "tools" not in config:
        return False, ["Missing 'tools' key in configuration"]

    if not isinstance(config["tools"], list):
        return False, ["'tools' must be a list of tool configurations"]

    all_errors = []
    all_valid = True

    for i, tool_config in enumerate(config["tools"], 1):
        if not isinstance(tool_config, dict):
            all_errors.append(
                f"Tool at index {i}: Expected dictionary, got {type(tool_config).__name__}"
            )
            all_valid = False
            continue

        tool_id = tool_config.get("id", f"at index {i}")
        is_valid, errors = validate_tool_config(tool_config)

        if not is_valid:
            all_errors.append(f"Tool '{tool_id}' has validation errors:")
            all_errors.extend(f"  {e}" for e in errors)
            all_valid = False

    return all_valid, all_errors


def _resolve_log_level(log_level: str) -> int:
    """Resolve log level string into a logging constant."""

    actual_level = _extract_value(log_level, "INFO")
    return getattr(logging, str(actual_level).upper(), logging.INFO)


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
        rows = _collect_tool_rows(context.tool_manager, available_only)

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
            "Failed to list tools: %s", exc, exc_info=log.isEnabledFor(logging.DEBUG)
        )
        raise typer.Exit(1) from exc


@app.command("list-options")
@safe_entrypoint("cli.tool.list_options")
def list_options() -> None:
    """List all available options for tool configuration."""

    def format_values(enum_class: type[Enum]) -> str:
        return "\n  " + "\n  ".join(f"- {v}" for v in get_enum_values(enum_class))

    console.print("\n[bold]Available Tool Configuration Options:[/bold]\n")

    console.print("[bold]Categories:[/bold]")
    console.print(format_values(ToolCategory))

    console.print("\n[bold]Permissions:[/bold]")
    console.print(format_values(ToolPermission))

    console.print("\n[bold]Tags:[/bold]")
    console.print(format_values(ToolTag))


@app.command()
@safe_entrypoint("cli.tool.add")
def add(
    tool_id: str = typer.Argument(..., help="Unique ID for the tool"),
    # Common options
    name: str = typer.Option(None, "--name", "-n", help="Display name for the tool"),
    description: str = typer.Option(
        ..., "--description", "-d", help="Tool description"
    ),
    enabled: bool = typer.Option(
        True, "--enabled/--disabled", help="Enable or disable the tool"
    ),
    category: str = typer.Option(
        "utility",
        "--category",
        "-c",
        help=f"Tool category. Valid values: {TOOL_CATEGORIES}",
    ),
    # Regular tool options
    module: str = typer.Option(
        None,
        "--module",
        "-m",
        help="Python module path (e.g., my_package.tools). Not used with --endpoint.",
    ),
    path: str = typer.Option(
        None,
        "--path",
        help="Path to Python module file. Not used with --endpoint.",
    ),
    tool_class: str = typer.Option(
        None,
        "--tool-class",
        help="Name of the tool class to use from the module. Required for regular tools.",
    ),
    # MCP tool options
    endpoint: str = typer.Option(
        None,
        "--endpoint",
        help="Endpoint URL for MCP tools (e.g., https://api.example.com/tools/summarize)",
    ),
    provider: str = typer.Option(
        None,
        "--provider",
        help="Provider name for MCP tools (e.g., openai, anthropic)",
    ),
    # Common metadata
    permissions: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--permission",
        "-p",
        help=(
            "Required permissions for the tool (can be specified multiple times). "
            f"Valid values: {TOOL_PERMISSIONS}"
        ),
    ),
    tags: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--tag",
        "-t",
        help=(
            "Tags for categorizing the tool (can be specified multiple times). "
            f"Valid values: {TOOL_TAGS}"
        ),
    ),
    # Other options
    config_file: str | None = typer.Option(
        None, "--config-file", help="Configuration file to update"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    ),
):
    """Add a new tool to the configuration.

    For regular tools, specify either --module or --path.
    For MCP tools, specify both --endpoint and --provider.
    """
    try:
        tool_config = _build_tool_config(
            tool_id=tool_id,
            name=name,
            description=description,
            enabled=enabled,
            category=category,
            module=module,
            path=path,
            tool_class=tool_class,
            endpoint=endpoint,
            provider=provider,
            permissions=permissions,
            tags=tags,
        )

        action = _persist_tool_config(tool_config, config_file=config_file)

        if enabled:
            context = ToolCLIContext.create(log_level, config_file=config_file)
            _verify_tool_loaded(
                context.tool_manager,
                tool_config["id"],
                tool_config["name"],
                action=action,
            )
        else:
            typer.echo(
                f"✅ Successfully {action.lower()} disabled tool "
                f"'{tool_config['name']}' with ID '{tool_config['id']}'"
            )

    except ToolCLIError as exc:
        log.error(str(exc))
        raise typer.Exit(1) from exc
    except Exception as exc:  # pragma: no cover - unforeseen runtime issues
        log.error(
            "Failed to add tool: %s", exc, exc_info=log.isEnabledFor(logging.DEBUG)
        )
        raise typer.Exit(1) from exc


@app.command()
@safe_entrypoint("cli.tool.remove")
def remove(
    tool_id: str = typer.Argument(..., help="ID of the tool to remove"),
    config_file: str | None = typer.Option(
        None, "--config-file", help="Configuration file to update"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    ),
):
    """Remove a tool from the configuration."""

    try:
        config_path = _remove_tool_from_config(tool_id, config_file=config_file)
        typer.echo(f"Removed tool with ID '{tool_id}' from {config_path}")

        context = ToolCLIContext.create(log_level, config_file=config_file)

        try:
            context.tool_manager.get_tool(tool_id)
        except Exception:
            typer.echo(f"✅ Successfully removed tool with ID '{tool_id}'")
            return

        raise ToolCLIError(f"Tool with ID '{tool_id}' is still available after removal")

    except ToolCLIError as exc:
        typer.echo(f"❌ {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - unforeseen runtime issues
        typer.echo(f"❌ Error removing tool: {exc}", err=True)
        if log_level.upper() == "DEBUG":
            import traceback

            typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(code=1) from exc


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
    show_stats: bool = typer.Option(
        False, "--stats", help="Show execution statistics after running the tool"
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

        # Run a tool and show statistics
        locca tool run --stats my_tool arg1=value1
    """
    # Create the context with sandbox enabled if requested
    ctx = ToolCLIContext.create(log_level, sandbox=sandbox)
    tool_manager = ctx.tool_manager

    try:
        # Parse the arguments for the single tool
        parsed_args = _parse_tool_args(args or [])

        # Run the tool with parsed arguments
        result = asyncio.run(
            _run_tool(tool_manager, tool_id, parsed_args, sandbox=sandbox)
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

        # Show statistics if requested
        if show_stats:
            _display_tool_stats(tool_manager, tool_id, console)

    except ToolCLIError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if show_stats:
            _display_tool_stats(tool_manager, tool_id, console)
        raise typer.Exit(1) from e
    except Exception as exc:  # pragma: no cover - unforeseen runtime issues
        log.error(
            "Failed to execute tool: %s", exc, exc_info=log.isEnabledFor(logging.DEBUG)
        )
        console.print(f"\n[red]Error: {exc}[/red]")
        if show_stats:
            _display_tool_stats(tool_manager, tool_id, console)
        raise typer.Exit(1) from exc


def _validate_tool_specs(
    tool_specs_list: list[tuple[str, dict]], parallel: bool
) -> None:
    """Validate tool specifications and parallel execution constraints.

    Args:
        tool_specs_list: List of (tool_id, args_dict) tuples
        parallel: Whether parallel execution is requested

    Raises:
        typer.Exit: If validation fails
    """
    if not tool_specs_list:
        typer.echo("Error: No valid tool specifications provided", err=True)
        raise typer.Exit(1)

    if parallel and len(tool_specs_list) > 4:
        typer.echo(
            f"Error: Cannot run {len(tool_specs_list)} tools in parallel. "
            "Maximum of 4 tools allowed in parallel mode.",
            err=True,
        )
        raise typer.Exit(1)


def _display_execution_plan(
    tool_specs_list: list[tuple[str, dict]], parallel: bool
) -> None:
    """Display the execution plan to the user.

    Args:
        tool_specs_list: List of (tool_id, args_dict) tuples
        parallel: Whether tools will run in parallel
    """
    console.print(
        f"\n[bold]Execution Plan ({'parallel' if parallel else 'sequential'}):[/bold]"
    )
    for i, (tool_id, args) in enumerate(tool_specs_list, 1):
        args_str = f" with args: {args}" if args else ""
        console.print(f"  {i}. [bold]{tool_id}[/bold]{args_str}")


def _display_tool_result(index: int, tool_id: str, success: bool, result: Any) -> int:
    """Display the result of a single tool execution.

    Args:
        index: The 1-based index of the tool
        tool_id: The ID of the tool
        success: Whether the tool executed successfully
        result: The result of the tool execution

    Returns:
        int: 1 if successful, 0 otherwise
    """
    status = "[green]✓[/green]" if success else "[red]✗[/red]"
    console.print(f"\n{status} [bold]{index}. {tool_id}[/bold]")
    console.print("-" * 50)

    if isinstance(result, str):
        console.print(result)
    elif result is not None:
        console.print(yaml.dump(result, default_flow_style=False, sort_keys=False))

    return 1 if success else 0


def _display_summary(success_count: int, total_tools: int, duration: float) -> None:
    """Display the execution summary.

    Args:
        success_count: Number of tools that executed successfully
        total_tools: Total number of tools executed
        duration: Total execution time in seconds
    """
    console.print("\n" + "=" * 50)
    console.print(
        f"[bold]Summary:[/bold] {success_count} of {total_tools} tools executed successfully"
    )
    console.print(f"Total duration: {_format_duration(duration)}")


@app.command()
@safe_entrypoint("cli.tool.run_multiple")
def run_multiple(
    tool_specs: str = typer.Argument(
        ...,
        help=(
            "Tool specifications in format 'tool1:arg1=val1,arg2=val2;tool2:arg1=val1'. "
            "Separate tools with semicolons and arguments with commas."
        ),
    ),
    sandbox: bool = typer.Option(
        False, "--sandbox", "-s", help="Execute tools in sandbox environment"
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        "-p",
        help="Execute tools in parallel (max 4 tools at once)",
    ),
    show_stats: bool = typer.Option(
        False, "--stats", help="Show execution statistics after running the tools"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    ),
) -> Any:
    """Execute multiple tools in sequence or parallel.
    
    Examples:
        # Run tools sequentially
        locca tool run-multiple "tool1:arg1=val1;tool2:arg1=val1"
        
        # Run tools in parallel
        locca tool run-multiple --parallel "tool1:arg1=val1;tool2:arg1=val1"
        
        # Run tools in a sandbox
        locca tool run-multiple --sandbox "tool1:arg1=val1;tool2:arg1=val1"
        
        # Complex example with multiple tools and arguments
        locca tool run-multiple \\
            "tool1:arg1=val1,arg2=val2;\
            tool2:arg1=val3;\
            tool3:arg1=val4,arg2=val5"
    """
    tool_specs_list: list[tuple[str, dict[str, Any]]] = []
    ctx = ToolCLIContext.create(log_level, sandbox=sandbox)
    tool_manager = ctx.tool_manager

    try:
        # Parse and validate tool specifications
        tool_specs_list = _parse_tool_specs(tool_specs)
        _validate_tool_specs(tool_specs_list, parallel)

        _display_execution_plan(tool_specs_list, parallel)

        # Execute tools and measure duration
        start_time = time.time()
        results = asyncio.run(
            _run_tools(
                tool_manager=tool_manager,
                tool_specs=tool_specs_list,
                sandbox=sandbox,
                parallel=parallel,
            )
        )
        duration = time.time() - start_time

        # Process and display results
        console.print("\n[bold]Execution Results:[/bold]")
        success_count = 0
        for i, (tool_id, success, result) in enumerate(results, 1):
            success_count += _display_tool_result(i, tool_id, success, result)

        # Show summary and statistics
        _display_summary(success_count, len(results), duration)

        if show_stats:
            console.print("\n[bold]Tool Statistics:[/bold]")
            for tool_id, _ in tool_specs_list:
                _display_tool_stats(tool_manager, tool_id, console)

            # Show system-wide stats
            _display_system_stats(tool_manager, console)

        # Return non-zero exit code if any tool failed
        if success_count < len(results):
            # Show which tools failed
            failed_tools = [tool_id for tool_id, success, _ in results if not success]
            console.print(
                f"\n[red]Error: The following tools failed: {', '.join(failed_tools)}[/red]"
            )

            for _, success, error in results:
                if not success and error:
                    console.print(f"\n[red]Error details: {error}[/red]")

            return typer.Exit(1)

    except Exception as e:
        console.print(f"\n[red]Error: {e!s}[/red]")
        if "tool_specs_list" in locals() and "ctx" in locals() and show_stats:
            console.print("\n[bold]Tool Statistics:[/bold]")
            for tool_id, _ in tool_specs_list:
                _display_tool_stats(tool_manager, tool_id, console)
        raise typer.Exit(1) from e


@app.command()
@safe_entrypoint("cli.tool.validate")
def validate(
    config_file: str | None = typer.Option(
        None, "--config-file", help="Configuration file to validate"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    ),
) -> None:
    """Validate tool configurations and check for errors."""
    level = _resolve_log_level(log_level)
    log.setLevel(level)

    try:
        config_path = _get_config_path(config_file)
        is_valid, errors = validate_configuration_file(config_path)

        if is_valid:
            console.print("[green]✓ Configuration is valid[/green]")
            return

        # Print errors with rich formatting
        console.print("[red]❌ Configuration validation failed:[/red]")
        for error in errors:
            if error.endswith(":"):  # Header for tool errors
                console.print(f"\n[bold]{error}[/bold]")
            else:
                console.print(f"  {error}")

        raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]❌ Validation failed: {e}[/red]")
        if log_level.upper() == "DEBUG":
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(code=1) from e


@app.command()
@safe_entrypoint("cli.tool.reload")
def reload(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    ),
) -> None:
    """Reload tools from all configured sources."""
    try:
        context = ToolCLIContext.create(log_level)
        typer.echo("Reloading tools...")
        context.tool_manager.reload_tools()
        typer.echo("✅ Successfully reloaded all tools")
    except ToolCLIError as exc:
        typer.echo(f"❌ {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - unforeseen runtime issues
        typer.echo(f"❌ Failed to reload tools: {exc}", err=True)
        raise typer.Exit(code=1) from exc
