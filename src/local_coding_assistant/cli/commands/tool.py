"""
Tool management commands for the CLI.

This module provides commands to manage tools through configuration files
and the bootstrap system, following the same patterns as other CLI commands.
"""

import ast
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import typer
import yaml
from rich.console import Console
from rich.table import Table

from local_coding_assistant.core import bootstrap
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.core.exceptions import CLIError
from local_coding_assistant.tools.types import (
    ToolCategory,
    ToolExecutionRequest,
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
    ) -> "ToolCLIContext":
        """Bootstrap the application and return a ready-to-use context."""

        level = _resolve_log_level(log_level)
        ctx = bootstrap(config_path=config_file, log_level=level)
        tool_manager = ctx.get("tools")
        if tool_manager is None:
            raise ToolCLIError("Tool manager not available (initialization failed)")
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
    tool_infos = tool_manager.list_tools()
    rows = []
    for info in tool_infos:
        row = _normalize_tool_info(info)
        if available_only and not row.available:
            continue
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


def _validate_regular_tool_config(module: str | None, path: str | None) -> None:
    """Validate regular tool configuration."""
    if not (module or path):
        raise ToolCLIError("Either --module or --path is required for regular tools")
    if module and path:
        raise ToolCLIError("Cannot provide both --module and --path")


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
) -> dict[str, Any]:
    """Construct a validated tool configuration payload."""
    is_mcp = endpoint is not None or provider is not None

    # Validate configuration based on tool type
    if is_mcp:
        _validate_mcp_tool_config(endpoint, provider, module, path)
    else:
        _validate_regular_tool_config(module, path)

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
    elif module:
        tool_config["module"] = module
    elif path:
        tool_config["path"] = path

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


def _execute_tool(tool_manager: Any, tool_id: str, payload: dict[str, Any]) -> Any:
    """Execute a tool using the best available manager interface."""

    try:
        if hasattr(tool_manager, "execute"):
            request = ToolExecutionRequest(tool_name=tool_id, payload=payload)
            response = tool_manager.execute(request)
            if hasattr(response, "model_dump"):
                return response.model_dump()
            if hasattr(response, "result"):
                return response.result
            return response

        if hasattr(tool_manager, "run_tool"):
            return tool_manager.run_tool(tool_id, payload)

    except Exception as exc:  # pragma: no cover - delegates to manager
        raise ToolCLIError(str(exc)) from exc

    raise ToolCLIError("Tool manager does not support synchronous execution")


def _run_tool(tool_manager: Any, tool_id: str, args: list[str]) -> None:
    """Execute a tool and echo the result."""

    try:
        tool = tool_manager.get_tool(tool_id)
    except Exception as exc:
        raise ToolCLIError(str(exc)) from exc

    payload = _parse_tool_args(args)

    tool_info = None
    if hasattr(tool_manager, "get_tool_info"):
        try:
            tool_info = tool_manager.get_tool_info(tool_id)
        except Exception:  # pragma: no cover - optional hook
            tool_info = None

    display_name = (
        getattr(tool_info, "name", None) or getattr(tool, "name", None) or tool_id
    )

    typer.echo(f"Executing tool: {display_name} ({tool_id})")

    result = _execute_tool(tool_manager, tool_id, payload)
    typer.echo("\nTool execution result:")
    typer.echo(yaml.dump(result, default_flow_style=False, sort_keys=False))


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


def _get_config_path(config_file: str | None = None, dev: bool = False) -> Path:
    """Get the configuration file path.

    Args:
        config_file: Custom config file path if provided
        dev: If True, use development path in the project directory

    Returns:
        Path to the configuration file
    """
    if config_file:
        return Path(config_file)

    if dev or os.getenv("LOCCA_DEV_MODE"):
        # Try to find project root by looking for pyproject.toml
        current_path = Path(__file__).resolve()
        for parent in [current_path, *current_path.parents]:
            if (parent / "pyproject.toml").exists():
                return (
                    parent
                    / "src"
                    / "local_coding_assistant"
                    / "config"
                    / "tools.local.yaml"
                )

    # Default path for non-dev mode
    return Path.home() / ".local_coding_assistant" / "config" / "tools.yaml"


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
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    ),
):
    """Execute a tool with the given arguments."""

    try:
        context = ToolCLIContext.create(log_level)
        _run_tool(context.tool_manager, tool_id, args or [])
    except ToolCLIError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:  # pragma: no cover - unforeseen runtime issues
        log.error(
            "Failed to execute tool: %s", exc, exc_info=log.isEnabledFor(logging.DEBUG)
        )
        raise typer.Exit(1) from exc


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
