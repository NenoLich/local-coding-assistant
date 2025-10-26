"""
Provider management commands for the CLI.

This module provides commands to manage LLM providers through configuration files
and the bootstrap system, following the same patterns as other CLI commands.
"""

import logging
from pathlib import Path
from typing import Any

import typer
import yaml

from local_coding_assistant.core import bootstrap
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.utils.logging import get_logger

app = typer.Typer(name="provider", help="Manage LLM providers")
log = get_logger("cli.provider")


def _extract_value(param_value: Any, default: Any = None) -> Any:
    """Extract the actual value, handling typer objects."""
    if isinstance(param_value, typer.models.OptionInfo):
        return param_value.default
    return param_value if param_value is not None else default


def _get_config_path(config_file: str | None = None) -> Path:
    """Get the configuration file path."""
    if config_file:
        return Path(config_file)
    return Path.home() / ".local-coding-assistant" / "config" / "providers.local.yaml"


def _load_config(config_path: Path) -> dict:
    """Load provider configuration from file."""
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _save_config(config_path: Path, config: dict) -> None:
    """Save provider configuration to file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


@app.command()
@safe_entrypoint("cli.provider.add")
def add(
    name: str = typer.Argument(..., help="Provider name"),
    *models: str,
    api_key: str | None = typer.Option(
        None, "--api-key", help="API key for the provider"
    ),
    api_key_env: str | None = typer.Option(
        None, "--api-key-env", help="Environment variable containing the API key"
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Base URL for the provider API"
    ),
    driver: str = typer.Option(
        "openai_chat",
        "--driver",
        help="Driver type (openai_chat, openai_responses, local)",
    ),
    config_file: str | None = typer.Option(
        None, "--config-file", help="Configuration file to update"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    ),
) -> None:
    """Add a new provider to the configuration.

    This command adds a provider to the local configuration file and triggers
    a reload through the bootstrap system to make it available immediately.
    """
    # Extract actual parameter values - handle typer objects and None
    actual_api_key_env = _extract_value(api_key_env)
    actual_base_url = _extract_value(base_url)
    actual_driver = _extract_value(driver, "openai_chat")
    actual_log_level = _extract_value(log_level, "INFO")
    actual_config_file = _extract_value(config_file)
    actual_api_key = _extract_value(api_key)

    if actual_base_url:
        typer.echo(f"Base URL: {actual_base_url}")
    typer.echo(f"Driver: {actual_driver}")

    # Build provider configuration
    provider_config = {
        "driver": actual_driver,
        "models": {model_name: {} for model_name in models},
    }

    if actual_base_url:
        provider_config["base_url"] = actual_base_url

    if actual_api_key_env:
        provider_config["api_key_env"] = actual_api_key_env
    elif actual_api_key:
        provider_config["api_key"] = actual_api_key

    # Update configuration file
    config_path = _get_config_path(actual_config_file)
    config = _load_config(config_path)
    config[name] = provider_config
    _save_config(config_path, config)

    typer.echo(f"Added provider '{name}' to {config_path}")

    # Map provided level string to logging.* constant (default INFO)
    level = getattr(logging, actual_log_level.upper(), logging.INFO)

    # Bootstrap with the new configuration
    ctx = bootstrap(log_level=level)
    llm_manager = ctx["llm"]

    if llm_manager is None:
        typer.echo("Error: LLM manager not available (provider initialization failed)")
        raise typer.Exit(code=1)

    # Trigger reload through the bootstrap system
    llm_manager.reload_providers()
    typer.echo(f"Provider '{name}' is now available")


@app.command(name="list")
@safe_entrypoint("cli.provider.list")
def list_providers(
    provider: str | None = typer.Option(
        None, "--provider", help="Specific provider to list models for"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    ),
) -> None:
    """List available providers or models for a specific provider."""
    # Extract actual parameter values - handle typer objects and None
    actual_provider = _extract_value(provider)
    actual_log_level = _extract_value(log_level, "INFO")

    # Map provided level string to logging.* constant (default INFO)
    level = getattr(logging, actual_log_level.upper(), logging.INFO)

    ctx = bootstrap(log_level=level)
    llm_manager = ctx["llm"]

    if llm_manager is None:
        typer.echo("Error: LLM manager not available (provider initialization failed)")
        raise typer.Exit(code=1)

    if actual_provider:
        _list_specific_provider(llm_manager, actual_provider)
    else:
        _list_all_providers(llm_manager)


def _list_specific_provider(llm_manager, provider_name: str) -> None:
    """List models for a specific provider."""
    if provider_name in llm_manager.provider_manager.list_providers():
        provider_instance = llm_manager.provider_manager.get_provider(provider_name)
        source = (
            llm_manager.provider_manager.get_provider_source(provider_name) or "unknown"
        )

        typer.echo(f"Provider: {provider_name} ({source})")
        if provider_instance:
            models = provider_instance.get_available_models()
            if models:
                typer.echo("Models:")
                for model_name in models:
                    typer.echo(f"  - {model_name}")
            else:
                typer.echo("No models available")
        else:
            typer.echo(f"Provider '{provider_name}' failed to initialize")
    else:
        typer.echo(f"Provider '{provider_name}' not found")


def _list_all_providers(llm_manager) -> None:
    """List all available providers."""
    status_list = llm_manager.get_provider_status_list()
    if status_list:
        typer.echo("Available providers:")
        typer.echo(f"{'Name':<20} {'Source':<10} {'Status':<12} {'Models'}")
        typer.echo("-" * 60)

        for provider_status in status_list:
            typer.echo(
                f"{provider_status['name']:<20} "
                f"{provider_status['source']:<10} "
                f"{provider_status['status']:<12} "
                f"{provider_status['models']} models"
            )
    else:
        typer.echo("No providers configured")


@app.command()
@safe_entrypoint("cli.provider.remove")
def remove(
    name: str = typer.Argument(..., help="Provider name to remove"),
    config_file: str | None = typer.Option(
        None, "--config-file", help="Configuration file to update"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    ),
) -> None:
    """Remove a provider from the configuration."""
    # Extract actual parameter values - handle typer objects and None
    actual_config_file = _extract_value(config_file)
    actual_log_level = _extract_value(log_level, "INFO")

    typer.echo(f"Removing provider: {name}")

    config_path = _get_config_path(actual_config_file)
    config = _load_config(config_path)

    if name not in config:
        typer.echo(f"Provider '{name}' not found in configuration")
        raise typer.Exit(code=1)

    del config[name]
    _save_config(config_path, config)

    typer.echo(f"Removed provider '{name}' from {config_path}")

    # Map provided level string to logging.* constant (default INFO)
    level = getattr(logging, actual_log_level.upper(), logging.INFO)

    # Bootstrap with updated configuration
    ctx = bootstrap(log_level=level)
    llm_manager = ctx["llm"]

    if llm_manager is None:
        typer.echo("Error: LLM manager not available (provider initialization failed)")
        raise typer.Exit(code=1)

    # Trigger reload through the bootstrap system
    llm_manager.reload_providers()
    typer.echo(f"Provider '{name}' has been removed")


@app.command()
@safe_entrypoint("cli.provider.validate")
def validate(
    config_file: str | None = typer.Option(
        None, "--config-file", help="Configuration file to validate"
    ),
) -> None:
    """Validate provider configuration files."""
    # Extract actual parameter values - handle typer objects and None
    actual_config_file = _extract_value(config_file)

    typer.echo("Validating provider configuration...")

    config_path = _get_config_path(actual_config_file)
    if not config_path.exists():
        typer.echo(f"No provider configuration found at {config_path}")
        return

    _validate_configuration(config_path)


def _validate_configuration(config_path: Path) -> None:
    """Validate the provider configuration file."""
    try:
        config = _load_config(config_path)

        if not config:
            typer.echo("Configuration is empty")
            return

        typer.echo(f"Found {len(config)} provider(s):")
        for provider_name, provider_config in config.items():
            if not isinstance(provider_config, dict):
                typer.echo(
                    f"❌ Error: Provider '{provider_name}' configuration must be a dictionary"
                )
                continue

            # Check required fields
            required_fields = ["driver"]
            missing_fields = [
                field for field in required_fields if field not in provider_config
            ]
            if missing_fields:
                typer.echo(
                    f"⚠️  Warning: Provider '{provider_name}' missing fields: {missing_fields}"
                )

            # Check models
            models = provider_config.get("models", {})
            if not models:
                typer.echo(
                    f"⚠️  Warning: Provider '{provider_name}' has no models configured"
                )
            else:
                typer.echo(f"✅ Provider '{provider_name}' has {len(models)} model(s)")

        typer.echo("Configuration validation completed")

    except yaml.YAMLError as e:
        typer.echo(f"❌ YAML parsing error: {e}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"❌ Validation error: {e}")
        raise typer.Exit(code=1) from e


@app.command()
@safe_entrypoint("cli.provider.reload")
def reload(
    log_level: str | None = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    ),
) -> None:
    """Reload providers from all sources (builtin, global, local)."""
    # Extract actual parameter values - handle typer objects and None
    actual_log_level = _extract_value(log_level, "INFO")

    typer.echo("Reloading providers...")

    # Map provided level string to logging.* constant (default INFO)
    if actual_log_level is None:
        level = logging.INFO
    else:
        level = getattr(logging, actual_log_level.upper(), logging.INFO)

    ctx = bootstrap(log_level=level)
    llm_manager = ctx["llm"]

    if llm_manager is None:
        typer.echo("Error: LLM manager not available (provider initialization failed)")
        raise typer.Exit(code=1)

    # Trigger reload through the bootstrap system
    llm_manager.reload_providers()
    typer.echo("Providers reloaded successfully")
