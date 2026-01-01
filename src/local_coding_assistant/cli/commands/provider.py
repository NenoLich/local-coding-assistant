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

from local_coding_assistant.config.env_manager import EnvManager
from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.utils.logging import get_logger

app = typer.Typer(name="provider", help="Manage LLM providers")
log = get_logger("cli.provider")


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
        Path to the providers configuration file
    """
    if config_file:
        return Path(config_file)

    # Create or use provided env_manager
    env_manager = env_manager or EnvManager.create(load_env=True)

    # Use PathManager to resolve the config path based on environment
    return env_manager.path_manager.resolve_path("@config/providers.local.yaml")


def _load_config(config_path: Path) -> dict:
    """Load provider configuration from file."""
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _ensure_config_directory(config_path: Path) -> None:
    """Ensure the configuration directory exists."""
    config_path.parent.mkdir(parents=True, exist_ok=True)


def _create_empty_config(config_path: Path) -> None:
    """Create an empty configuration file with default structure."""
    with open(config_path, "w") as f:
        yaml.dump({"providers": {}}, f, default_flow_style=False)


def _convert_to_new_format(config: dict) -> dict:
    """Convert old flat config format to new nested format if needed."""
    if not any(key == "providers" for key in config.keys()):
        return {"providers": config}
    return config


def _update_existing_provider(
    existing_config: dict, provider_name: str, new_config: dict
) -> tuple[dict, str]:
    """Update an existing provider's configuration."""
    if not isinstance(existing_config, dict):
        existing_config = {}

    # Ensure providers key exists
    if "providers" not in existing_config:
        existing_config["providers"] = {}

    # Check if we're updating from a nested config or adding a new one
    if "providers" in new_config and provider_name in new_config["providers"]:
        # Update existing provider config
        if provider_name in existing_config["providers"]:
            existing_config["providers"][provider_name].update(
                new_config["providers"][provider_name]
            )
        else:
            existing_config["providers"][provider_name] = new_config["providers"][
                provider_name
            ]
        action = "Updated"
    else:
        # Add new provider config
        existing_config["providers"][provider_name] = new_config
        action = "Added"

    return existing_config, action


def _save_config(
    config_path: Path,
    config: dict | None = None,
    provider_name: str | None = None,
) -> None:
    """Save provider configuration to file.

    This function handles both the new nested structure (with 'providers' key)
    and maintains backward compatibility with the old flat structure.

    Args:
        config_path: Path to the config file
        config: Config dictionary to save. Can be in either format:
            - New format: {'providers': {'provider1': {...}, 'provider2': {...}}}
            - Old format: {'provider1': {...}, 'provider2': {...}}
        provider_name: Optional name of the provider being saved (for logging)
    """
    try:
        _ensure_config_directory(config_path)

        # Handle empty config case
        if config is None:
            _create_empty_config(config_path)
            return

        # Ensure consistent config format
        config = _convert_to_new_format(config)
        action = None

        # Handle provider-specific updates
        if provider_name:
            try:
                existing_config = _load_config(config_path)
                config, action = _update_existing_provider(
                    existing_config, provider_name, config
                )
            except Exception as e:
                typer.echo(f"⚠️  Warning: Could not load existing config: {e}")
                action = "Added"

        # Save the final configuration
        try:
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except yaml.YAMLError:
            # Re-raise YAML serialization errors directly
            raise
        except Exception as e:
            # For other I/O or permission errors, wrap in a more specific error
            raise OSError(f"Failed to write to config file: {e}") from e

        # Provide user feedback if a provider was modified
        if provider_name and action:
            typer.echo(f"✅ {action} provider '{provider_name}' in {config_path}")

    except yaml.YAMLError:
        # Re-raise YAML serialization errors directly
        raise
    except Exception as e:
        error_msg = f"❌ Failed to save configuration to {config_path}: {e!s}"
        typer.echo(error_msg, err=True)
        raise typer.Exit(code=1) from e


def _create_models_config(models: list[str] | None) -> dict:
    """Create configuration for models with default supported parameters.

    Args:
        models: List of model names. If None or empty, returns an empty dict.
    """
    if not models:
        return {}

    default_params = ["max_tokens", "temperature", "top_p"]

    return {
        model_name: {"supported_parameters": default_params.copy()}
        for model_name in models
    }


def _create_provider_config(
    driver: str,
    base_url: str,
    models_config: dict,
    api_key: str | None = None,
    api_key_env: str | None = None,
    health_check_endpoint: str | None = None,
) -> dict:
    """Create a provider configuration dictionary."""
    config = {
        "driver": driver,
        "base_url": base_url,
        "models": models_config,
    }

    if api_key_env:
        config["api_key_env"] = api_key_env
    if api_key:
        config["api_key"] = api_key
    if health_check_endpoint:
        config["health_check_endpoint"] = health_check_endpoint

    return config


def _verify_provider_health(llm_manager, provider_name: str) -> None:
    """Verify that the provider was loaded correctly and is healthy."""
    try:
        provider_status_list = llm_manager.get_provider_status_list()
        available_providers = [p["name"] for p in provider_status_list]

        if provider_name not in available_providers:
            typer.echo(
                f"Error: Provider '{provider_name}' is not available after loading. "
                f"Available providers: {', '.join(available_providers)}",
                err=True,
            )
            raise typer.Exit(code=1)

        # Check provider status
        provider_status = next(
            (p for p in provider_status_list if p.get("name") == provider_name), None
        )

        if provider_status:
            status_msg = provider_status.get("status", "unknown")
            if status_msg == "health_check_not_configured":
                typer.echo(
                    f"Provider '{provider_name}' was added successfully. "
                    "Note: Health check is not configured for this provider.",
                    err=True,
                )
            elif status_msg != "available":
                error_msg = provider_status.get("error", "")
                health_status = f"Status: {status_msg}"
                if error_msg:
                    health_status = f"{health_status} - {error_msg}"
                typer.echo(
                    f"⚠️  Warning: Provider '{provider_name}' is available but not healthy. {health_status}",
                    err=True,
                )
    except Exception as e:
        typer.echo(f"Error checking provider status: {e!s}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
@safe_entrypoint("cli.provider.add")
def add(
    name: str = typer.Argument(..., help="Provider name"),
    models: list[str] | None = typer.Argument(  # noqa B008
        None, help="Model names (space-separated)"
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", help="API key for the provider"
    ),
    api_key_env: str | None = typer.Option(
        None, "--api-key-env", help="Environment variable containing the API key"
    ),
    base_url: str = typer.Option(
        ..., "--base-url", help="Base URL for the provider API (required)"
    ),
    dev: bool = typer.Option(False, "--dev", help="Use development configuration path"),
    driver: str = typer.Option(
        "openai_chat",
        "--driver",
        help="Driver type (openai_chat, openai_responses, local)",
    ),
    health_check_endpoint: str | None = typer.Option(
        None,
        "--health-check-endpoint",
        help="Health check endpoint URL. Can be a full URL or a path that will be appended to base_url",
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
    # Extract and validate parameters
    actual_api_key_env = _extract_value(api_key_env)
    actual_base_url = _extract_value(base_url)
    actual_driver = _extract_value(driver, "openai_chat")
    actual_log_level = _extract_value(log_level, "INFO")
    actual_config_file = _extract_value(config_file)
    actual_api_key = _extract_value(api_key)
    actual_health_check_endpoint = _extract_value(health_check_endpoint)

    # Show configuration info
    if actual_base_url:
        typer.echo(f"Base URL: {actual_base_url}")
    typer.echo(f"Driver: {actual_driver}")

    # Create models configuration (can be empty)
    models_config = _create_models_config(models)

    # Create provider configuration
    provider_config = _create_provider_config(
        driver=actual_driver,
        base_url=actual_base_url,
        models_config=models_config,
        api_key=actual_api_key,
        api_key_env=actual_api_key_env,
        health_check_endpoint=actual_health_check_endpoint,
    )

    # Show warning if no models were provided
    if not models:
        typer.echo(
            "⚠️  Warning: No models provided. You can add models later by editing the configuration file.",
            err=True,
        )

    # Warn if no API key method is provided
    if not actual_api_key_env and not actual_api_key:
        typer.echo(
            "Warning: No API key provided. You'll need to set it in the configuration file later.",
            err=True,
        )

    # Save the configuration
    config_path = _get_config_path(actual_config_file)
    _save_config(
        config_path=config_path,
        provider_name=name,
        config={"providers": {name: provider_config}},
    )

    # Initialize the provider and verify it's working
    level = getattr(logging, actual_log_level.upper(), logging.INFO)
    try:
        ctx = bootstrap(log_level=level)
        llm_manager = ctx["llm"]
        if llm_manager is None:
            typer.echo(
                "Error: LLM manager not available (initialization failed)", err=True
            )
            raise typer.Exit(code=1)

        _verify_provider_health(llm_manager, name)
        if models:
            typer.echo(
                f"✅ Successfully added and verified provider '{name}'. "
                f"Available models: {', '.join(models)}"
            )
        else:
            typer.echo(
                f"✅ Successfully added and verified provider '{name}'. "
                "No models configured yet."
            )
    except Exception as e:
        typer.echo(f"❌ Error initializing provider: {e!s}", err=True)
        raise typer.Exit(code=1) from e


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

    # Get the providers dictionary, defaulting to empty dict if not present
    providers = config.get("providers", {})

    if name not in providers:
        typer.echo(f"Provider '{name}' not found in configuration")
        raise typer.Exit(code=1)

    # Remove the provider
    del providers[name]

    # Update the config with the modified providers
    config["providers"] = providers

    # Save the updated configuration
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

    # Verify the provider is no longer available
    try:
        # This should raise an exception if the provider doesn't exist
        provider = llm_manager.get_provider(name)
        if provider is not None:
            typer.echo(
                f"❌ Error: Provider '{name}' is still available after removal",
                err=True,
            )
            raise typer.Exit(code=1)
    except Exception as e:
        # Expected case - provider should not be found
        if (
            "not found" not in str(e).lower()
            and "no such provider" not in str(e).lower()
        ):
            typer.echo(f"❌ Error verifying provider removal: {e}", err=True)
            if actual_log_level.upper() == "DEBUG":
                import traceback

                typer.echo(traceback.format_exc(), err=True)
            raise typer.Exit(code=1) from e

    typer.echo(f"✅ Successfully removed provider '{name}'")


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


def _validate_model_config(model_name: str, model_config: dict) -> tuple[bool, bool]:
    """Validate a single model's configuration.

    Args:
        model_name: Name of the model being validated
        model_config: Configuration dictionary for the model

    Returns:
        Tuple of (has_errors, has_warnings)
    """
    has_errors = False
    has_warnings = False

    if not isinstance(model_config, dict):
        typer.echo(f"❌ Error: Model '{model_name}' configuration must be a dictionary")
        return True, False

    # Check for supported_parameters in model config
    if "supported_parameters" not in model_config:
        typer.echo(
            f"⚠️  Warning: Model '{model_name}' is missing 'supported_parameters'"
        )
        has_warnings = True
    elif not isinstance(model_config["supported_parameters"], list):
        typer.echo(
            f"❌ Error: 'supported_parameters' for model '{model_name}' must be a list"
        )
        has_errors = True

    return has_errors, has_warnings


def _validate_provider_models(models: dict) -> tuple[bool, bool]:
    """Validate all models for a provider.

    Args:
        models: Dictionary of model configurations

    Returns:
        Tuple of (has_errors, has_warnings)
    """
    has_errors = False
    has_warnings = False

    if not models:
        typer.echo("⚠️  Warning: No models configured for this provider")
        return False, True

    typer.echo(f"✅ Found {len(models)} model(s)")
    for model_name, model_config in models.items():
        model_errors, model_warnings = _validate_model_config(model_name, model_config)
        has_errors |= model_errors
        has_warnings |= model_warnings

    return has_errors, has_warnings


def _validate_provider_config(
    provider_name: str, provider_config: dict
) -> tuple[bool, bool]:
    """Validate a single provider's configuration.

    Args:
        provider_name: Name of the provider being validated
        provider_config: Configuration dictionary for the provider

    Returns:
        Tuple of (has_errors, has_warnings)
    """
    has_errors = False
    has_warnings = False

    typer.echo(f"\nValidating provider: {provider_name}")

    # Check if provider config is a dictionary
    if not isinstance(provider_config, dict):
        typer.echo(
            f"❌ Error: Provider '{provider_name}' configuration must be a dictionary"
        )
        return True, False

    # Check required fields
    required_fields = ["driver", "base_url"]
    missing_fields = [
        field
        for field in required_fields
        if field not in provider_config or not provider_config[field]
    ]

    if missing_fields:
        typer.echo(f"❌ Error: Missing required fields: {missing_fields}")
        has_errors = True

    # Check for at least one of api_key or api_key_env
    if not any(key in provider_config for key in ["api_key", "api_key_env"]):
        typer.echo(
            "⚠️  Warning: Neither 'api_key' nor 'api_key_env' is set. "
            "At least one is recommended for API authentication."
        )
        has_warnings = True

    # Validate models
    models = provider_config.get("models", {})
    model_errors, model_warnings = _validate_provider_models(models)
    has_errors |= model_errors
    has_warnings |= model_warnings

    # Validate driver type if provided
    if "driver" in provider_config:
        valid_drivers = ["openai_chat", "openai_responses", "local"]
        if provider_config["driver"] not in valid_drivers:
            typer.echo(
                f"⚠️  Warning: Unknown driver type: {provider_config['driver']}. "
                f"Valid drivers are: {', '.join(valid_drivers)}"
            )
            has_warnings = True

    return has_errors, has_warnings


def _print_validation_summary(has_errors: bool, has_warnings: bool) -> None:
    """Print the validation summary based on the validation results."""
    typer.echo("\nValidation completed:")
    if has_errors:
        typer.echo("❌ Validation failed with errors")
        raise typer.Exit(code=1)
    elif has_warnings:
        typer.echo("⚠️  Validation completed with warnings")
    else:
        typer.echo("✅ Configuration is valid")


def _validate_configuration(config_path: Path) -> None:
    """Validate the provider configuration file.

    The configuration should be a dictionary where each key is a provider name
    and each value is a dictionary with the following structure:

    provider_name:
      driver: str  # Required: The driver to use (e.g., 'openai_chat', 'openai_responses')
      base_url: str  # Required: Base URL for the provider's API
      api_key: str  # Optional: Direct API key
      api_key_env: str  # Optional: Environment variable containing the API key
      health_check_endpoint: str  # Optional: Endpoint for health checks
      models:  # Required: Dictionary of available models
        model_name:
          supported_parameters: list[str]  # List of supported parameters for the model
    """
    try:
        config = _load_config(config_path)
        providers = config.get("providers", {})

        if not config:
            typer.echo("Configuration is empty")
            return

        if not isinstance(providers, dict):
            typer.echo("❌ Error: Configuration must be a dictionary")
            raise typer.Exit(code=1)

        typer.echo(f"Found {len(providers)} provider(s):")

        # Track validation results
        has_errors = False
        has_warnings = False

        for provider_name, provider_config in providers.items():
            provider_errors, provider_warnings = _validate_provider_config(
                provider_name, provider_config
            )
            has_errors |= provider_errors
            has_warnings |= provider_warnings

        _print_validation_summary(has_errors, has_warnings)

    except yaml.YAMLError as e:
        typer.echo(f"❌ YAML parsing error: {e}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"❌ Unexpected error during validation: {e}")
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
