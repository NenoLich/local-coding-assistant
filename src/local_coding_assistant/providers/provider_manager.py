"""
Provider manager for dynamic provider loading and registration
"""

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any

from local_coding_assistant.providers.base import BaseProvider
from local_coding_assistant.providers.generic_provider import GenericProvider
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("providers.manager")


def _deep_merge_dicts(d1: dict, d2: dict) -> dict:
    """
    Recursively merges d2 into d1. d2 values override d1 values.
    """
    merged = d1.copy()
    for k, v in d2.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


class ProviderSource:
    """Information about where a provider comes from."""

    BUILTIN = "builtin"
    GLOBAL = "global"
    LOCAL = "local"


# Module-level registry for provider classes (used by decorators)
_provider_registry: dict[str, type[BaseProvider]] = {}
_provider_sources: dict[str, str] = {}


class ProviderManager:
    """Manages dynamic loading and registration of LLM providers"""

    def __init__(self):
        self._providers: dict[str, type[BaseProvider]] = {}
        self._instances: dict[str, BaseProvider] = {}
        self._provider_configs: dict[
            str, dict[str, Any]
        ] = {}  # provider_name -> config
        self._auto_discovery_paths = [
            Path(__file__).parent,  # Current providers directory
        ]

        # Copy from module-level registry
        self._providers.update(_provider_registry)
        self._provider_sources = _provider_sources.copy()

    def register_provider(self, name: str):
        """Decorator to register a provider class"""

        def decorator(cls: type[BaseProvider]) -> type[BaseProvider]:
            # Register in module-level registry
            _provider_registry[name] = cls
            _provider_sources[name] = ProviderSource.BUILTIN
            logger.debug(f"Registered builtin provider: {name}")
            return cls

        return decorator

    def get_provider_class(self, name: str) -> type[BaseProvider] | None:
        """Get a provider class by name"""
        return self._providers.get(name)

    def get_provider(self, name: str) -> BaseProvider | None:
        """Get a provider instance by name"""
        if name not in self._instances:
            logger.warning(f"Provider '{name}' not found or not initialized.")
            return None
        return self._instances.get(name)

    def list_providers(self) -> list[str]:
        """List all registered provider names"""
        return sorted(list(self._instances.keys()))

    def get_provider_source(self, name: str) -> str | None:
        """Get the source of a provider (builtin, global, local)"""
        # Check module-level registry first (for builtin providers)
        if name in _provider_sources:
            return _provider_sources[name]
        # Check instance registry (for config providers)
        if name in self._provider_sources:
            return self._provider_sources.get(name)
        return None

    def reload(self, config_manager=None) -> None:
        """Reload providers from all sources with layer merging.

        Layer priority (later layers override earlier ones):
        1. Builtins (from *_provider.py files)
        2. Global config (from config manager if available)
        2.5. Global YAML (from providers.yaml file directly)
        3. Local config (from providers.local.yaml)
        """
        logger.info("Reloading providers from all sources")

        # Clear current instances and configs
        self._instances.clear()
        self._provider_configs.clear()

        # Copy fresh from module-level registry
        self._providers.update(_provider_registry)
        self._provider_sources = _provider_sources.copy()
        logger.debug(
            f"Loaded module-level builtin providers: {list(_provider_registry.keys())}"
        )

        # Layer 2: Load from global config if available
        if config_manager:
            self._load_providers_from_config_layer(
                config_manager, ProviderSource.GLOBAL
            )

        # Layer 2.5: Load from global providers.yaml file directly
        self._load_providers_from_global_yaml()

        # Layer 3: Load from local config
        self._load_providers_from_local_config()

        # After all configs are loaded, instantiate the providers
        self._instantiate_providers()

        # Log the final state
        total_providers = len(self._instances)
        logger.info(f"Provider reload complete: {total_providers} providers loaded")
        for name, _instance in self._instances.items():
            logger.debug(f"  {name} ({self._provider_sources.get(name, 'unknown')})")

    def _auto_discover_providers_from_paths(self) -> None:
        """Discover and import provider classes from auto-discovery paths."""
        for path in self._auto_discovery_paths:
            for provider_file in path.glob("*_provider.py"):
                module_name = provider_file.stem
                name = module_name.replace("_provider", "")
                if (
                    name not in self._providers
                ):  # Only discover if not already registered
                    try:
                        spec = importlib.util.spec_from_file_location(
                            module_name, provider_file
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)

                            provider_class = getattr(
                                module, f"{name.title()}Provider", None
                            )
                            if provider_class and issubclass(
                                provider_class, BaseProvider
                            ):
                                self._providers[name] = provider_class
                                self._provider_sources[name] = ProviderSource.BUILTIN
                                logger.debug(
                                    f"Auto-discovered provider from file: {name}"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load provider from {provider_file}: {e}"
                        )

    def _load_providers_from_config_layer(self, config_manager, source: str) -> None:
        """Load providers from a specific config layer."""
        try:
            config = config_manager.global_config
            if not config or not hasattr(config, "providers"):
                return

            providers_config = config.providers
            if not isinstance(providers_config, dict):
                logger.warning("Invalid providers configuration format")
                return

            for provider_name, provider_config in providers_config.items():
                # Convert ProviderConfig to dict if needed
                if hasattr(provider_config, "model_dump"):
                    provider_dict = provider_config.model_dump()
                else:
                    provider_dict = provider_config

                # Register the provider configuration
                existing_config = self._provider_configs.get(provider_name, {})
                self._provider_configs[provider_name] = _deep_merge_dicts(
                    existing_config, provider_dict
                )
                self._provider_sources[provider_name] = source

                logger.debug(f"Loaded {source} provider config: {provider_name}")

        except Exception as e:
            logger.error(f"Failed to load providers from config: {e}")

    def _load_providers_from_global_yaml(self) -> None:
        """Load providers from global providers.yaml file directly."""
        global_config_path = Path(__file__).parent.parent / "config" / "providers.yaml"

        if not global_config_path.exists():
            logger.debug(
                f"Global providers.yaml file does not exist: {global_config_path}"
            )
            return

        try:
            import yaml

            with open(global_config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            if not isinstance(config, dict) or "providers" not in config:
                logger.warning(
                    f"Invalid global providers.yaml format in {global_config_path}"
                )
                return

            providers_config = config["providers"]
            if not isinstance(providers_config, dict):
                logger.warning(f"Invalid providers section in {global_config_path}")
                return

            for provider_name, provider_config in providers_config.items():
                if isinstance(provider_config, dict):
                    existing_config = self._provider_configs.get(provider_name, {})
                    self._provider_configs[provider_name] = _deep_merge_dicts(
                        existing_config, provider_config
                    )
                    self._provider_sources[provider_name] = ProviderSource.GLOBAL
                    logger.debug(f"Loaded global provider config: {provider_name}")

        except Exception as e:
            logger.error(
                f"Failed to load global providers from {global_config_path}: {e}"
            )

    def _load_providers_from_local_config(self) -> None:
        """Load providers from local configuration file."""
        # Use the same path resolution as in _get_config_path
        if os.getenv("LOCCA_DEV_MODE"):
            # For development, use a path relative to the project root
            current_file = Path(__file__).resolve()
            path_str = str(current_file)
            src_index = path_str.find("src/local_coding_assistant")
            if src_index == -1:
                src_index = path_str.find("src\\local_coding_assistant")
            if src_index != -1:
                base_path = path_str[:src_index]
                local_config_path = (
                    Path(base_path)
                    / "src"
                    / "local_coding_assistant"
                    / "config"
                    / "providers.local.yaml"
                )
            else:
                # Fallback to the old method if the path doesn't contain src/local_coding_assistant
                local_config_path = (
                    current_file.parents[3]
                    / "src"
                    / "local_coding_assistant"
                    / "config"
                    / "providers.local.yaml"
                )
        else:
            # For production, use the user's home directory
            local_config_path = (
                Path.home()
                / ".local-coding-assistant"
                / "config"
                / "providers.local.yaml"
            )

        if not local_config_path.exists():
            logger.debug(f"Local config file does not exist: {local_config_path}")
            return

        try:
            import yaml

            with open(local_config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            if not isinstance(config, dict):
                logger.warning(f"Invalid local config format in {local_config_path}")
                return

            # Get the 'providers' dictionary from config, defaulting to empty dict
            providers_config = config.get("providers", {})
            if not isinstance(providers_config, dict):
                logger.warning(f"Invalid 'providers' format in {local_config_path}")
                return

            for provider_name, provider_config in providers_config.items():
                if not isinstance(provider_config, dict):
                    logger.warning(
                        f"Invalid config for provider '{provider_name}' in {local_config_path}"
                    )
                    continue

                existing_config = self._provider_configs.get(provider_name, {})
                self._provider_configs[provider_name] = _deep_merge_dicts(
                    existing_config, provider_config
                )
                self._provider_sources[provider_name] = ProviderSource.LOCAL
                logger.debug(f"Loaded local provider config: {provider_name}")

        except Exception as e:
            logger.error(
                f"Failed to load local providers from {local_config_path}: {e}"
            )

    def _instantiate_providers(self) -> None:
        """Instantiate all loaded providers based on their configurations and classes."""
        all_provider_names = sorted(
            list(set(self._providers.keys()) | set(self._provider_configs.keys()))
        )

        allowed_drivers = {"openai_chat", "openai_responses", "local"}

        for name in all_provider_names:
            try:
                instance_kwargs = {"name": name}
                provider_class = self._providers.get(name)
                config = self._provider_configs.get(name)

                if config:
                    # Normalize models: allow dict (from YAML) or list
                    models = config.get("models")
                    if isinstance(models, dict):
                        # Convert dict of model -> metadata to list of model names
                        config = dict(config)  # shallow copy
                        config["models"] = list(models.keys())

                    instance_kwargs.update(config)

                    # Validation for generic provider
                    required_fields = ["base_url", "models", "driver"]
                    if not all(field in instance_kwargs for field in required_fields):
                        missing = [
                            f for f in required_fields if f not in instance_kwargs
                        ]
                        raise ValueError(
                            f"Provider '{name}' config is missing required fields: {missing}"
                        )

                    if instance_kwargs.get("driver") not in allowed_drivers:
                        raise ValueError(
                            f"Provider '{name}' has unsupported driver: {instance_kwargs.get('driver')}. Supported: {sorted(allowed_drivers)}"
                        )

                    if (
                        "api_key" not in instance_kwargs
                        and "api_key_env" not in instance_kwargs
                    ):
                        raise ValueError(
                            f"Provider '{name}' config requires 'api_key' or 'api_key_env'"
                        )

                    instance = GenericProvider(**instance_kwargs)
                    self._instances[name] = instance
                    logger.debug(f"Created generic instance for provider: {name}")
                elif provider_class:
                    # Use dedicated provider class
                    instance = provider_class(**instance_kwargs)
                    self._instances[name] = instance
                    logger.debug(f"Created instance for provider: {name}")
                else:
                    logger.warning(f"No class or config found for provider: {name}")

            except Exception as e:
                logger.error(f"Failed to initialize provider {name}: {e!s}")
                # Do not re-raise, allow other providers to load


# Global provider manager instance
provider_manager = ProviderManager()


def register_provider(name: str):
    """Convenience function to register providers"""
    return provider_manager.register_provider(name)


def get_provider(name: str) -> BaseProvider | None:
    """Get a provider instance"""
    return provider_manager.get_provider(name)


def list_providers() -> list[str]:
    """List all registered providers"""
    return provider_manager.list_providers()
