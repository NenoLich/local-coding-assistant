"""
Provider manager for dynamic provider loading and registration
"""

import importlib
import importlib.util
from pathlib import Path
from typing import Any

try:
    from .base import BaseProvider
    from .exceptions import ProviderNotFoundError
except ImportError:
    from base import BaseProvider
    from exceptions import ProviderNotFoundError

from local_coding_assistant.utils.logging import get_logger

logger = get_logger("providers.manager")


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
        if name in self._providers:
            return self._providers[name]

        # Try to auto-discover and import
        return self._auto_discover_provider(name)

    def get_provider(self, name: str, **kwargs) -> BaseProvider | None:
        """Get a provider instance by name"""
        if name in self._instances:
            return self._instances[name]

        # Get provider config if available
        config = self._provider_configs.get(name, {})

        # Merge with kwargs (kwargs take precedence)
        instance_kwargs = {**config, **kwargs}

        provider_class = self.get_provider_class(name)
        if provider_class:
            try:
                instance = provider_class(provider_name=name, **instance_kwargs)
                self._instances[name] = instance
                logger.debug(f"Created instance for provider: {name}")
                return instance
            except Exception as e:
                logger.error(f"Failed to initialize provider {name}: {e!s}")
                raise ProviderNotFoundError(
                    f"Failed to initialize provider {name}: {e!s}"
                ) from e

        return None

    def list_providers(self) -> list[str]:
        """List all registered provider names"""
        # Return all providers: both those with classes and those from config
        all_providers = set(self._providers.keys())
        all_providers.update(self._provider_configs.keys())
        return sorted(list(all_providers))

    def get_provider_source(self, name: str) -> str | None:
        """Get the source of a provider (builtin, global, local)"""
        # Check module-level registry first (for builtin providers)
        if name in _provider_sources:
            return _provider_sources[name]
        # Check instance registry (for config providers)
        if name in self._provider_sources:
            return self._provider_sources.get(name)
        return None

    def _auto_discover_provider(self, name: str) -> type[BaseProvider] | None:
        """Try to auto-discover and import a provider"""
        # Convert name to file/module name
        module_name = f"{name}_provider"

        # Try to import from local providers directory
        try:
            module = importlib.import_module(
                f".{module_name}", package="local_coding_assistant.providers"
            )
            provider_class = getattr(module, f"{name.title()}Provider", None)
            if provider_class and issubclass(provider_class, BaseProvider):
                self._providers[name] = provider_class
                _provider_sources[name] = ProviderSource.BUILTIN
                logger.debug(f"Auto-discovered builtin provider: {name}")
                return provider_class
        except ImportError:
            pass

        # Try custom discovery paths
        for path in self._auto_discovery_paths:
            provider_file = path / f"{module_name}.py"
            if provider_file.exists():
                try:
                    # Dynamically import the module
                    spec = importlib.util.spec_from_file_location(
                        module_name, provider_file
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        provider_class = getattr(
                            module, f"{name.title()}Provider", None
                        )
                        if provider_class and issubclass(provider_class, BaseProvider):
                            self._providers[name] = provider_class
                            _provider_sources[name] = ProviderSource.BUILTIN
                            logger.debug(f"Auto-discovered provider from file: {name}")
                            return provider_class
                except Exception as e:
                    logger.warning(f"Failed to load provider from {provider_file}: {e}")
                    continue

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

        # Clear current instances but keep class registrations
        self._instances.clear()
        self._provider_configs.clear()

        # Copy fresh from module-level registry
        self._providers.update(_provider_registry)
        self._provider_sources = _provider_sources.copy()

        # Layer 2: Load from global config if available
        if config_manager:
            self._load_providers_from_config_layer(
                config_manager, ProviderSource.GLOBAL
            )

        # Layer 2.5: Load from global providers.yaml file directly (fallback if config manager doesn't have it)
        if not self._provider_configs:  # If no providers loaded from config manager
            self._load_providers_from_global_yaml()

        # Layer 3: Load from local config
        self._load_providers_from_local_config()

        # Log the final state
        total_providers = len(self._providers)
        logger.info(f"Provider reload complete: {total_providers} providers loaded")
        for name, source in self._provider_sources.items():
            logger.debug(f"  {name} ({source})")

    def _load_providers_from_config_layer(
        self, config_manager, source: str
    ) -> dict[str, Any]:
        """Load providers from a specific config layer."""
        try:
            config = config_manager.global_config
            if not config or not hasattr(config, "providers"):
                return {}

            providers_config = config.providers
            if not isinstance(providers_config, dict):
                logger.warning("Invalid providers configuration format")
                return {}

            loaded_configs = {}
            for provider_name, provider_config in providers_config.items():
                # Convert ProviderConfig to dict if needed
                if hasattr(provider_config, "model_dump"):
                    provider_dict = provider_config.model_dump()
                else:
                    provider_dict = provider_config

                # Register the provider configuration
                self._provider_configs[provider_name] = provider_dict
                _provider_sources[provider_name] = source

                logger.debug(f"Loaded {source} provider config: {provider_name}")

            return loaded_configs

        except Exception as e:
            logger.error(f"Failed to load providers from config: {e}")
            return {}

    def _load_providers_from_global_yaml(self) -> dict[str, Any]:
        """Load providers from global providers.yaml file directly."""
        global_config_path = Path(__file__).parent.parent / "config" / "providers.yaml"

        if not global_config_path.exists():
            logger.debug(
                f"Global providers.yaml file does not exist: {global_config_path}"
            )
            return {}

        try:
            import yaml

            with open(global_config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            if not isinstance(config, dict) or "providers" not in config:
                logger.warning(
                    f"Invalid global providers.yaml format in {global_config_path}"
                )
                return {}

            providers_config = config["providers"]
            if not isinstance(providers_config, dict):
                logger.warning(f"Invalid providers section in {global_config_path}")
                return {}

            loaded_configs = {}
            for provider_name, provider_config in providers_config.items():
                if isinstance(provider_config, dict):
                    self._provider_configs[provider_name] = provider_config
                    _provider_sources[provider_name] = ProviderSource.GLOBAL
                    logger.debug(f"Loaded global provider config: {provider_name}")

            return loaded_configs

        except Exception as e:
            logger.error(
                f"Failed to load global providers from {global_config_path}: {e}"
            )
            return {}

    def _load_providers_from_local_config(self) -> dict[str, Any]:
        """Load providers from local configuration file."""
        local_config_path = (
            Path.home() / ".local-coding-assistant" / "config" / "providers.local.yaml"
        )

        if not local_config_path.exists():
            logger.debug(f"Local config file does not exist: {local_config_path}")
            return {}

        try:
            import yaml

            with open(local_config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            if not isinstance(config, dict):
                logger.warning(f"Invalid local config format in {local_config_path}")
                return {}

            loaded_configs = {}
            for provider_name, provider_config in config.items():
                if isinstance(provider_config, dict):
                    self._provider_configs[provider_name] = provider_config
                    _provider_sources[provider_name] = ProviderSource.LOCAL
                    logger.debug(f"Loaded local provider config: {provider_name}")

            return loaded_configs

        except Exception as e:
            logger.error(
                f"Failed to load local providers from {local_config_path}: {e}"
            )
            return {}

    def load_providers_from_config(self, config: dict[str, Any]) -> list[BaseProvider]:
        """Load providers from configuration"""
        providers = []

        for provider_name, provider_config in config.items():
            try:
                provider = self.get_provider(provider_name, **provider_config)
                if provider:
                    providers.append(provider)
            except Exception as e:
                # Log error but continue with other providers
                logger.error(f"Failed to load provider {provider_name}: {e!s}")

        return providers

    def clear_cache(self):
        """Clear provider instances cache"""
        self._instances.clear()
        logger.debug("Cleared provider instances cache")


# Global provider manager instance
provider_manager = ProviderManager()


def register_provider(name: str):
    """Convenience function to register providers"""
    return provider_manager.register_provider(name)


def get_provider(name: str, **kwargs) -> BaseProvider | None:
    """Get a provider instance"""
    return provider_manager.get_provider(name, **kwargs)


def list_providers() -> list[str]:
    """List all registered providers"""
    return provider_manager.list_providers()
