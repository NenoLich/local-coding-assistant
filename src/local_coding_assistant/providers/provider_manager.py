"""
Provider manager for dynamic provider loading and registration
"""

import importlib
import importlib.util
from enum import Enum
from pathlib import Path
from typing import Any

from local_coding_assistant.config import EnvManager
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

    def __init__(self, env_manager: EnvManager, allow_test_requests: bool = False):
        self._providers: dict[str, type[BaseProvider]] = {}
        self._instances: dict[str, BaseProvider] = {}
        self._provider_configs: dict[
            str, dict[str, Any]
        ] = {}  # provider_name -> config
        self._auto_discovery_paths = [
            Path(__file__).parent,  # Current providers directory
        ]
        self._env_manager = env_manager
        self._allow_test_requests = allow_test_requests

        # Reference the module-level registry
        self._providers = _provider_registry
        self._provider_sources = _provider_sources

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
        2.5. Global YAML (from providers.default.yaml file directly)
        3. Local config (from providers.local.yaml)
        """
        logger.info("Reloading providers from all sources")

        # Clear current instances and configs
        self._instances.clear()
        self._provider_configs.clear()
        self._provider_sources.clear()

        # Copy fresh from module-level registry
        self._providers = _provider_registry.copy()
        self._provider_sources = _provider_sources.copy()
        logger.debug(
            f"Loaded module-level builtin providers: {list(_provider_registry.keys())}"
        )

        # Layer 2: Load from global config if available
        if config_manager:
            self._load_providers_from_config_layer(
                config_manager, ProviderSource.GLOBAL
            )

        # Layer 2.5: Load from global providers.default.yaml file directly
        self._load_providers_from_global_yaml()

        # Layer 3: Load from local config (this will override any previous sources)
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
                self._process_provider_config(provider_name, provider_dict, source)

        except Exception as e:
            logger.error(f"Failed to load providers from config: {e}")

    def _load_providers_from_global_yaml(self) -> None:
        """Load providers from global providers.default.yaml file directly."""
        try:
            # Use "default" as the config type but pass ProviderSource.GLOBAL for the source
            global_config_path = self._get_config_path("default")
            config = self._load_yaml_config(global_config_path)
            if config is None:
                return

            if not isinstance(config.get("providers"), dict):
                logger.warning(f"Invalid providers section in {global_config_path}")
                return

            for provider_name, provider_config in config["providers"].items():
                if not isinstance(provider_config, dict):
                    logger.warning(
                        f"Invalid config for provider '{provider_name}' in global config"
                    )
                    continue

                # Process the provider configuration using the helper method
                # Use ProviderSource.GLOBAL to indicate this is from the global config
                self._process_provider_config(
                    provider_name, provider_config, ProviderSource.GLOBAL
                )
        except ValueError as e:
            logger.error(f"Error getting global config path: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading global config: {e}")

    def _load_providers_from_local_config(self) -> None:
        """Load providers from local configuration file."""
        try:
            local_config_path = self._get_config_path("local")
            config = self._load_yaml_config(local_config_path)
            if not config or not isinstance(config.get("providers"), dict):
                logger.warning(f"Invalid local config format in {local_config_path}")
                return

            for provider_name, provider_config in config["providers"].items():
                if not isinstance(provider_config, dict):
                    logger.warning(
                        f"Invalid config for provider '{provider_name}' in {local_config_path}"
                    )
                    continue

                self._process_provider_config(
                    provider_name, provider_config, ProviderSource.LOCAL
                )

        except ValueError as e:
            logger.error(f"Error getting local config path: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading local config: {e}")

    def _get_config_path(self, config_type: str) -> Path:
        """Get the path to a configuration file.

        Args:
            config_type: Type of config to get path for ('local' or 'global')

        Returns:
            Path: Path to the requested configuration file

        Raises:
            ValueError: If an invalid config_type is provided
        """
        # Use PathManager to get the config directory and construct the path
        config_dir = self._env_manager.path_manager.get_config_dir()
        return config_dir / f"providers.{config_type}.yaml"

    def _get_config_path_from_src(self, config_type: str) -> Path:
        """Get the path to a configuration file from the project source directory.

        This is primarily used in development mode to load configurations
        directly from the source tree.

        Args:
            config_type: Type of config to get path for ('local' or 'global')

        Returns:
            Path: Path to the requested configuration file in the source tree
        """
        # In development mode, look for configs in the project's config directory
        project_root = self._env_manager.path_manager.get_project_root()
        return project_root / "config" / f"providers.{config_type}.yaml"

    def _load_yaml_config(self, path: Path) -> dict:
        """Load and parse YAML config file."""
        import yaml

        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _process_provider_config(
        self, provider_name: str, provider_config: dict, source: str | ProviderSource
    ) -> None:
        """Process and merge a provider configuration.

        Args:
            provider_name: Name of the provider
            provider_config: Provider configuration dictionary
            source: Source of the configuration (ProviderSource enum or string)
        """
        # Handle models configuration
        if "models" in provider_config:
            provider_config = self._normalize_models_config(provider_config)

        # Get source as string (handles both enum and string inputs)
        if isinstance(source, Enum):
            source_str = source.name.lower()  # Use .name instead of .value for enums
        else:
            source_str = str(source).lower()

        # Merge with existing config
        existing_config = self._provider_configs.get(provider_name, {})
        self._provider_configs[provider_name] = _deep_merge_dicts(
            existing_config, provider_config
        )
        self._provider_sources[provider_name] = source_str
        logger.debug(f"Loaded {source_str} provider config: {provider_name}")

    def _normalize_models_config(self, provider_config: dict) -> dict:
        """Normalize the models configuration format."""
        provider_config = provider_config.copy()
        models_config = provider_config["models"]
        if isinstance(models_config, dict):
            # Convert {model_name: {config}} to list of model configs
            provider_config["models"] = [
                {"name": name, **config} if isinstance(config, dict) else {"name": name}
                for name, config in models_config.items()
            ]
        elif isinstance(models_config, list):
            # Ensure each model has a name field
            provider_config["models"] = [
                model if isinstance(model, dict) else {"name": model}
                for model in models_config
            ]

        return provider_config

    def _instantiate_providers(self) -> None:
        """Instantiate provider instances from loaded configurations."""
        from local_coding_assistant.config.schemas import ModelConfig

        for name, config in self._provider_configs.items():
            try:
                # Skip if already instantiated
                if name in self._instances:
                    continue

                # Make a copy of the config to avoid modifying the original
                config = config.copy()

                # Ensure name is included in the config
                config["name"] = name

                # Convert models to ModelConfig objects if needed
                if "models" in config:
                    config["models"] = [
                        ModelConfig(name=m) if isinstance(m, str) else ModelConfig(**m)
                        for m in config["models"]
                    ]

                # Get provider class (use GenericProvider if not explicitly registered)
                provider_class = self._providers.get(name, GenericProvider)

                # Add env_manager and allow_test_requests to config
                provider_config = config.copy()

                provider_config["env_manager"] = self._env_manager
                provider_config["allow_test_requests"] = self._allow_test_requests

                # Create provider instance
                self._instances[name] = provider_class(**provider_config)
                logger.debug(
                    f"Created instance for provider: {name} "
                    f"(test_requests_allowed={self._allow_test_requests})"
                )

            except Exception as e:
                logger.error(
                    f"Failed to initialize provider {name}: {e}", exc_info=True
                )
                # Continue to next provider instead of failing completely
                continue


# Global provider manager instance
provider_manager = ProviderManager(EnvManager(load_env=True))


def register_provider(name: str):
    """Convenience function to register providers"""
    return provider_manager.register_provider(name)


def get_provider(name: str) -> BaseProvider | None:
    """Get a provider instance"""
    return provider_manager.get_provider(name)


def list_providers() -> list[str]:
    """List all registered providers"""
    return provider_manager.list_providers()
