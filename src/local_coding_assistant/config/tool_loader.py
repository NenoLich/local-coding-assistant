"""Tool configuration loading and module importing utilities.

This module handles all file I/O operations, module imports, and configuration
loading for tools. It provides a clean interface for the ToolManager to interact
with tool configurations and modules.
"""

import importlib
import importlib.util
import inspect
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError

if TYPE_CHECKING:
    from yaml.error import MarkedYAMLError

from local_coding_assistant.config.schemas import ToolConfig
from local_coding_assistant.core.exceptions import ConfigError
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("config.tool_loader")


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries, giving precedence to override."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class ToolSourceInfo:
    """Source metadata for a tool definition."""

    file: str
    line: int
    config_type: str


@dataclass
class RawToolDefinition:
    """Intermediate representation of a tool definition before validation."""

    tool_id: str
    config: dict[str, Any] = field(default_factory=dict)
    base_dir: Path | None = None
    source: ToolSourceInfo | None = None

    def merge_entry(
        self,
        entry: dict[str, Any],
        base_dir: Path | None,
        source: ToolSourceInfo,
    ) -> None:
        """Merge a YAML entry into the raw definition."""

        self.config = deep_merge_dicts(self.config, entry)
        if (
            base_dir
            and entry.get("source") != "mcp"
            and ("path" in entry or "module" in entry or self.base_dir is None)
        ):
            self.base_dir = base_dir
        self.source = source

    def to_serializable_mapping(self) -> dict[str, Any]:
        """Return a mapping compatible with legacy loader expectations."""

        data: dict[str, Any] = {
            "config": self.config.copy(),
            "base_dir": self.base_dir,
        }
        if self.source:
            data["_source"] = asdict(self.source)
        return data


class ToolConfigLoader:
    """Handles reading and combining raw tool configuration documents."""

    CONFIG_TYPES = ("default", "local")

    def __init__(self) -> None:
        self._raw_definitions: dict[str, RawToolDefinition] = {}

    def load(self) -> dict[str, dict[str, Any]]:
        """Load raw tool configuration dictionaries from known sources."""
        for config_type in self.CONFIG_TYPES:
            config_path = self._get_config_path(config_type)
            if not config_path or not config_path.exists():
                logger.debug("Tool configuration file for '%s' not found", config_type)
                continue

            for entry, line_number in self._iter_tool_entries(config_path):
                self._consume_entry(entry, config_path, config_type, line_number)

        return {
            tool_id: definition.to_serializable_mapping()
            for tool_id, definition in self._raw_definitions.items()
        }

    def _get_config_path(self, config_type: str) -> Path | None:
        if config_type not in self.CONFIG_TYPES:
            raise ConfigError(f"Unknown tool config type: {config_type}")

        if os.getenv("LOCCA_DEV_MODE"):
            return (
                Path(__file__).resolve().parents[1]
                / "config"
                / f"tools.{config_type}.yaml"
            )

        return (
            Path.home()
            / ".local-coding-assistant"
            / "config"
            / f"tools.{config_type}.yaml"
        )

    def _iter_tool_entries(self, config_path: Path) -> list[tuple[dict[str, Any], int]]:
        try:
            with config_path.open("r", encoding="utf-8") as file_handle:
                config_data = yaml.safe_load(file_handle)
        except yaml.YAMLError as error:
            self._handle_yaml_error(error, config_path)
            return []
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Failed to load tool configuration from %s: %s",
                config_path,
                exc,
                exc_info=True,
            )
            return []

        if not config_data:
            logger.debug("Empty configuration file: %s", config_path)
            return []

        tools_section = config_data.get("tools", [])
        if not isinstance(tools_section, list):
            logger.warning(
                "Invalid tools section in configuration '%s'; expected a list",
                config_path,
            )
            return []

        entries: list[tuple[dict[str, Any], int]] = []
        for index, entry in enumerate(tools_section, start=1):
            if not isinstance(entry, dict):
                logger.warning(
                    "Skipping invalid tool entry at line %d in '%s'; expected a mapping",
                    index + 1,
                    config_path,
                )
                continue
            entries.append((entry.copy(), index + 1))
        return entries

    def _consume_entry(
        self,
        entry: dict[str, Any],
        config_path: Path,
        config_type: str,
        line_number: int,
    ) -> None:
        tool_id = entry.get("id") or entry.get("name")
        if not tool_id:
            logger.warning(
                "Skipping tool entry without an 'id' at line %d in %s",
                line_number,
                config_path,
            )
            return

        if entry.get("source") != "mcp":
            entry.setdefault(
                "source", "builtin" if config_type == "default" else "external"
            )

        definition = self._raw_definitions.setdefault(
            tool_id, RawToolDefinition(tool_id=tool_id)
        )
        definition.merge_entry(
            entry={key: value for key, value in entry.items() if key != "id"},
            base_dir=config_path.parent,
            source=ToolSourceInfo(
                file=str(config_path),
                line=line_number,
                config_type=config_type,
            ),
        )

    @staticmethod
    def _handle_yaml_error(
        error: "MarkedYAMLError | Exception", config_path: Path
    ) -> None:
        if hasattr(error, "problem_mark") and error.problem_mark is not None:
            mark = error.problem_mark
            line = getattr(mark, "line", 0) + 1
            column = getattr(mark, "column", 0) + 1
            logger.error(
                "YAML syntax error in %s at line %d, column %d: %s",
                config_path,
                line,
                column,
                str(error),
            )
        else:  # pragma: no cover - defensive
            logger.error(
                "Error parsing YAML file %s: %s",
                config_path,
                str(error),
                exc_info=not str(error),
            )


class ToolModuleLoader:
    """Responsible for importing tool modules based on configuration."""

    def __init__(self) -> None:
        self._loaded_modules: set[str] = set()
        self._loaded_paths: set[Path] = set()

    def enrich_with_modules(self, tool_configs: dict[str, dict[str, Any]]) -> None:
        for tool_id, tool_data in tool_configs.items():
            config = tool_data.get("config", {})
            if not self._should_load_class(tool_id, config, tool_data):
                continue

            try:
                module = self._load_module(config, tool_data.get("base_dir"))
                if module is None:
                    continue
                tool_class = self._resolve_tool_class(tool_id, config, module)
                self._populate_capabilities(config, tool_class)
                tool_data["tool_class"] = tool_class
                config["available"] = True
                logger.info(
                    "Successfully loaded tool class '%s' for tool '%s'",
                    tool_class.__name__,
                    tool_id,
                )
            except Exception as exc:  # pragma: no cover - logging only
                config["available"] = False
                logger.warning(
                    "Tool '%s' is not available due to: %s",
                    tool_id,
                    exc,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )

    def _should_load_class(
        self,
        tool_id: str,
        config: dict[str, Any],
        tool_data: dict[str, Any],
    ) -> bool:
        is_enabled = config.get("enabled", tool_data.get("enabled", True))
        if not is_enabled:
            logger.debug("Skipping disabled tool: %s", tool_id)
            return False
        if config.get("source") == "mcp":
            logger.debug("Skipping class loading for MCP tool: %s", tool_id)
            return False
        if "path" not in config and "module" not in config:
            logger.debug("No path or module specified for tool: %s", tool_id)
            return False
        return True

    def _load_module(self, config: dict[str, Any], base_dir: Path | None) -> Any | None:
        if "path" in config:
            module_path = self._resolve_path(config["path"], base_dir)
            if not module_path:
                logger.warning("Could not resolve path '%s'", config["path"])
                return None
            return self._import_module_from_path(module_path)

        module_name = config["module"]
        if module_name in self._loaded_modules:
            return importlib.import_module(module_name)

        importlib.import_module(module_name)
        self._loaded_modules.add(module_name)
        return importlib.import_module(module_name)

    def _resolve_tool_class(
        self, tool_id: str, config: dict[str, Any], module: Any
    ) -> type:
        class_name = config.get(
            "tool_class",
            f"{tool_id.replace('_', ' ').title().replace(' ', '')}Tool",
        )
        if hasattr(module, class_name):
            tool_class = getattr(module, class_name)
        else:
            candidates = [
                name
                for name, value in vars(module).items()
                if isinstance(value, type)
                and value.__module__ == module.__name__
                and (name.endswith("Tool") or name == tool_id.replace("_", ""))
            ]
            if not candidates and hasattr(module, "__all__"):
                candidates = [
                    name
                    for name in module.__all__
                    if isinstance(getattr(module, name, None), type)
                ]
            if not candidates:
                available = "None found"
                raise ImportError(
                    f"Module '{module.__name__}' does not contain class '{class_name}'."
                    f" Available classes: {available}"
                )
            class_name = candidates[0]
            tool_class = getattr(module, class_name)
            logger.debug("Using class '%s' for tool '%s'", class_name, tool_id)

        if not isinstance(tool_class, type):
            raise ValueError(f"'{class_name}' is not a class")

        required_methods = ["run"]
        required_attributes = ["Input", "Output"]

        missing_methods = [
            method
            for method in required_methods
            if not callable(getattr(tool_class, method, None))
        ]
        missing_attributes = [
            attr for attr in required_attributes if not hasattr(tool_class, attr)
        ]
        if missing_methods or missing_attributes:
            error_parts = []
            if missing_methods:
                error_parts.append(f"missing methods: {', '.join(missing_methods)}")
            if missing_attributes:
                error_parts.append(
                    f"missing attributes: {', '.join(missing_attributes)}"
                )
            raise ValueError(
                f"Class '{class_name}' is not a valid tool. It must have: {'; '.join(error_parts)}"
            )

        return tool_class

    @staticmethod
    def _populate_capabilities(config: dict[str, Any], tool_class: type) -> None:
        if "is_async" not in config:
            run_method = getattr(tool_class, "run", None)
            config["is_async"] = run_method is not None and inspect.iscoroutinefunction(
                run_method
            )

        if "supports_streaming" not in config:
            stream_method = getattr(tool_class, "stream", None)
            config["supports_streaming"] = stream_method is not None and (
                inspect.iscoroutinefunction(stream_method)
                or inspect.isasyncgenfunction(stream_method)
            )

    def _resolve_path(self, path_value: str, base_dir: Path | None) -> Path | None:
        path = Path(path_value)
        if path.is_absolute():
            return path if path.exists() else None

        if base_dir:
            full_path = (base_dir / path).resolve()
            if full_path.exists():
                return full_path

        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path

        pkg_path = Path(__file__).resolve().parents[2] / path
        if pkg_path.exists():
            return pkg_path

        return None

    def _import_module_from_path(self, module_path: Path) -> Any | None:
        module_path = module_path.resolve()
        if module_path in self._loaded_paths:
            module_name = self._module_name_from_path(module_path)
            return sys.modules.get(module_name)

        if not module_path.exists():
            logger.warning("Tool module path does not exist: %s", module_path)
            return None

        module_name = self._module_name_from_path(module_path)
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if not spec or not spec.loader:
                logger.warning(
                    "Unable to create import spec for module at %s", module_path
                )
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self._loaded_paths.add(module_path)
            return module
        except Exception as exc:  # pragma: no cover - logging only
            logger.error(
                "Failed to import tool module %s: %s", module_path, exc, exc_info=True
            )
            return None

    @staticmethod
    def _module_name_from_path(module_path: Path) -> str:
        if module_path.is_dir():
            init_file = module_path / "__init__.py"
            if not init_file.exists():
                raise ConfigError(
                    f"Directory {module_path} is not a Python package (no __init__.py)"
                )
            module_path = init_file
        parent_dir = str(module_path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        # Using Python's built-in hash() instead of MD5 for non-cryptographic use case
        return f"_locca_tool_{abs(hash(str(module_path)))}"


class ToolConfigConverter:
    """Convert raw tool configuration dictionaries into ToolConfig objects."""

    def convert(self, config_dict: dict[str, dict[str, Any]]) -> dict[str, ToolConfig]:
        tool_configs: dict[str, ToolConfig] = {}
        error_count = 0

        for tool_id, tool_data in config_dict.items():
            config = tool_data.get("config", {})
            base_dir = tool_data.get("base_dir")
            merged_config = config.copy()

            try:
                if config.get("source") == "mcp":
                    tool_configs[tool_id] = self._convert_mcp_tool(tool_id, config)
                    continue

                decorator_metadata = self._get_decorator_metadata(tool_id)
                merged_config = self._merge_decorator_and_yaml_config(
                    merged_config, decorator_metadata
                )
                self._resolve_tool_paths(merged_config, base_dir)

                try:
                    config_copy = merged_config.copy()
                    config_copy.setdefault("enabled", True)
                    if "available" in tool_data.get("config", {}):
                        config_copy["available"] = tool_data["config"]["available"]

                    tool_config = self._create_tool_config(tool_id, config_copy)
                    if "tool_class" in tool_data:
                        tool_config.tool_class = tool_data["tool_class"]
                    tool_configs[tool_id] = tool_config
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(
                        "Failed to create tool config for '%s': %s", tool_id, exc
                    )
                    tool_configs[tool_id] = self._create_unavailable_tool(
                        tool_id, merged_config, exc
                    )

            except (ValidationError, ConfigError) as error:
                error_count += 1
                self._log_tool_processing_error(tool_id, error, tool_data)
                tool_configs[tool_id] = self._create_unavailable_tool(
                    tool_id, merged_config, error
                )
            except Exception as error:  # pragma: no cover - unexpected
                error_count += 1
                self._log_tool_processing_error(tool_id, error, tool_data)

        if error_count > 0:
            logger.warning(
                "Skipped %d invalid tool configuration(s). %d valid tools loaded.",
                error_count,
                len(tool_configs),
            )

        return tool_configs

    @staticmethod
    def _convert_mcp_tool(tool_id: str, config: dict[str, Any]) -> ToolConfig:
        if not config.get("endpoint"):
            raise ConfigError(f"MCP tool '{tool_id}' is missing required 'endpoint'")
        if not config.get("provider"):
            raise ConfigError(f"MCP tool '{tool_id}' is missing required 'provider'")

        mcp_config = config.copy()
        mcp_config.setdefault("enabled", True)
        mcp_config.setdefault("is_async", True)
        return ToolConfigConverter._create_tool_config(tool_id, mcp_config)

    @staticmethod
    def _get_decorator_metadata(tool_id: str) -> dict[str, Any] | None:
        from local_coding_assistant.tools.tool_registry import get_tool_registry

        tool_registry = get_tool_registry()
        if tool_id in tool_registry:
            return ToolConfigConverter._registration_to_metadata(tool_registry[tool_id])

        for registered_id, registration in tool_registry.items():
            if registered_id.lower() == tool_id.lower():
                return ToolConfigConverter._registration_to_metadata(registration)

        return None

    @staticmethod
    def _registration_to_metadata(registration: Any) -> dict[str, Any]:
        return {
            "name": registration.name,
            "description": registration.description,
            "category": registration.category,
            "source": registration.source,
            "permissions": registration.permissions,
            "tags": registration.tags,
            "is_async": registration.is_async,
            "supports_streaming": registration.supports_streaming,
        }

    @staticmethod
    def _merge_decorator_and_yaml_config(
        yaml_config: dict[str, Any], decorator_metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        if decorator_metadata:
            merged = decorator_metadata.copy()
            merged.update(yaml_config)
            return merged
        return yaml_config.copy()

    @staticmethod
    def _resolve_tool_paths(config: dict[str, Any], base_dir: Path | None) -> None:
        if "path" in config and base_dir:
            resolved_path = ToolConfigConverter._resolve_tool_path(
                config["path"], base_dir
            )
            if resolved_path:
                config["path"] = str(resolved_path)
            else:
                raise ConfigError(
                    f"Could not resolve tool path: {config['path']} (base_dir: {base_dir})"
                )

    @staticmethod
    def _resolve_tool_path(path_value: str, base_dir: Path | None) -> Path | None:
        path_obj = Path(path_value)
        if path_obj.is_absolute():
            return path_obj if path_obj.exists() else None

        if base_dir:
            full_path = (base_dir / path_obj).resolve()
            if full_path.exists():
                return full_path

        cwd_path = Path.cwd() / path_obj
        if cwd_path.exists():
            return cwd_path

        pkg_path = Path(__file__).resolve().parents[2] / path_obj
        if pkg_path.exists():
            return pkg_path

        return None

    @staticmethod
    def _create_tool_config(tool_id: str, config: dict[str, Any]) -> ToolConfig:
        config["id"] = tool_id
        config.setdefault("name", tool_id)
        return ToolConfig(**config)

    @staticmethod
    def _create_unavailable_tool(
        tool_id: str, config: dict[str, Any], error: Exception
    ) -> ToolConfig:
        error_config = config.copy()
        error_config.update(
            {
                "available": False,
                "description": f"Error loading tool: {str(error)[:200]}",
            }
        )
        return ToolConfigConverter._create_tool_config(tool_id, error_config)

    @staticmethod
    def _log_tool_processing_error(
        tool_id: str, error: Exception, tool_data: dict[str, Any]
    ) -> None:
        source_info = tool_data.get("_source", {})
        file_info = (
            f" in {source_info.get('file')} (line {source_info.get('line', '?')})"
            if source_info
            else ""
        )

        if isinstance(error, ValidationError):
            logger.warning(
                "Skipping invalid tool configuration '%s'%s: %s",
                tool_id,
                file_info,
                str(error).replace("\n", " "),
            )
        elif isinstance(error, ConfigError):
            logger.warning("Skipping tool '%s': %s", tool_id, error)
        else:  # pragma: no cover - unexpected
            logger.error(
                "Unexpected error processing tool '%s': %s",
                tool_id,
                error,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )


class ToolLoader:
    """Public facade for loading and validating tool configurations."""

    def __init__(self) -> None:
        self._config_loader = ToolConfigLoader()
        self._module_loader = ToolModuleLoader()
        self._converter = ToolConfigConverter()

    def load_tool_configs(self) -> dict[str, ToolConfig]:
        raw_configs = self._config_loader.load()
        self._module_loader.enrich_with_modules(raw_configs)
        tool_configs = self._converter.convert(raw_configs)
        logger.info("Successfully loaded and validated %d tools", len(tool_configs))
        return tool_configs
