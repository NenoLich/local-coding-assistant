"""Tool configuration loading and module importing utilities.

This module handles all file I/O operations, module imports, and configuration
loading for tools. It provides a clean interface for the ToolManager to interact
with tool configurations and modules.
"""

import importlib
import importlib.util
import inspect
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

import yaml
from docstring_parser import parse
from pydantic import BaseModel, ValidationError

from local_coding_assistant.config import EnvManager

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

    CONFIG_TYPES = ("default", "local", "sandbox")

    def __init__(self, env_manager: EnvManager | None = None) -> None:
        """Initialize the ToolConfigLoader.

        Args:
            env_manager: Optional EnvManager instance for path resolution
        """
        self._raw_definitions: dict[str, RawToolDefinition] = {}
        self._env_manager = env_manager or EnvManager.create(load_env=True)
        self._path_manager = self._env_manager.path_manager

    def load(
        self, tool_config_paths: list[Path | str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Load raw tool configuration dictionaries from known sources.

        Args:
            tool_config_paths: Optional list of paths to tool configuration files.
                If provided, these paths will be used instead of the default configuration.
                Paths can be strings or Path objects, and will be resolved using the path manager.
        """
        if tool_config_paths:
            # Load from custom paths if provided
            for path in tool_config_paths:
                config_path = self._path_manager.resolve_path(path)
                if not config_path.exists():
                    logger.warning("Tool configuration file not found: %s", config_path)
                    continue
                for entry, line_number in self._iter_tool_entries(config_path):
                    self._consume_entry(entry, config_path, "custom", line_number)
        else:
            # Default behavior: load from CONFIG_TYPES
            for config_type in self.CONFIG_TYPES:
                config_path = self._get_config_path(config_type)
                if not config_path or not config_path.exists():
                    logger.debug(
                        "Tool configuration file for '%s' not found", config_type
                    )
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

        # Use PathManager from EnvManager if available
        if self._path_manager and hasattr(self._path_manager, "resolve_path"):
            if config_type == "default":
                return self._path_manager.resolve_path("@config/tools.default.yaml")
            elif config_type == "sandbox":
                return self._path_manager.resolve_path("@config/sandbox_tools.yaml")
            return self._path_manager.resolve_path("@config/tools.local.yaml")

        raise ConfigError("EnvManager not available (initialization failed)")

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
        if not tools_section:
            logger.warning("No 'tools' section found in config file %s", config_path)
            return []

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
            if config_type == "default":
                entry["source"] = "builtin"
            elif config_type == "sandbox":
                entry["source"] = "sandbox"
            else:
                entry["source"] = "external"

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
    """Responsible for importing tool modules based on configuration.

    This class handles dynamic loading of tool modules with support for environment-aware
    path resolution through the provided EnvManager instance.
    """

    def __init__(self, env_manager: EnvManager | None = None) -> None:
        """Initialize the ToolModuleLoader.

        Args:
            env_manager: EnvManager instance for environment and path management
        """
        self._env_manager = env_manager or EnvManager.create(load_env=False)
        self._path_manager = self._env_manager.path_manager
        self._module_cache: dict[str, Any] = {}

    def enrich_with_modules(self, tool_configs: dict[str, dict[str, Any]]) -> None:
        for tool_id, tool_data in tool_configs.items():
            config = tool_data.get("config", {})

            if not self._should_load_class(tool_id, config, tool_data):
                continue

            try:
                module = self._load_module(tool_id, config, tool_data.get("base_dir"))
                if module is None:
                    continue

                tool_class = self._resolve_tool_class(tool_id, config, module)
                self._populate_capabilities(config, tool_class)

                # Extract parameters from the tool class
                if "parameters" not in config:
                    config["parameters"] = self._retrieve_parameters(tool_class)

                tool_data["tool_class"] = tool_class
                config["tool_class"] = tool_class
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
        if "tool_class" not in config:
            logger.debug("No tool_class specified for tool: %s", tool_id)
            return False
        return True

    def _load_module(
        self, tool_id: str, config: dict[str, Any], base_dir: Path | None
    ) -> Any:
        """Import the Python module that should contain the tool implementation."""
        module_spec = config.get("module")
        path_spec = config.get("path")

        if not module_spec and not path_spec:
            raise ConfigError(
                f"Tool '{tool_id}' must define either 'module' or 'path' to load implementation"
            )

        errors: list[str] = []

        # Try loading via module specification first
        if module_spec:
            try:
                return self._load_module_from_spec(tool_id, module_spec, base_dir)
            except Exception as exc:
                if not isinstance(exc, ConfigError):
                    errors.append(str(exc))
                else:
                    raise

        # Fall back to path-based loading if module loading failed or wasn't specified
        if path_spec:
            try:
                return self._load_module_from_path_spec(tool_id, path_spec, base_dir)
            except Exception as exc:
                if not isinstance(exc, ConfigError):
                    errors.append(str(exc))
                else:
                    raise

        if errors:
            raise ConfigError(
                f"Failed to load module for tool '{tool_id}': " + "; ".join(errors)
            )

        raise ConfigError(
            f"Tool '{tool_id}' module configuration is invalid (no usable module or path)"
        )

    def _load_module_from_spec(
        self, tool_id: str, module_spec: str, base_dir: Path | None
    ) -> Any:
        """Load a module from a module specification."""
        if not isinstance(module_spec, str):
            raise ConfigError(
                f"Module specification for tool '{tool_id}' must be a string"
            )

        if module_spec.startswith("@module/"):
            resolved_path = self._path_manager.resolve_path(
                module_spec, base_dir=base_dir
            )
            return self._import_module_from_path(resolved_path)

        return self._import_module_by_name(module_spec)

    def _load_module_from_path_spec(
        self, tool_id: str, path_spec: str, base_dir: Path | None
    ) -> Any:
        """Load a module from a file path specification."""
        if not isinstance(path_spec, str):
            raise ConfigError(
                f"Path specification for tool '{tool_id}' must be a string"
            )

        resolved_path = self._path_manager.resolve_path(path_spec, base_dir=base_dir)
        return self._import_module_from_path(resolved_path)

    def _import_module_by_name(self, module_name: str) -> Any:
        if module_name in self._module_cache:
            return self._module_cache[module_name]

        module = importlib.import_module(module_name)
        self._module_cache[module_name] = module
        return module

    def _resolve_tool_class(
        self, tool_id: str, config: dict[str, Any], module: Any
    ) -> type:
        """Resolve and validate a tool class from the provided module."""

        if "tool_class" not in config:
            available = (
                ", ".join(
                    name
                    for name, obj in vars(module).items()
                    if isinstance(obj, type) and obj.__module__ == module.__name__
                )
                or "None found"
            )

            raise ValueError(
                f"Tool '{tool_id}' is missing required 'tool_class' in its configuration. "
                f"Available classes in {module.__name__}: {available}"
            )

        class_name = config["tool_class"]

        if not isinstance(class_name, str):
            raise ValueError(
                f"tool_class for tool '{tool_id}' must be provided as a class name string"
            )

        if not hasattr(module, class_name):
            available = (
                ", ".join(
                    name
                    for name, obj in vars(module).items()
                    if isinstance(obj, type) and obj.__module__ == module.__name__
                )
                or "None found"
            )

            raise ImportError(
                f"Module '{module.__name__}' does not contain class '{class_name}'. "
                f"Available classes: {available}"
            )

        tool_class = getattr(module, class_name)

        if not isinstance(tool_class, type):
            raise ValueError(
                f"'{class_name}' in module '{module.__name__}' is not a class"
            )

        if inspect.isabstract(tool_class):
            raise ValueError(
                f"Cannot use abstract class '{class_name}' as a tool implementation"
            )

        required_methods = ["run"]

        missing_methods = [
            method
            for method in required_methods
            if not callable(getattr(tool_class, method, None))
        ]

        error_parts = []
        if missing_methods:
            error_parts.append(f"missing methods: {', '.join(missing_methods)}")

            raise ValueError(
                f"Class '{class_name}' is not a valid tool. It must have: {'; '.join(error_parts)}"
            )

        return tool_class

    def _extract_param_descriptions_from_docstring(
        self, docstring: str
    ) -> dict[str, str]:
        """Extract parameter descriptions from a function's docstring.

        Uses docstring_parser to parse the docstring.

        Args:
            docstring: The docstring to parse

        Returns:
            Dictionary mapping parameter names to their descriptions
        """
        if not docstring:
            return {}

        parsed = parse(docstring)
        return {str(p.arg_name): str(p.description) for p in parsed.params}

    def _retrieve_parameters(self, tool_class: type) -> dict:
        """Extract parameters from either Input model or run() method."""
        # Try to extract from Input model first
        params = self._extract_parameters_from_input_model(tool_class)
        if params is not None:
            return params

        # Fallback to extracting from run() method signature and docstring
        params = self._extract_parameters_from_run_method(tool_class)
        if params is not None:
            return params

        # Ultimate fallback
        return {"type": "object", "properties": {}, "required": []}

    def _extract_parameters_from_input_model(self, tool_class: type) -> dict | None:
        """Extract parameters from a Pydantic Input model if present."""
        try:
            input_field = getattr(tool_class, "Input", None)
            if not (
                isinstance(input_field, type) and issubclass(input_field, BaseModel)
            ):
                return None

            schema = input_field.model_json_schema()
            return self._build_parameter_schema(input_field, schema)

        except Exception as e:
            logger.warning(
                "Failed to extract parameters from Input model: %s",
                str(e),
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return None

    def _build_parameter_schema(
        self, input_model: type[BaseModel], schema: dict
    ) -> dict:
        """Build a parameter schema from a Pydantic model and its JSON schema."""
        properties = {}
        for field_name, field_def in schema.get("properties", {}).items():
            field_info = input_model.model_fields[field_name]
            prop = self.normalize_type(field_info.annotation)

            if field_info.description:
                prop["description"] = field_info.description

            if "enum" in field_def:
                prop["enum"] = field_def["enum"]

            properties[field_name] = prop

        return {
            "type": "object",
            "properties": properties,
            "required": schema.get("required", []),
        }

    def _extract_parameters_from_run_method(self, tool_class: type) -> dict | None:
        """Extract parameters from a run() method signature and docstring."""
        try:
            run_method = getattr(tool_class, "run", None)
            if not (run_method and callable(run_method)):
                return None

            signature = inspect.signature(run_method)
            resolved = get_type_hints(run_method)
            docstring = inspect.getdoc(run_method) or ""
            param_descriptions = self._extract_param_descriptions_from_docstring(
                docstring
            )

            properties = {}
            required = []

            for param_name, param in signature.parameters.items():
                if param_name in ("self", "cls"):
                    continue

                annotation = resolved.get(param_name, str)
                param_info = self._process_parameter(
                    param, annotation, param_descriptions
                )
                properties[param_name] = param_info

                if param.default is param.empty:
                    required.append(param_name)

            return {
                "type": "object",
                "properties": properties,
                "required": required,
            }

        except Exception as e:
            logger.warning(
                "Failed to extract parameters from run() signature: %s", str(e)
            )
            return None

    def _process_parameter(
        self, param: inspect.Parameter, annotation: Any, param_descriptions: dict
    ) -> dict:
        """Process a single parameter from a function signature."""
        param_info = self.normalize_type(annotation)

        if param.name in param_descriptions:
            param_info["description"] = param_descriptions[param.name]

        if param.default is not param.empty:
            param_info["default"] = param.default

        return param_info

    def normalize_type(self, py_type: Any) -> dict:  # noqa: C901
        """
        Normalize Python or Pydantic types into a single JSON Schema type
        suitable for LLM tool invocation.
        """

        origin = get_origin(py_type)
        args = get_args(py_type)

        # ---------------------------------------------------------
        # Optional[T] and Union[T, None] and T | None
        # ---------------------------------------------------------
        if origin in (Union, UnionType):
            non_none = [a for a in args if a is not type(None)]
            nullable = len(non_none) != len(args)

            if not non_none:
                # pure None — fallback
                return {"type": "string", "nullable": True}

            # Take first non-null type deterministically
            inner = self.normalize_type(non_none[0])

            if nullable:
                inner["nullable"] = True

            return inner

        # ------------------------------------------
        # Case 2: Lists / arrays
        # ------------------------------------------
        if origin is list:
            item_type = args[0] if args else Any
            return {"type": "array", "items": self.normalize_type(item_type)}

        # ------------------------------------------
        # Case 3: Dicts / objects
        # ------------------------------------------
        if origin is dict:
            # No strict schema for arbitrary dicts
            return {"type": "object"}

        # ------------------------------------------
        # Case 4: Pydantic model
        # ------------------------------------------
        if isinstance(py_type, type) and issubclass(py_type, BaseModel):
            # Use the model's own json schema
            return py_type.model_json_schema()

        # ------------------------------------------
        # Case 5: Primitive types → JSON types
        # ------------------------------------------
        if py_type is str:
            return {"type": "string"}

        if py_type is int:
            return {"type": "integer"}

        if py_type is float:
            return {"type": "number"}

        if py_type is bool:
            return {"type": "boolean"}

        if py_type is Any:
            return {"type": "string"}  # safest default for LLMs

        # ------------------------------------------
        # Fallback: stringify everything else
        # ------------------------------------------
        return {"type": "string"}

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

    def _resolve_path(self, path_value: str, base_dir: Path | None = None) -> Path:
        """Resolve a path using PathManager.

        Args:
            path_value: Path to resolve (can include @module/ prefix)
            base_dir: Optional base directory for backward compatibility

        Returns:
            Resolved absolute path

        Raises:
            ConfigError: If the path cannot be resolved
        """
        try:
            # Use PathManager for environment-aware path resolution
            if path_value.startswith("@"):
                return self._path_manager.resolve_path(path_value, base_dir=base_dir)

            # Handle relative paths (legacy behavior)
            path = Path(path_value)
            if path.is_absolute():
                return path

            base = base_dir or self._path_manager.get_project_root()
            return (Path(base) / path).resolve()

        except Exception as e:
            raise ConfigError(f"Failed to resolve path '{path_value}': {e}") from e

    def _import_module_from_path(self, module_path: Path) -> Any:
        module_path = module_path.resolve()

        if not module_path.exists():
            raise ConfigError(f"Tool module path does not exist: {module_path}")

        cache_key = str(module_path)
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]

        module_name = self._module_name_from_path(module_path)

        if module_name in sys.modules:
            module = sys.modules[module_name]
            self._module_cache[cache_key] = module
            return module

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ConfigError(f"Could not import module from {module_path}")

        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise ConfigError(f"Error executing module '{module_path}': {exc}") from exc

        sys.modules[module_name] = module
        self._module_cache[cache_key] = module

        logger.debug("Successfully loaded module %s", module_path)
        return module

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
    """Convert raw tool configuration dictionaries into ToolConfig objects.

    This class handles the conversion of raw tool configurations into validated ToolConfig
    objects, including resolving paths using the provided EnvManager.
    """

    def __init__(self, env_manager: EnvManager | None = None) -> None:
        """Initialize the ToolConfigConverter.

        Args:
            env_manager: Optional EnvManager instance for path resolution
        """
        self._env_manager = env_manager or EnvManager.create(load_env=True)
        self._path_manager = self._env_manager.path_manager

    def convert(self, config_dict: dict[str, dict[str, Any]]) -> dict[str, ToolConfig]:
        tool_configs: dict[str, ToolConfig] = {}
        error_count = 0

        for tool_id, tool_data in config_dict.items():
            config = tool_data.get("config", {})
            base_dir = tool_data.get("base_dir")
            merged_config = config.copy()

            tool_config = None

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
                except Exception as exc:
                    error_count += 1
                    logger.warning(
                        "Failed to create tool config for '%s': %s", tool_id, exc
                    )
                    tool_config = self._create_unavailable_tool(
                        tool_id, merged_config, exc
                    )

            except Exception as error:
                error_count += 1
                self._log_tool_processing_error(tool_id, error, tool_data)
                tool_config = self._create_unavailable_tool(
                    tool_id, merged_config, error
                )

            if tool_config:
                tool_configs[tool_id] = tool_config

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

    def _resolve_tool_paths(
        self, config: dict[str, Any], base_dir: Path | None
    ) -> None:
        """Resolve paths in the tool configuration.

        Args:
            config: Tool configuration dictionary
            base_dir: Base directory for relative paths (legacy, prefer @-prefixed paths)

        Raises:
            ConfigError: If a path cannot be resolved
        """
        if "path" not in config:
            return

        path_value = config["path"]
        try:
            resolved_path = self._path_manager.resolve_path(
                path_value, base_dir=base_dir
            )
            if not resolved_path.exists():
                raise ConfigError(f"Could not resolve tool path: {resolved_path}")

            config["path"] = str(resolved_path)

        except Exception as e:
            raise ConfigError(f"Error resolving tool path '{path_value}': {e}") from e

    @staticmethod
    def _create_tool_config(tool_id: str, config: dict[str, Any]) -> ToolConfig:
        config["id"] = tool_id
        config.setdefault("name", tool_id)
        return ToolConfig(**config)

    @staticmethod
    def _create_unavailable_tool(
        tool_id: str, config: dict[str, Any], error: Exception
    ) -> ToolConfig | None:
        """Create a ToolConfig for an unavailable tool.

        Args:
            tool_id: ID of the tool
            config: Original tool config
            error: The error that caused the tool to be unavailable

        Returns:
            ToolConfig for the unavailable tool, or None if we can't create one
        """
        try:
            error_config = config.copy()
            error_config.update(
                {
                    "available": False,
                    "description": f"Error loading tool: {str(error)[:200]}",
                }
            )
            return ToolConfigConverter._create_tool_config(tool_id, error_config)
        except Exception as e:
            logger.warning(
                "Failed to create unavailable tool config for '%s', skipping: %s",
                tool_id,
                str(e),
            )
            return None

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
    """Public facade for loading and validating tool configurations.

    This class coordinates the loading, validation, and conversion of tool configurations
    from various sources, ensuring proper environment-aware behavior.
    """

    def __init__(
        self,
        env_manager: EnvManager | None = None,
        tool_config_paths: list[Path | str] | None = None,
    ) -> None:
        """Initialize the ToolLoader.

        Args:
            env_manager: EnvManager instance for environment and path management
        """
        self._env_manager = env_manager or EnvManager.create(load_env=False)
        self._config_loader = ToolConfigLoader(env_manager=self._env_manager)
        self._module_loader = ToolModuleLoader(env_manager=self._env_manager)
        self._converter = ToolConfigConverter(env_manager=self._env_manager)
        self._tool_config_paths = tool_config_paths

        logger.debug(
            "Initialized ToolLoader with environment: %s",
            self._env_manager.get_environment(),
        )

    def load_tool_configs(self) -> dict[str, ToolConfig]:
        raw_configs = self._config_loader.load(self._tool_config_paths)
        self._module_loader.enrich_with_modules(raw_configs)
        tool_configs = self._converter.convert(raw_configs)
        logger.info("Successfully loaded and validated %d tools", len(tool_configs))
        return tool_configs
