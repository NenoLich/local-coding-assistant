"""Enhanced tool management with registry, async, and streaming support.

This module provides the ToolManager class which handles tool registration,
execution, and discovery. It works with the ToolLoader for configuration
and module loading.
"""

import asyncio
import inspect
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
)

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from local_coding_assistant.core.exceptions import ToolRegistryError
from local_coding_assistant.core.protocols import IToolManager
from local_coding_assistant.tools.base import Tool
from local_coding_assistant.tools.types import (
    ToolCategory,
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolInfo,
    ToolSource,
)
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.core.protocols import IConfigManager

logger = get_logger("tools.tool_manager")


class ToolInputTransformer:
    """Transforms input data to match a tool's expected format.

    This class handles common input transformations to make tool usage more flexible.
    """

    @classmethod
    def transform(
        cls,
        input_data: Any,
        tool_info: ToolInfo | None = None,
        tool_instance: Any | None = None,
    ) -> Any:
        """Transform input data to match the tool's expected format.

        Args:
            input_data: The input data to transform
            tool_info: Optional ToolInfo for the tool
            tool_instance: Optional tool instance for inspection

        Returns:
            Transformed input data
        """
        if input_data is None or not (tool_info or tool_instance):
            return input_data

        # Try to get the input model from the tool instance
        input_model = None
        if tool_instance and hasattr(tool_instance, "Input"):
            input_model = tool_instance.Input

        # If we have an input model, use it to guide the transformation
        if input_model and hasattr(input_model, "model_fields"):
            return cls._transform_with_model(input_data, input_model)

        # Fall back to tool info if available
        if (
            tool_info
            and hasattr(tool_info, "tool_class")
            and hasattr(tool_info.tool_class, "Input")
        ):
            input_model = tool_info.tool_class.Input
            if hasattr(input_model, "model_fields"):
                return cls._transform_with_model(input_data, input_model)

        return input_data

    @classmethod
    def _get_multi_value_handler(
        cls, input_data: Any
    ) -> Callable[[str], list[Any]] | None:
        """Get a function to handle multi-value form data if available.

        Returns:
            A callable that takes a field name and returns a list of values, or None if no
            suitable handler is found.
        """
        if hasattr(input_data, "getall") and callable(input_data.getall):
            return (
                lambda key: input_data.getall(key)
                if hasattr(input_data, "getall")
                else []
            )
        if hasattr(input_data, "getlist") and callable(input_data.getlist):
            return (
                lambda key: input_data.getlist(key)
                if hasattr(input_data, "getlist")
                else []
            )
        return None

    @classmethod
    def _process_multi_value_field(
        cls,
        data: dict[str, Any],
        field_name: str,
        field_info: FieldInfo,
        get_multi_value: Callable[[str], list[Any]] | None,
    ) -> None:
        """Process a field that might have multiple values."""
        if field_name not in data or not isinstance(data[field_name], str):
            return

        # Get all values for this field (handles multiple parameters with same name)
        if get_multi_value:
            values = get_multi_value(field_name)
        else:
            values = [data[field_name]]

        if len(values) <= 1:
            return

        # Convert all values to the appropriate type
        try:
            if cls._is_list_or_tuple_field(field_info):
                data[field_name] = [float(v) for v in values]
            else:
                data[field_name] = values[0] if len(values) == 1 else values
        except (ValueError, TypeError):
            # If conversion fails, keep the original value
            pass

    @classmethod
    def _transform_dict_input(
        cls, input_data: dict, model_fields: dict[str, FieldInfo]
    ) -> dict:
        """Transform dictionary input according to the model."""
        data = dict(input_data)
        get_multi_value = cls._get_multi_value_handler(input_data)

        # Process each field in the model
        for field_name, field_info in model_fields.items():
            if field_name in data and isinstance(data[field_name], list):
                # Already in the correct format
                continue
            cls._process_multi_value_field(
                data, field_name, field_info, get_multi_value
            )

        # Handle positional arguments if present
        if "args" in data and isinstance(data["args"], list) and data["args"]:
            return cls._transform_positional_args(data, model_fields)

        return data

    @classmethod
    def _transform_sequence_input(
        cls, input_data: list | tuple, model_fields: dict[str, FieldInfo]
    ) -> dict:
        """Transform sequence input (list/tuple) into a dictionary."""
        return cls._transform_positional_args({"args": input_data}, model_fields)

    @classmethod
    def _transform_with_model(
        cls, input_data: Any, input_model: type[BaseModel]
    ) -> Any:
        """Transform input data using the provided Pydantic model."""
        model_fields = getattr(input_model, "model_fields", {})

        if isinstance(input_data, dict):
            return cls._transform_dict_input(input_data, model_fields)
        if isinstance(input_data, list | tuple):
            return cls._transform_sequence_input(input_data, model_fields)

        return input_data

    @classmethod
    def _handle_numbers_field(cls, args: list) -> dict[str, list[float]] | None:
        """Handle case where tool expects a 'numbers' field with positional args."""
        try:
            return {"numbers": [float(arg) for arg in args]}
        except (ValueError, TypeError):
            return None

    @classmethod
    def _handle_single_field(
        cls, args: list, field_name: str, field_info: FieldInfo
    ) -> dict[str, Any]:
        """Handle case where tool has a single field that can accept a list."""
        # If the field is a list/tuple, pass all args as a list
        if cls._is_list_or_tuple_field(field_info):
            return {field_name: list(args)}

        # If there's only one arg, pass it directly
        if len(args) == 1:
            return {field_name: args[0]}

        # Otherwise, pass all args as a list (may raise validation error)
        return {field_name: list(args)}

    @classmethod
    def _map_args_to_fields(
        cls, args: list, model_fields: dict[str, FieldInfo]
    ) -> dict[str, Any]:
        """Map positional arguments to field names by position."""
        result = {}
        for i, (field_name, field_info) in enumerate(model_fields.items()):
            if i < len(args):
                result[field_name] = args[i]
            elif hasattr(field_info, "default") and field_info.default is not None:
                # For missing args, use default if available
                result[field_name] = field_info.default
        return result

    @classmethod
    def _transform_positional_args(
        cls, input_data: dict[str, Any], model_fields: dict[str, FieldInfo]
    ) -> dict[str, Any]:
        """Transform positional arguments to named fields based on the model."""
        args = input_data.get("args", [])
        if not args or not model_fields:
            return input_data

        # Case 1: Tool expects a 'numbers' field but got positional args
        if "numbers" in model_fields and cls._is_list_or_tuple_field(
            model_fields["numbers"]
        ):
            if result := cls._handle_numbers_field(args):
                return result

        # Case 2: Tool has a single field that can accept a list
        if len(model_fields) == 1:
            field_name = next(iter(model_fields))
            return cls._handle_single_field(args, field_name, model_fields[field_name])

        # Case 3: Try to map positional args to field names by position
        return cls._map_args_to_fields(args, model_fields)

    @staticmethod
    def _is_list_or_tuple_field(field_info: FieldInfo) -> bool:
        """Check if a field is a list or tuple type."""
        if not hasattr(field_info, "annotation"):
            return False

        annotation = field_info.annotation

        # Handle direct type annotation
        if annotation in (list, list, tuple, tuple):
            return True

        # Handle generic types like List[int]
        if hasattr(annotation, "__origin__"):
            return annotation.__origin__ in (list, list, tuple, tuple)

        return False


@dataclass
class ToolRuntime:
    """Runtime wrapper encapsulating execution and validation behavior for a tool."""

    info: ToolInfo
    instance: Any
    kind: str  # "tool", "callable", or "mcp"
    run_is_async: bool
    supports_streaming: bool
    has_input_validation: bool
    has_output_validation: bool

    async def execute(self, payload: dict[str, Any]) -> Any:
        """Execute the tool with the provided payload."""
        input_data = self._prepare_input(payload)
        result = await self._invoke_run(input_data)
        return self._normalize_output(result)

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream results from the tool if supported."""
        if self.kind != "tool":
            raise ToolRegistryError(
                f"Tool '{self.info.name}' does not support streaming operations"
            )

        if not self.supports_streaming:
            raise ToolRegistryError(
                f"Tool '{self.info.name}' does not support streaming"
            )

        input_data = self._prepare_input(payload)
        async for chunk in self._invoke_stream(input_data):
            yield self._normalize_output(chunk)

    def _transform_payload(self, payload: Any) -> Any:
        """Transform input payload to match the expected format."""
        if self.kind != "tool" or not self.has_input_validation:
            return payload

        try:
            return ToolInputTransformer.transform(
                payload, tool_info=self.info, tool_instance=self.instance
            )
        except Exception as e:
            logger.warning(
                "Error transforming input for tool '%s': %s",
                self.info.name,
                str(e),
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return payload

    def _validate_payload_against_model(
        self, payload: Any, input_model: type[BaseModel]
    ) -> Any:
        """Validate and convert payload against the input model."""
        if not isinstance(input_model, type):
            raise ValueError(
                f"Expected a class/type for input_model, got {type(input_model).__name__}"
            )

        # If payload is already an instance of the model, return as-is
        if isinstance(payload, input_model):
            return payload

        # Convert payload to dict if it's not already
        if not isinstance(payload, dict):
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump()
            elif hasattr(payload, "dict"):
                payload = payload.dict()
            elif isinstance(payload, str | bytes | bytearray):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError as err:
                    raise ValueError(
                        "Could not parse string/bytes payload as JSON"
                    ) from err

        # Use Pydantic's model_validate
        return input_model.model_validate(payload)

    def _format_validation_error(self, exc: ValidationError) -> str:
        """Format validation error messages into a user-friendly string."""
        errors = []
        for error in exc.errors():
            loc = ".".join(str(loc_part) for loc_part in error["loc"])
            msg = error["msg"]
            errors.append(f"{loc}: {msg}")
        return "\n  - " + "\n  - ".join(errors)

    def _prepare_input(self, payload: Any) -> Any:
        """Validate and transform tool input."""
        # Early return for non-tools or tools without input validation
        if self.kind != "tool" or not self.has_input_validation:
            return payload

        # Transform the input to match the expected format
        payload = self._transform_payload(payload)

        # Get the input model if available
        input_model = getattr(self.instance, "Input", None)
        if input_model is None:
            return payload

        # Validate and convert the payload
        try:
            return self._validate_payload_against_model(payload, input_model)
        except ValidationError as exc:
            error_msg = self._format_validation_error(exc)
            raise ToolRegistryError(
                f"Invalid input for tool '{self.info.name}':{error_msg}"
            ) from exc
        except (ValueError, AttributeError) as exc:
            raise ToolRegistryError(
                f"Invalid input for tool '{self.info.name}': {exc}"
            ) from exc

    async def _invoke_run(self, input_data: Any) -> Any:
        """Invoke the underlying tool execution."""
        if self.kind == "tool":
            runner = getattr(self.instance, "run", None)
            if runner is None:
                raise ToolRegistryError(
                    f"Tool '{self.info.name}' is missing a run implementation"
                )

            if self.run_is_async:
                return await runner(input_data)

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, runner, input_data)

        if self.kind == "mcp":
            if not isinstance(input_data, dict):
                raise ToolRegistryError(
                    f"MCP tool '{self.info.name}' expects dictionary payloads"
                )

            runner = getattr(self.instance, "execute", None)
            if runner is None:
                raise ToolRegistryError(
                    f"MCP tool '{self.info.name}' is missing an execute implementation"
                )

            if asyncio.iscoroutinefunction(runner):
                return await runner(**input_data)

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, runner, input_data)

        # Fallback for plain callables
        runner = self.instance
        if asyncio.iscoroutinefunction(runner):
            return await runner(input_data)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, runner, input_data)

    async def _invoke_stream(self, input_data: Any) -> AsyncIterator[Any]:
        """Invoke the streaming interface on the tool."""
        stream_method = getattr(self.instance, "stream", None)
        if stream_method is None:
            raise ToolRegistryError(
                f"Tool '{self.info.name}' does not implement streaming"
            )

        stream_obj = stream_method(input_data)

        if inspect.isawaitable(stream_obj) and not inspect.isasyncgen(stream_obj):
            stream_obj = await stream_obj

        if hasattr(stream_obj, "__aiter__"):
            async for chunk in stream_obj:
                yield chunk
            return

        if inspect.isasyncgen(stream_obj):
            async for chunk in stream_obj:
                yield chunk
            return

        raise ToolRegistryError(
            f"Streaming tool '{self.info.name}' must return an async iterator"
        )

    def _convert_to_dict(self, obj: Any) -> dict:
        """Convert an object to a dictionary using the appropriate method.

        Args:
            obj: The object to convert to a dictionary

        Returns:
            The object converted to a dictionary
        """
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return self._coerce_to_serializable(obj)

    def _validate_with_model(self, output_model: type[BaseModel], result: Any) -> Any:
        """Validate the result against the output model.

        Args:
            output_model: The Pydantic model to validate against
            result: The result to validate

        Returns:
            The validated result as a dictionary
        """
        if result is None:
            validated = output_model()
        elif isinstance(result, dict):
            validated = output_model.model_validate(result)
        elif hasattr(result, "model_dump"):
            validated = output_model.model_validate(result.model_dump())
        elif hasattr(result, "dict"):
            validated = output_model.model_validate(result.dict())
        else:
            # Last resort - try to convert to dict
            try:
                validated = output_model.model_validate(dict(result))
            except (TypeError, ValueError):
                # If we can't convert to dict, try to create with the raw value
                validated = output_model.model_validate({"result": result})

        return self._convert_to_dict(validated)

    def _normalize_output(self, result: Any) -> Any:
        """Normalize tool outputs to dictionaries when appropriate.

        Args:
            result: The raw result from the tool execution

        Returns:
            The normalized result (usually a dictionary for Pydantic models)

        Raises:
            ToolRegistryError: If output validation fails
        """
        # If we don't have output validation, or it's not a tool, just make it serializable
        if self.kind != "tool" or not self.has_output_validation:
            return self._coerce_to_serializable(result)

        # Get the output model from the tool
        output_model = getattr(self.instance, "Output", None)
        if output_model is None:
            return self._coerce_to_serializable(result)

        # Handle case where result is already the correct model
        if isinstance(result, output_model):
            return self._convert_to_dict(result)

        try:
            return self._validate_with_model(output_model, result)
        except (ValidationError, ValueError, TypeError) as exc:
            logger.error(
                "Output validation failed for tool '%s': %s",
                self.info.name,
                str(exc),
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            # Return the original result if validation fails
            return self._coerce_to_serializable(result)

    @staticmethod
    def _coerce_to_serializable(result: Any) -> Any:
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "dict"):
            return result.dict()
        return result


class ToolManager(IToolManager, Iterable[Any]):
    """Enhanced tool manager with comprehensive logging, error handling, and validation.

    This class provides:
    - Automatic tool discovery via registry
    - Async/sync execution support
    - Permission checking
    - Backward compatibility
    """

    def __init__(
        self,
        config_manager: "IConfigManager",
        auto_load: bool = True,
        *,
        auto_load_registry: bool | None = None,
    ) -> None:
        """Initialize the tool manager.

        Args:
            config_manager: IConfigManager instance for tool configurations.
                If not provided, a default one will be created.
            auto_load: Whether to automatically load tools
            auto_load_registry: Legacy flag for tests/backwards compatibility
        """
        self._tools: dict[str, ToolInfo] = {}
        self._runtimes: dict[str, ToolRuntime] = {}
        self._execution_stats: dict[str, dict[str, Any]] = {}
        self._config_manager: IConfigManager = config_manager

        if auto_load_registry is not None:
            auto_load = auto_load_registry

        if auto_load:
            self.load_tools()

        logger.info("ToolManager initialized with %d tools", len(self._tools))

    def _run_sync(self, awaitable: Awaitable[Any]) -> Any:
        """Safely execute an awaitable from synchronous contexts.

        Args:
            awaitable: An awaitable object to execute

        Returns:
            The result of the awaitable

        Raises:
            RuntimeError: If called from within a running event loop
        """
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None:
            raise RuntimeError(
                "Attempted synchronous execution while an event loop is running. "
                "Use the asynchronous interface instead."
            )

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)

            if inspect.iscoroutine(awaitable):
                coroutine = awaitable
            else:

                async def run_awaitable() -> Any:
                    return await awaitable

                coroutine = run_awaitable()

            return loop.run_until_complete(coroutine)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def _create_mcp_tool_executor(self, tool_info: ToolInfo) -> Any:
        """Create a dummy MCP tool executor.

        Args:
            tool_info: The MCP tool's information

        Returns:
            An instance of MCPToolExecutor

        Note:
            This is a temporary implementation that will be replaced with actual MCP client integration.
        """

        class MCPToolExecutor:
            def __init__(self, _tool_info: ToolInfo):
                self.tool_info = _tool_info
                self.endpoint = _tool_info.endpoint
                self.provider = _tool_info.provider

            async def execute(self, **kwargs) -> dict[str, Any]:
                """Execute the MCP tool with the given arguments."""
                # This is a dummy implementation that will be replaced with actual MCP client calls
                logger.info(
                    "[MCP Tool] Executing %s via %s at %s with args: %s",
                    self.tool_info.name,
                    self.provider,
                    self.endpoint,
                    kwargs,
                )

                # Simulate network delay
                await asyncio.sleep(0.5)

                # Return a dummy response
                return {
                    "status": "success",
                    "result": f"MCP tool '{self.tool_info.name}' executed successfully",
                    "provider": self.provider,
                    "endpoint": self.endpoint,
                    "arguments": kwargs,
                }

        return MCPToolExecutor(tool_info)

    def load_tools(self) -> None:
        """Load tools using ConfigManager.

        This will:
        1. Clear any existing tool instances and stats
        2. Get the latest tool configurations from ConfigManager
        3. Initialize tool instances, including MCP tools

        Raises:
            RuntimeError: If there's an error loading tools
        """
        self._tools.clear()
        self._runtimes.clear()
        self._execution_stats.clear()

        try:
            # Get tool configs from ConfigManager
            tool_configs = self._config_manager.get_tools()
            logger.info("Loaded %d tool configurations", len(tool_configs))

            # Initialize tool instances
            for tool_id, tool_config in tool_configs.items():
                try:
                    # Convert ToolConfig to ToolInfo for runtime use
                    if hasattr(tool_config, "to_tool_info"):
                        tool_info = tool_config.to_tool_info()
                    else:
                        raise ToolRegistryError(
                            f"Failed to load tool '{tool_id}': ToolConfig does not have a 'to_tool_info' method"
                        )

                    # Always store the tool info, even for disabled or unavailable tools
                    self._tools[tool_id] = tool_info

                    # Skip instance creation for disabled tools
                    if not tool_config.enabled:
                        logger.debug(
                            "Skipping instance creation for disabled tool: %s", tool_id
                        )
                        self._execution_stats[tool_id] = self._create_stats_bucket()
                        continue

                    # Skip instance creation for unavailable tools
                    if not tool_config.available:
                        logger.debug(
                            "Skipping instance creation for unavailable tool: %s",
                            tool_id,
                        )
                        self._execution_stats[tool_id] = self._create_stats_bucket()
                        continue

                    runtime = self._build_runtime(tool_id, tool_info)
                    self._runtimes[tool_id] = runtime
                    self._execution_stats[tool_id] = self._create_stats_bucket()
                    logger.debug("Successfully initialized tool runtime: %s", tool_id)

                except Exception as e:
                    logger.error(
                        "Failed to initialize tool %s: %s",
                        tool_id,
                        str(e),
                        exc_info=True,
                    )

        except Exception as e:
            logger.error("Error loading tools: %s", str(e), exc_info=True)
            raise RuntimeError(f"Failed to load tools: {e}") from e

    def reload_tools(self) -> None:
        """Reload tools from configuration files.

        This will:
        1. Tell ConfigManager to reload tool configurations
        2. Re-initialize all tool instances

        Raises:
            RuntimeError: If there's an error reloading tools
        """
        try:
            # Reload tools through ConfigManager
            self._config_manager.reload_tools()

            # Re-initialize tools
            self.load_tools()

            logger.info("Successfully reloaded %d tools", len(self._tools))

        except Exception as e:
            logger.error("Error reloading tools: %s", str(e), exc_info=True)
            raise RuntimeError(f"Failed to reload tools: {e}") from e

    # Tool loading methods have been moved to ToolLoader class

    def __iter__(self) -> Iterator[tuple[str, ToolInfo]]:
        """Iterate over registered tools.

        Yields:
            Tuples of (tool_name, tool_info) for each registered tool
        """
        yield from self._tools.items()

    def __len__(self) -> int:
        """Return the number of registered tools.

        Returns:
            Number of registered tools
        """
        return len(self._tools)

    def invoke(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Legacy method for backward compatibility. Same as run_tool.

        Args:
            tool_name: Name of the tool to execute
            payload: Input parameters for the tool

        Returns:
            The tool's execution result

        Raises:
            ToolRegistryError: If tool execution fails
        """
        return self.run_tool(tool_name, payload)

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        """Execute a tool using a ToolExecutionRequest and return ToolExecutionResponse.

        Args:
            request: ToolExecutionRequest containing tool name and payload.

        Returns:
            ToolExecutionResponse with execution results.
        """
        tool_name = request.tool_name
        start_time = time.perf_counter()

        try:
            runtime = self._get_runtime(tool_name)
            result = self._run_sync(runtime.execute(request.payload))
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._record_success(tool_name, execution_time_ms / 1000.0)

            return ToolExecutionResponse(
                success=True,
                result=result,
                tool_name=tool_name,
                execution_time_ms=execution_time_ms,
            )
        except Exception as exc:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._record_error(tool_name, execution_time_ms / 1000.0, exc)
            error_msg = f"Error executing tool '{tool_name}': {exc}"
            logger.exception(error_msg)

            return ToolExecutionResponse(
                success=False,
                tool_name=tool_name,
                error_message=error_msg,
                execution_time_ms=execution_time_ms,
            )

    def get_tool_info(self, tool_name: str) -> ToolInfo | None:
        """Get detailed information about a registered tool.

        Args:
            tool_name: Name of the tool to get info for.

        Returns:
            ToolInfo object with tool details, or None if not found.
        """
        return self._tools.get(tool_name)

    def list_tools(self, category: str | ToolCategory | None = None) -> list[ToolInfo]:
        """List all registered tools, optionally filtered by category.

        Args:
            category: Optional category to filter tools by (can be string or ToolCategory).
                     If the category doesn't exist, returns an empty list.

        Returns:
            List of ToolInfo objects.
        """
        if category is None:
            return list(self._tools.values())

        # Convert string category to ToolCategory if needed
        if isinstance(category, str):
            try:
                category = ToolCategory(category)
            except ValueError:
                # For unknown categories, return an empty list
                return []

        return [info for info in self._tools.values() if info.category == category]

    def get_execution_stats(self) -> dict[str, dict[str, Any]]:
        """Get execution statistics for all tools."""
        return self._execution_stats.copy()

    def run_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Run a tool with the given parameters.

        Args:
            tool_name: Name of the tool to run
            parameters: Parameters to pass to the tool
            **kwargs: Additional keyword arguments

        Returns:
            The result of the tool execution

        Raises:
            ToolRegistryError: If the tool is not found, not enabled, not available, or execution fails
        """
        return self._run_sync(self.run_tool_async(tool_name, parameters, **kwargs))

    async def run_tool_async(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Run a tool asynchronously with the given parameters.

        Args:
            tool_name: Name of the tool to run
            parameters: Parameters to pass to the tool
            **kwargs: Additional keyword arguments

        Returns:
            The result of the tool execution

        Raises:
            ToolRegistryError: If the tool is not found, not enabled, not available, or execution fails
        """
        runtime = self._get_runtime(tool_name)

        start_time = time.perf_counter()
        try:
            result = await runtime.execute(parameters)
            execution_time = time.perf_counter() - start_time
            self._record_success(tool_name, execution_time)
            return result
        except Exception as exc:
            execution_time = time.perf_counter() - start_time
            self._record_error(tool_name, execution_time, exc)
            raise ToolRegistryError(
                f"Error executing tool '{tool_name}': {exc}"
            ) from exc

    def _create_stats_bucket(self) -> dict[str, Any]:
        return {
            "call_count": 0,
            "error_count": 0,
            "last_called": None,
            "average_duration": 0.0,
            "last_error": None,
            "last_error_time": None,
        }

    def _ensure_stats_bucket(self, tool_name: str) -> dict[str, Any]:
        return self._execution_stats.setdefault(tool_name, self._create_stats_bucket())

    def _record_invocation(self, stats: dict[str, Any], duration_sec: float) -> None:
        stats["call_count"] += 1
        stats["last_called"] = time.time()
        stats["average_duration"] = (
            duration_sec
            if stats["average_duration"] == 0.0
            else 0.8 * stats["average_duration"] + 0.2 * duration_sec
        )

    def _record_success(self, tool_name: str, duration_sec: float) -> None:
        stats = self._ensure_stats_bucket(tool_name)
        self._record_invocation(stats, duration_sec)

    def _record_error(
        self, tool_name: str, duration_sec: float, error: Exception
    ) -> None:
        stats = self._ensure_stats_bucket(tool_name)
        self._record_invocation(stats, duration_sec)
        stats["error_count"] += 1
        stats["last_error"] = str(error)
        stats["last_error_time"] = time.time()

        logger.error(
            "Error in tool '%s': %s",
            tool_name,
            str(error),
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )

    def _build_runtime(self, tool_id: str, tool_info: "ToolInfo") -> "ToolRuntime":
        """Build a ToolRuntime instance from a tool configuration.

        Args:
            tool_id: Unique identifier for the tool
            tool_info: Tool information

        Returns:
            A configured ToolRuntime instance

        Raises:
            ToolRegistryError: If there's an error building the runtime
        """
        try:
            # Determine the tool kind based on source
            if tool_info.source == ToolSource.MCP:
                tool_instance = self._create_mcp_tool_executor(tool_info)
                kind = "mcp"
            else:
                # The tool class should already be imported by ToolLoader
                if not hasattr(tool_info, "tool_class") or tool_info.tool_class is None:
                    raise ToolRegistryError(
                        f"Tool '{tool_id}' is missing tool class. "
                        "Ensure ToolLoader is properly loading the tool class."
                    )

                try:
                    tool_instance = tool_info.tool_class()
                    kind = "tool"
                except Exception as e:
                    logger.error(
                        "Failed to instantiate tool class for '%s': %s",
                        tool_id,
                        str(e),
                        exc_info=logger.isEnabledFor(logging.DEBUG),
                    )
                    raise ToolRegistryError(
                        f"Failed to instantiate tool '{tool_id}': {e!s}"
                    ) from e

            # Use the pre-computed values from config
            run_is_async = tool_info.is_async
            supports_streaming = tool_info.supports_streaming

            # Check for input/output validation models
            has_input_validation = (
                hasattr(tool_instance, "Input")
                and issubclass(tool_instance.Input, BaseModel)
                if hasattr(tool_instance, "Input")
                else False
            )

            has_output_validation = (
                hasattr(tool_instance, "Output")
                and issubclass(tool_instance.Output, BaseModel)
                if hasattr(tool_instance, "Output")
                else False
            )

            # Create and return the runtime
            return ToolRuntime(
                info=tool_info,
                instance=tool_instance,
                kind=kind,
                run_is_async=run_is_async,
                supports_streaming=supports_streaming,
                has_input_validation=has_input_validation,
                has_output_validation=has_output_validation,
            )

        except Exception as e:
            logger.error(
                "Error building runtime for tool '%s': %s",
                tool_id,
                str(e),
                exc_info=True,
            )
            raise ToolRegistryError(
                f"Failed to build runtime for tool '{tool_id}': {e!s}"
            ) from e

    def _runtime_from_instance(
        self, tool_info: "ToolInfo", tool_instance: Any
    ) -> "ToolRuntime":
        """Create a ToolRuntime from an existing tool instance.

        Args:
            tool_info: Tool information
            tool_instance: The tool instance

        Returns:
            A configured ToolRuntime instance
        """
        # Determine if the tool has async run method
        run_method = getattr(tool_instance, "run", None)
        run_is_async = run_method is not None and inspect.iscoroutinefunction(
            run_method
        )

        # Check if tool supports streaming
        stream_method = getattr(tool_instance, "stream", None)
        supports_streaming = stream_method is not None and (
            inspect.iscoroutinefunction(stream_method)
            or inspect.isasyncgenfunction(stream_method)
        )

        # Check for input/output validation models
        has_input_validation = (
            hasattr(tool_instance, "Input")
            and issubclass(tool_instance.Input, BaseModel)
            if hasattr(tool_instance, "Input")
            else False
        )

        has_output_validation = (
            hasattr(tool_instance, "Output")
            and issubclass(tool_instance.Output, BaseModel)
            if hasattr(tool_instance, "Output")
            else False
        )

        # Determine the kind of tool
        kind = "mcp" if tool_info.source == ToolSource.MCP else "tool"

        return ToolRuntime(
            info=tool_info,
            instance=tool_instance,
            kind=kind,
            run_is_async=run_is_async,
            supports_streaming=supports_streaming,
            has_input_validation=has_input_validation,
            has_output_validation=has_output_validation,
        )

    def _get_runtime(self, tool_name: str) -> ToolRuntime:
        """Get the runtime instance for a tool.

        Args:
            tool_name: Name of the tool to get runtime for

        Returns:
            The ToolRuntime instance for the tool

        Raises:
            ToolRegistryError: If the tool is not found or not available
        """
        if tool_name not in self._runtimes:
            raise ToolRegistryError(f"Tool '{tool_name}' not found")

        runtime = self._runtimes[tool_name]

        # Check if tool is enabled and available
        tool_info = self._tools.get(tool_name)
        if tool_info and not tool_info.enabled:
            raise ToolRegistryError(f"Tool '{tool_name}' is disabled")

        if tool_info and not tool_info.available:
            raise ToolRegistryError(f"Tool '{tool_name}' is not available")

        return runtime

    def get_tool(self, tool_name: str) -> Tool:
        """Get a tool instance by name."""
        runtime = self._get_runtime(tool_name)
        return runtime.instance

    async def arun_tool(
        self, tool_name: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously execute a tool.

        Args:
            tool_name: Name of the tool to execute
            payload: Input parameters for the tool

        Returns:
            Tool execution result

        Raises:
            ToolRegistryError: If tool execution fails or tool is not async
        """
        start_time = time.time()

        try:
            runtime = self._get_runtime(tool_name)

            if runtime.supports_streaming:
                raise ToolRegistryError(
                    f"Tool '{tool_name}' supports streaming. Use stream_tool() instead."
                )

            logger.info("Executing async tool: %s", tool_name)
            result = await runtime.execute(payload)

            # Record success with execution time in ms
            execution_time = (time.time() - start_time) * 1000
            self._record_success(tool_name, execution_time)
            logger.info("Async tool '%s' executed in %.2fms", tool_name, execution_time)

            return result if result is not None else {}

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._record_error(tool_name, execution_time, e)
            error_msg = f"Error executing async tool '{tool_name}': {e!s}"
            logger.exception(error_msg)
            raise ToolRegistryError(error_msg) from e

    async def stream_tool(
        self, tool_name: str, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream results from a tool that supports streaming.

        Args:
            tool_name: Name of the tool to execute
            payload: Input parameters for the tool

        Yields:
            Chunks of tool output

        Raises:
            ToolRegistryError: If tool execution fails or tool doesn't support streaming
        """
        start_time = time.time()

        try:
            runtime = self._get_runtime(tool_name)

            if not runtime.supports_streaming:
                raise ToolRegistryError(
                    f"Tool '{tool_name}' does not support streaming"
                )

            if not runtime.run_is_async:
                raise ToolRegistryError(f"Streaming tool '{tool_name}' must be async")

            logger.info("Starting stream from tool: %s", tool_name)

            # Stream chunks using the runtime
            async for chunk in runtime.stream(payload):
                yield chunk

            # Record success with execution time in ms
            execution_time = (time.time() - start_time) * 1000
            self._record_success(tool_name, execution_time)
            logger.info(
                "Streaming tool '%s' completed in %.2fms", tool_name, execution_time
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._record_error(tool_name, execution_time, e)
            error_msg = f"Error streaming from tool '{tool_name}': {e!s}"
            logger.exception(error_msg)
            raise ToolRegistryError(error_msg) from e
