"""Tool runtime and input transformation utilities.

This module provides the ToolRuntime class for executing tools and ToolInputTransformer
for handling input transformations.
"""

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from local_coding_assistant.core.exceptions import ToolRegistryError
from local_coding_assistant.tools.types import ToolInfo
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("tools.tool_runtime")

T = TypeVar("T")


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
        cls, input_data: Any, input_model: type[BaseModel] | Any
    ) -> Any:
        """Transform input data using the provided Pydantic model or any type with model_fields."""
        if not (isinstance(input_model, type) and hasattr(input_model, "model_fields")):
            raise TypeError(f"Expected a type with model_fields, got {input_model}")

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


class ToolRuntime:
    """Runtime wrapper encapsulating execution and validation behavior for a tool."""

    def __init__(
        self,
        info: ToolInfo,
        instance: Any,
        kind: str,
        run_is_async: bool,
        supports_streaming: bool,
        has_input_validation: bool,
        has_output_validation: bool,
    ) -> None:
        """Initialize the ToolRuntime.

        Args:
            info: Tool information
            instance: The tool instance
            kind: The type of tool ("tool", "callable", or "mcp")
            run_is_async: Whether the tool's run method is async
            supports_streaming: Whether the tool supports streaming
            has_input_validation: Whether the tool has input validation
            has_output_validation: Whether the tool has output validation
        """
        self.info = info
        self.instance = instance
        self.kind = kind
        self.run_is_async = run_is_async
        self.supports_streaming = supports_streaming
        self.has_input_validation = has_input_validation
        self.has_output_validation = has_output_validation

    async def execute(self, payload: dict[str, Any]) -> Any:
        """Execute the tool with the provided payload."""
        input_data = await self._prepare_input(payload)
        result = await self._invoke_run(input_data)
        return await self._normalize_output(result)

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream results from the tool if supported.

        Args:
            payload: The input payload for the tool

        Yields:
            Chunks of the streamed result

        Raises:
            ToolRegistryError: If the tool doesn't support streaming
        """
        if not self.supports_streaming:
            raise ToolRegistryError(
                f"Tool '{self.info.name}' does not support streaming"
            )

        input_data = await self._prepare_input(payload)
        async for chunk in self._invoke_stream(input_data):
            yield await self._normalize_output(chunk)

    def _get_parameters_schema(self) -> dict[str, Any] | None:
        """Get the parameters schema for the tool.

        Returns:
            The tool's parameters schema
        """
        schema = getattr(self.info, "parameters", None)
        return schema or None

    async def _ensure_mapping(self, payload: Any) -> dict[str, Any]:
        """Coerce the payload into a dictionary-like structure.

        Args:
            payload: The input payload to coerce

        Returns:
            A dictionary containing the payload data

        Raises:
            ValueError: If the payload cannot be coerced to a dictionary
        """
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "model_dump"):
            return payload.model_dump()
        if hasattr(payload, "dict"):
            return payload.dict()
        if isinstance(payload, str | bytes | bytearray):
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError as err:
                raise ValueError(
                    "Could not parse string/bytes payload as JSON"
                ) from err

            if isinstance(decoded, dict):
                return decoded
            raise ValueError("JSON payload must decode to an object")

        try:
            return dict(payload)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Cannot convert payload to dict: {err}") from err

    def _transform_payload(self, payload: Any) -> dict:
        """Transform input payload to match the expected format.

        Args:
            payload: The input payload to transform

        Returns:
            Transformed payload as a dictionary

        Raises:
            ValueError: If transformation fails
        """
        if self.kind != "tool" or not self.has_input_validation:
            return payload

        try:
            return ToolInputTransformer.transform(
                payload, tool_info=self.info, tool_instance=self.instance
            )
        except Exception as e:
            logger.warning(
                "Error transforming input for tool '%s'",
                self.info.name,
                error=str(e),
                exc_info=logger.is_enabled_for(logging.DEBUG)
                if hasattr(logger, "is_enabled_for")
                else True,
            )
            return payload

    async def _validate_payload_against_model(
        self, payload: Any, input_model: type[BaseModel]
    ) -> Any:
        """Validate and convert payload using the tool's input model."""
        if not isinstance(input_model, type):
            raise ValueError(
                f"Expected a class/type for input_model, got {type(input_model).__name__}"
            )

        if isinstance(payload, input_model):
            return payload

        payload_dict = await self._ensure_mapping(payload)

        try:
            return input_model.model_validate(payload_dict)
        except ValidationError as model_error:
            schema = self._get_parameters_schema()
            if schema:
                try:
                    return await self._validate_with_parameters_schema(
                        payload_dict, schema
                    )
                except Exception as schema_error:
                    logger.debug(
                        "Falling back to parameters schema validation failed",
                        error=str(schema_error),
                    )

            raise model_error

    async def _validate_field_type(
        self, value: Any, field_type: str, field_schema: dict
    ) -> bool:
        """Validate a field value against its declared type in the schema."""
        if value is None:
            return field_schema.get("nullable", False)

        type_checkers = {
            "string": lambda x: isinstance(x, str),
            "number": lambda x: isinstance(x, int | float) and not isinstance(x, bool),
            "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
            "boolean": lambda x: isinstance(x, bool),
            "array": lambda x: isinstance(x, list | tuple),
            "object": lambda x: isinstance(x, dict),
        }

        # Handle union types (e.g., "string|null")
        if "|" in field_type:
            types = field_type.split("|")
            results = await asyncio.gather(
                *(
                    self._validate_field_type(value, t.strip(), field_schema)
                    for t in types
                )
            )
            return any(results)

        # Handle array type with items schema
        if field_type == "array":
            if not isinstance(value, (list, tuple)):
                return False
            if "items" in field_schema:
                item_schema = field_schema["items"]
                item_type = item_schema.get("type")
                if item_type:  # Only validate items if we have a type
                    results = await asyncio.gather(
                        *(
                            self._validate_field_type(item, item_type, item_schema)
                            for item in value
                        ),
                        return_exceptions=True,
                    )
                    return all(not isinstance(r, Exception) and r for r in results)
            return True

        # Handle standard types
        checker = type_checkers.get(field_type, lambda x: True)
        return checker(value)

    async def _validate_with_parameters_schema(
        self, payload: Any, schema: dict
    ) -> dict:
        """
        Validate the payload against the parameters schema.

        Args:
            payload: The input payload to validate
            schema: The parameters schema to validate against

        Returns:
            The validated payload as a dictionary

        Raises:
            ValueError: If validation fails
        """
        payload_dict = await self._ensure_mapping(payload)

        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        missing_fields = [
            field
            for field in required_fields
            if field not in payload_dict or payload_dict[field] is None
        ]

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        for field_name, field_schema in properties.items():
            if field_name not in payload_dict:
                continue

            value = payload_dict[field_name]
            if value is None:
                continue

            field_type = field_schema.get("type")
            if field_type and not await self._validate_field_type(
                value, field_type, field_schema
            ):
                raise ValueError(
                    f"Field '{field_name}' has invalid type. "
                    f"Expected {field_type}, got {type(value).__name__}"
                )

            if "enum" in field_schema and value not in field_schema["enum"]:
                raise ValueError(
                    f"Field '{field_name}' must be one of {field_schema['enum']}"
                )

        return payload_dict

    async def _format_validation_error(self, exc: ValidationError) -> str:
        """
        Format validation error messages into a user-friendly string.

        Args:
            exc: The ValidationError to format

        Returns:
            A formatted error message string
        """
        errors = []
        for error in exc.errors():
            loc = ".".join(str(loc_part) for loc_part in error.get("loc", []))
            msg = error.get("msg", "Validation error")
            if loc:
                errors.append(f"{loc}: {msg}")
            else:
                errors.append(msg)

        if not errors:
            return "Unknown validation error"

        return "\n  - " + "\n  - ".join(errors)

    async def _prepare_input(self, payload: Any) -> Any:
        """Validate and transform tool input."""
        # Early return for non-tools
        if self.kind != "tool":
            return payload

        # Transform the input to match the expected format
        try:
            payload = self._transform_payload(payload)
        except Exception as e:
            logger.warning(
                "Error transforming input for tool '%s'",
                self.info.name,
                error=str(e),
                exc_info=logger.is_enabled_for(logging.DEBUG)
                if hasattr(logger, "is_enabled_for")
                else True,
            )
            # Continue with the original payload if transformation fails
            pass

        schema = self._get_parameters_schema()

        input_model = getattr(self.instance, "Input", None)
        if input_model is not None and issubclass(input_model, BaseModel):
            try:
                return await self._validate_payload_against_model(payload, input_model)
            except ValidationError as model_error:
                error_msg = await self._format_validation_error(model_error)
                raise ToolRegistryError(
                    f"Invalid input for tool '{self.info.name}':{error_msg}"
                ) from model_error
            except (ValueError, AttributeError) as exc:
                raise ToolRegistryError(
                    f"Invalid input for tool '{self.info.name}': {exc}"
                ) from exc

        if schema:
            try:
                return await self._validate_with_parameters_schema(payload, schema)
            except Exception as exc:
                raise ToolRegistryError(
                    f"Input validation failed for tool '{self.info.name}': {exc}"
                ) from exc

        # No validation to perform
        return payload

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
        """Invoke the streaming interface on the tool.

        Args:
            input_data: The prepared input data

        Yields:
            Chunks of output from the tool

        Raises:
            NotImplementedError: If the tool doesn't support streaming
        """
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

    def _convert_to_dict(self, obj: Any) -> Any:
        """Convert an object to a dictionary using the appropriate method.

        Args:
            obj: The object to convert to a dictionary

        Returns:
            The object converted to a dictionary
        """
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "__dict__"):
            return vars(obj)
        if isinstance(obj, (list, tuple)):
            return [self._convert_to_dict(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._convert_to_dict(v) for k, v in obj.items()}
        return str(obj)

    async def _validate_with_model(
        self, output_model: type[BaseModel], result: Any
    ) -> dict:
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

    async def _normalize_output(self, result: Any) -> Any:
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
            return await self._coerce_to_serializable(result)

        # Get the output model from the tool
        output_model = getattr(self.instance, "Output", None)
        if output_model is None:
            return await self._coerce_to_serializable(result)

        # Handle case where result is already the correct model
        if isinstance(result, output_model):
            return self._convert_to_dict(result)

        try:
            return await self._validate_with_model(output_model, result)
        except (ValidationError, ValueError, TypeError) as exc:
            logger.error(
                "Output validation failed for tool '%s'",
                self.info.name,
                error=str(exc),
                exc_info=logger.is_enabled_for(logging.DEBUG)
                if hasattr(logger, "is_enabled_for")
                else True,
            )
            # Return the original result if validation fails
            return await self._coerce_to_serializable(result)

    @staticmethod
    async def _coerce_to_serializable(result: Any) -> Any:
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "dict"):
            return result.dict()
        return result
