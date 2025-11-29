"""Unit tests for ToolRuntime internals."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from local_coding_assistant.core.exceptions import ToolRegistryError
from local_coding_assistant.tools.tool_manager import ToolRuntime, ToolInputTransformer
from local_coding_assistant.tools.types import ToolInfo, ToolSource


class ToolInputModel(BaseModel):
    value: int


class ToolOutputModel(BaseModel):
    result: int


class DummyTool:
    class Input(ToolInputModel):
        value: int

    class Output(ToolOutputModel):
        result: int

    def __init__(self, *, run_is_async: bool = False, supports_streaming: bool = False):
        self._run_is_async = run_is_async
        self._supports_streaming = supports_streaming

    def run(self, data):
        return self.run_sync(data)

    async def run_async(self, data):
        return {"result": data.value * 2}

    def run_sync(self, data):
        return {"result": data.value}

    async def stream_async(self, data):
        # Handle both dictionary and object access
        value = data.value if hasattr(data, 'value') else data.get('value', 0)
        for index in range(2):
            yield {"chunk": value + index}
            
    async def stream(self, data):
        # Handle both dictionary and object access
        value = data.value if hasattr(data, 'value') else data.get('value', 0)
        for index in range(2):
            yield {"chunk": value + index}


@pytest.fixture
def tool_info() -> ToolInfo:
    return ToolInfo(
        name="dummy",
        description="",
        source=ToolSource.BUILTIN,
        tool_class=None,
        permissions=[],
        tags=[],
        is_async=False,
        supports_streaming=False,
    )


@pytest.fixture
def runtime(tool_info):
    tool = DummyTool()
    runtime = ToolRuntime(
        info=tool_info,
        instance=tool,
        kind="tool",
        run_is_async=False,
        supports_streaming=False,
        has_input_validation=True,
        has_output_validation=True,
    )
    return runtime


def test_ensure_mapping_accepts_dict(runtime):
    payload = {"value": 3}
    assert runtime._ensure_mapping(payload) == payload


def test_ensure_mapping_handles_model_dump(runtime):
    class Payload(BaseModel):
        value: int

    payload = Payload(value=9)
    assert runtime._ensure_mapping(payload) == {"value": 9}


def test_ensure_mapping_parses_json(runtime):
    payload = json.dumps({"value": 5})
    assert runtime._ensure_mapping(payload) == {"value": 5}


def test_ensure_mapping_raises_for_bad_json(runtime):
    with pytest.raises(ValueError):
        runtime._ensure_mapping("not json")


def test_ensure_mapping_rejects_non_iterable(runtime):
    with pytest.raises(ValueError):
        runtime._ensure_mapping(object())


def test_transform_payload_uses_transformer(runtime, monkeypatch):
    monkeypatch.setattr(ToolInputTransformer, "transform", MagicMock(return_value={"value": 7}))
    payload = runtime._transform_payload({"value": 3})
    assert payload == {"value": 7}


def test_transform_payload_logs_but_returns_original(runtime, caplog, monkeypatch):
    def explode(*_, **__):
        raise RuntimeError("boom")

    monkeypatch.setattr(ToolInputTransformer, "transform", explode)
    payload = runtime._transform_payload({"value": 3})
    assert payload == {"value": 3}


def test_validate_payload_against_model_handles_instances(runtime):
    model_instance = ToolInputModel(value=4)
    assert runtime._validate_payload_against_model(model_instance, ToolInputModel) is model_instance


def test_validate_payload_against_model_uses_schema(runtime, monkeypatch):
    payload = {"value": -1}
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer", "enum": [0, 1]}},
        "required": ["value"],
    }
    runtime.info.parameters = schema

    class RejectingModel(ToolInputModel):
        @classmethod
        def model_validate(cls, value):  # pragma: no cover - replaced in monkeypatch
            raise ValidationError.from_exception_data("Error", [])

    monkeypatch.setattr(runtime.instance, "Input", RejectingModel)
    with pytest.raises(ValidationError):
        runtime._validate_payload_against_model(payload, RejectingModel)


def test_validate_field_type_for_union(runtime):
    field_schema = {"type": "string|number"}
    assert runtime._validate_field_type(5, field_schema["type"], field_schema)


def test_validate_field_type_array_items(runtime):
    field_schema = {"type": "array", "items": {"type": "integer"}}
    assert runtime._validate_field_type([1, 2], field_schema["type"], field_schema)
    assert not runtime._validate_field_type(["a"], field_schema["type"], field_schema)


def test_validate_with_parameters_schema_enforces_required(runtime):
    runtime.info.parameters = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }

    with pytest.raises(ValueError) as exc:
        runtime._validate_with_parameters_schema({}, runtime.info.parameters)

    assert "Missing required fields" in str(exc.value)


def test_validate_with_parameters_schema_checks_enum(runtime):
    runtime.info.parameters = {
        "type": "object",
        "properties": {"value": {"type": "integer", "enum": [1]}},
        "required": ["value"],
    }

    with pytest.raises(ValueError):
        runtime._validate_with_parameters_schema({"value": 2}, runtime.info.parameters)


def test_prepare_input_handles_validation_success(runtime, monkeypatch):
    monkeypatch.setattr(runtime.instance, "Input", ToolInputModel)
    payload = runtime._prepare_input({"value": 3})
    assert isinstance(payload, ToolInputModel)
    assert payload.value == 3


def test_prepare_input_raises_tool_registry_error(runtime, monkeypatch):
    # Import the ValidationError from the same module as BaseModel
    from pydantic import BaseModel, Field
    from pydantic_core import ValidationError
    
    # Create a model that will raise ValidationError
    class RejectingModel(BaseModel):
        required_field: int = Field(..., description="This field is required")
    
    # Create a function that will raise ValidationError when called
    def raise_validation_error(*args, **kwargs):
        # This will raise ValidationError because required_field is missing
        RejectingModel.model_validate({"invalid": "data"})
    
    # Patch the _validate_payload_against_model method to raise the error
    original_validate = runtime._validate_payload_against_model
    
    def mock_validate_payload(payload, model):
        if model == RejectingModel:
            raise_validation_error()
        return original_validate(payload, model)
    
    monkeypatch.setattr(runtime, "_validate_payload_against_model", mock_validate_payload)
    
    # Patch the Input class on the instance
    monkeypatch.setattr(runtime.instance, "Input", RejectingModel)
    
    # Now this should raise ToolRegistryError because the validation will fail
    with pytest.raises(ToolRegistryError) as exc_info:
        runtime._prepare_input({"invalid": "data"})
    
    # Verify the error message contains the expected text
    assert "required_field" in str(exc_info.value) or "missing" in str(exc_info.value)


def test_prepare_input_schema_validation_error(runtime):
    runtime.has_input_validation = False
    runtime.info.parameters = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }

    with pytest.raises(ToolRegistryError):
        runtime._prepare_input({})


def test_invoke_run_sync(runtime, monkeypatch):
    async def execute(data):
        return {"result": data.value * 2}

    monkeypatch.setattr(runtime.instance, "run", lambda data: {"result": data.value})
    result = asyncio.run(runtime._invoke_run(ToolInputModel(value=4)))
    assert result == {"result": 4}


def test_invoke_run_async(runtime, monkeypatch):
    async def run_async(data):
        return {"result": data.value * 5}

    monkeypatch.setattr(runtime.instance, "run", run_async)
    monkeypatch.setattr(runtime, "run_is_async", True)
    result = asyncio.run(runtime._invoke_run(ToolInputModel(value=4)))
    assert result == {"result": 20}


def test_invoke_run_missing_run_method(runtime, monkeypatch):
    class NoRunTool:
        Input = ToolInputModel
        Output = ToolOutputModel
        
    runtime.instance = NoRunTool()
    with pytest.raises(ToolRegistryError):
        asyncio.run(runtime._invoke_run({"value": 1}))


async def _collect_async_generator(async_gen):
    return [item async for item in async_gen]

def test_invoke_stream_requires_tool(runtime, monkeypatch):
    # Set kind to something other than 'tool' to trigger the error
    runtime.kind = "callable"
    
    # Create a valid input for the stream method
    input_data = {"value": 1}
    
    # The error should come from the stream method checking the kind
    with pytest.raises(ToolRegistryError) as exc_info:
        asyncio.run(_collect_async_generator(runtime.stream(input_data)))
    
    # The error message should indicate that streaming is not supported
    assert "does not support streaming operations" in str(exc_info.value)


def test_invoke_stream_requires_support(runtime, monkeypatch):
    # Create a valid input for the stream method
    input_data = {"value": 1}
    
    # The error should come from the stream method checking supports_streaming
    runtime.kind = "tool"  # Make sure kind is 'tool' to pass the first check
    runtime.supports_streaming = False  # Explicitly set to False
    
    with pytest.raises(ToolRegistryError) as exc_info:
        asyncio.run(_collect_async_generator(runtime.stream(input_data)))
    
    # The error message should indicate that streaming is not supported
    assert "does not support streaming" in str(exc_info.value)


def test_invoke_stream_success(runtime, monkeypatch):
    async def stream_mock(payload):
        for value in [1, 2]:
            yield value

    runtime.supports_streaming = True
    # Use a mock that returns our test data
    monkeypatch.setattr(runtime.instance, "stream", stream_mock)
    
    # Collect all items from the async generator
    results = asyncio.run(_collect_async_generator(runtime._invoke_stream({"value": 1})))
    assert results == [1, 2]


async def _collect(iterator):
    return [item async for item in iterator]
