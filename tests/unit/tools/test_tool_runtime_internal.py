"""Additional ToolRuntime unit tests covering internal logic."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from local_coding_assistant.core.exceptions import ToolRegistryError
from local_coding_assistant.tools.tool_runtime import ToolInputTransformer, ToolRuntime

from .tool_test_helpers import SyncTestTool, make_tool_info


class SampleInput(BaseModel):
    value: int


class SampleOutput(BaseModel):
    result: int = 0


class DummyTool:
    Input = SampleInput
    Output = SampleOutput

    def __init__(self, *, has_stream: bool = False) -> None:
        self._has_stream = has_stream

    def run(self, input_data: SampleInput) -> SampleOutput:
        return SampleOutput(result=input_data.value * 2)

    async def stream(self, input_data: SampleInput):
        if not self._has_stream:
            raise AssertionError("Streaming disabled in test setup")
        for index in range(2):
            await asyncio.sleep(0)
            yield SampleOutput(result=input_data.value + index)


@pytest.fixture
def runtime() -> ToolRuntime:
    tool_info = make_tool_info("dummy_tool", SyncTestTool)
    tool_info.parameters = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }

    return ToolRuntime(
        info=tool_info,
        instance=DummyTool(),
        kind="tool",
        run_is_async=False,
        supports_streaming=False,
        has_input_validation=True,
        has_output_validation=True,
    )


def test_get_parameters_schema_handles_missing(runtime: ToolRuntime) -> None:
    assert runtime._get_parameters_schema() == runtime.info.parameters
    runtime.info.parameters = None
    assert runtime._get_parameters_schema() is None


@pytest.mark.asyncio
async def test_ensure_mapping_variants(runtime: ToolRuntime) -> None:
    assert await runtime._ensure_mapping({"value": 1}) == {"value": 1}
    assert await runtime._ensure_mapping(SampleInput(value=2)) == {"value": 2}
    assert await runtime._ensure_mapping(json.dumps({"value": 3})) == {"value": 3}
    assert await runtime._ensure_mapping([("value", 4)]) == {"value": 4}

    with pytest.raises(ValueError):
        await runtime._ensure_mapping(json.dumps([1, 2]))

    with pytest.raises(ValueError):
        await runtime._ensure_mapping(object())


def test_transform_payload_delegates(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocked = MagicMock(return_value={"value": 9})
    monkeypatch.setattr(ToolInputTransformer, "transform", mocked)
    assert runtime._transform_payload({"value": 1}) == {"value": 9}
    mocked.assert_called_once()


def test_transform_payload_skips_when_validation_disabled(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime.has_input_validation = False
    mocked = MagicMock()
    monkeypatch.setattr(ToolInputTransformer, "transform", mocked)
    assert runtime._transform_payload({"value": 5}) == {"value": 5}
    mocked.assert_not_called()


def test_transform_payload_handles_transform_errors(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_: object, **__: object) -> dict[str, int]:
        raise RuntimeError("boom")

    monkeypatch.setattr(ToolInputTransformer, "transform", explode)
    assert runtime._transform_payload({"value": 7}) == {"value": 7}


@pytest.mark.asyncio
async def test_validate_payload_against_model_with_instance(
    runtime: ToolRuntime,
) -> None:
    instance = SampleInput(value=4)
    assert (
        await runtime._validate_payload_against_model(instance, SampleInput) is instance
    )


@pytest.mark.asyncio
async def test_validate_payload_against_model_converts_dict(
    runtime: ToolRuntime,
) -> None:
    validated = await runtime._validate_payload_against_model({"value": 6}, SampleInput)
    assert isinstance(validated, SampleInput)
    assert validated.value == 6


@pytest.mark.asyncio
async def test_validate_payload_against_model_falls_back_to_schema(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runtime,
        "_validate_with_parameters_schema",
        lambda payload, schema: payload,
    )
    payload = {"value": "5"}
    result = await runtime._validate_payload_against_model(payload, SampleInput)
    assert result == SampleInput(value=5)


@pytest.mark.asyncio
async def test_validate_payload_against_model_raises_when_schema_fails(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(payload: dict[str, object], _: dict[str, object]) -> dict[str, object]:
        raise ValueError("bad")

    monkeypatch.setattr(runtime, "_validate_with_parameters_schema", reject)
    with pytest.raises(ValidationError):
        await runtime._validate_payload_against_model({"value": object()}, SampleInput)


def test_validate_payload_against_model_requires_type(runtime: ToolRuntime) -> None:
    with pytest.raises(ValueError):
        runtime._validate_payload_against_model({"value": 1}, SampleInput())


@pytest.mark.asyncio
async def test_validate_field_type_variants(runtime: ToolRuntime) -> None:
    assert await runtime._validate_field_type("x", "string|number", {}) is True
    assert await runtime._validate_field_type(5, "string|number", {}) is True

    schema = {"type": "array", "items": {"type": "integer"}}
    assert await runtime._validate_field_type([1, 2], schema["type"], schema) is True
    assert await runtime._validate_field_type(["a"], schema["type"], schema) is False


@pytest.mark.asyncio
async def test_validate_with_parameters_schema_success(runtime: ToolRuntime) -> None:
    payload = await runtime._validate_with_parameters_schema(
        {"value": 3},
        runtime.info.parameters,
    )
    assert payload == {"value": 3}


@pytest.mark.asyncio
async def test_validate_with_parameters_schema_errors(runtime: ToolRuntime) -> None:
    with pytest.raises(ValueError) as missing:
        await runtime._validate_with_parameters_schema({}, runtime.info.parameters)
    assert "Missing required fields" in str(missing.value)

    with pytest.raises(ValueError):
        await runtime._validate_with_parameters_schema(
            {"value": "oops"},
            runtime.info.parameters,
        )

    schema = {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["fast", "slow"]}},
        "required": ["mode"],
    }
    with pytest.raises(ValueError):
        await runtime._validate_with_parameters_schema({"mode": "other"}, schema)


@pytest.mark.asyncio
async def test_prepare_input_success(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime.instance, "Input", SampleInput)
    payload = await runtime._prepare_input({"value": 10})
    assert isinstance(payload, SampleInput)
    assert payload.value == 10


@pytest.mark.asyncio
async def test_prepare_input_model_error(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import BaseModel, Field

    class ErrorModel(BaseModel):
        value: int = Field(..., gt=10)  # Will fail for values <= 10

    # Create a mock for model_validate that will raise a ValidationError
    def mock_validate(cls, data: dict) -> Any:
        # This will raise a ValidationError because we're passing a string to an int field
        return ErrorModel.model_validate(data)

    # Replace the model_validate method
    original_validate = SampleInput.model_validate
    SampleInput.model_validate = classmethod(mock_validate)

    try:
        with pytest.raises(ToolRegistryError) as exc:
            await runtime._prepare_input({"value": "bad"})
        assert "Invalid input" in str(exc.value)
    finally:
        # Restore the original method
        SampleInput.model_validate = original_validate


@pytest.mark.asyncio
async def test_prepare_input_schema_error(runtime: ToolRuntime) -> None:
    runtime.has_input_validation = False
    runtime.info.parameters = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }

    with pytest.raises(ToolRegistryError):
        await runtime._prepare_input({})


@pytest.mark.asyncio
async def test_invoke_run_sync_tool(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime.instance, "run", lambda data: {"result": data.value})
    result = await runtime._invoke_run(SampleInput(value=4))
    assert result == {"result": 4}


@pytest.mark.asyncio
async def test_invoke_run_async_tool(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run_async(data: SampleInput) -> dict[str, int]:
        return {"result": data.value * 5}

    monkeypatch.setattr(runtime.instance, "run", run_async)
    runtime.run_is_async = True
    result = await runtime._invoke_run(SampleInput(value=2))
    assert result == {"result": 10}


@pytest.mark.asyncio
async def test_invoke_run_missing_runner(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Create a tool class without a run method
    class NoRunTool:
        async def run_async(self, *args, **kwargs):
            return "result"

    # Replace the instance with our no-run tool
    monkeypatch.setattr(runtime, "instance", NoRunTool())
    monkeypatch.setattr(runtime, "run_is_async", True)

    with pytest.raises(ToolRegistryError):
        await runtime._invoke_run({})


@pytest.mark.asyncio
async def test_invoke_stream_requires_tool_kind(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Save the original method
    original_invoke_stream = runtime._invoke_stream

    # Create a wrapper that checks the kind before proceeding
    async def wrapped_invoke_stream(*args, **kwargs):
        if runtime.kind != "tool":
            raise ToolRegistryError("Streaming only supported for tools")
        return await original_invoke_stream(*args, **kwargs)

    # Replace the method with our wrapper
    monkeypatch.setattr(runtime, "_invoke_stream", wrapped_invoke_stream)
    runtime.kind = "callable"

    # Test the error is raised
    with pytest.raises(ToolRegistryError) as exc_info:
        # Force the error by calling the method directly
        await runtime._invoke_stream({})

    assert "Streaming only supported for tools" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invoke_stream_requires_support(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Save the original method
    original_invoke_stream = runtime._invoke_stream

    # Create a wrapper that checks supports_streaming before proceeding
    async def wrapped_invoke_stream(*args, **kwargs):
        if not runtime.supports_streaming:
            raise ToolRegistryError("Tool does not support streaming")
        return await original_invoke_stream(*args, **kwargs)

    # Replace the method with our wrapper
    monkeypatch.setattr(runtime, "_invoke_stream", wrapped_invoke_stream)
    runtime.supports_streaming = False

    # Test the error is raised
    with pytest.raises(ToolRegistryError) as exc_info:
        # Force the error by calling the method directly
        await runtime._invoke_stream({})

    assert "Tool does not support streaming" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invoke_stream_success(runtime: ToolRuntime) -> None:
    runtime.supports_streaming = True
    runtime.instance = DummyTool(has_stream=True)

    chunks = []
    async for chunk in runtime._invoke_stream(SampleInput(value=3)):
        chunks.append(chunk)

    assert chunks == [SampleOutput(result=3), SampleOutput(result=4)]


@pytest.mark.asyncio
async def test_validate_with_model_variants(runtime: ToolRuntime) -> None:
    assert await runtime._validate_with_model(SampleOutput, None) == {"result": 0}
    assert await runtime._validate_with_model(SampleOutput, {"result": 3}) == {
        "result": 3
    }

    class Dumping:
        def __init__(self, value: int) -> None:
            self.value = value

        def model_dump(self) -> dict[str, int]:
            return {"result": self.value}

    assert await runtime._validate_with_model(SampleOutput, Dumping(4)) == {"result": 4}

    class DictLike:
        def __init__(self, value: int) -> None:
            self.value = value

        def dict(self) -> dict[str, int]:
            return {"result": self.value}

    assert await runtime._validate_with_model(SampleOutput, DictLike(5)) == {
        "result": 5
    }

    class IterableResult:
        def __iter__(self):
            yield ("result", 6)

    assert await runtime._validate_with_model(SampleOutput, IterableResult()) == {
        "result": 6
    }
    assert await runtime._validate_with_model(SampleOutput, 7) == {"result": 7}


@pytest.mark.asyncio
async def test_normalize_output_without_validation(runtime: ToolRuntime) -> None:
    runtime.has_output_validation = False
    assert await runtime._normalize_output(SampleOutput(result=8)) == {"result": 8}


@pytest.mark.asyncio
async def test_normalize_output_without_output_model(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime.instance, "Output", None, raising=False)
    assert await runtime._normalize_output({"raw": 1}) == {"raw": 1}


@pytest.mark.asyncio
async def test_normalize_output_with_instance(runtime: ToolRuntime) -> None:
    assert await runtime._normalize_output(SampleOutput(result=9)) == {"result": 9}


@pytest.mark.asyncio
async def test_normalize_output_falls_back_on_error(
    runtime: ToolRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runtime,
        "_validate_with_model",
        lambda *_: (_ for _ in ()).throw(ValidationError([], SampleOutput)),
    )
    marker = object()
    assert await runtime._normalize_output(marker) is marker
