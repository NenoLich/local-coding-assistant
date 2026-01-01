"""Unit tests for the tool configuration loader module."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional, Union

import pytest
import yaml
from pydantic import BaseModel

from local_coding_assistant.config.schemas import ToolConfig
from local_coding_assistant.config.tool_loader import (
    ToolConfigConverter,
    ToolConfigLoader,
    ToolLoader,
    ToolModuleLoader,
    deep_merge_dicts,
)
from local_coding_assistant.core.exceptions import ConfigError


def test_deep_merge_dicts_merges_nested_maps_without_mutating_inputs():
    base = {
        "a": 1,
        "nested": {"x": 1, "shared": {"foo": "bar"}},
        "list_value": [1, 2],
    }
    override = {
        "b": 2,
        "nested": {"y": 2, "shared": {"baz": "qux"}},
        "list_value": [3],
    }

    merged = deep_merge_dicts(base, override)

    assert merged == {
        "a": 1,
        "b": 2,
        "nested": {
            "x": 1,
            "y": 2,
            "shared": {"foo": "bar", "baz": "qux"},
        },
        "list_value": [3],
    }
    # Ensure original dictionaries remain unchanged
    assert base["list_value"] == [1, 2]
    assert "b" not in base
    assert "y" not in base["nested"]


def test_tool_config_loader_merges_default_and_local_configs(test_configs):
    # Test data
    default_payload = {
        "tools": [
            {
                "id": "echo",
                "description": "default description",
                "path": "modules/echo_tool.py",
                "config": {"nested": {"a": 1}},
            }
        ]
    }
    local_payload = {
        "tools": [
            {
                "id": "echo",
                "description": "local description",
                "module": "pkg.echo_tool",
                "config": {"nested": {"b": 2}, "extra": True},
            },
            {
                "id": "new_tool",
                "description": "new tool",
                "module": "pkg.new_tool",
                "enabled": False,
            },
        ]
    }

    # Write test configs using the fixture
    test_configs["default"].write_text(
        yaml.safe_dump(default_payload), encoding="utf-8"
    )
    test_configs["local"].write_text(yaml.safe_dump(local_payload), encoding="utf-8")

    # Create a test echo module file
    echo_module = test_configs["modules_dir"] / "echo_tool.py"
    echo_module.write_text(
        """
class EchoTool:
    def run(self, *args, **kwargs):
        return {"result": "echo"}
""",
        encoding="utf-8",
    )

    # Create the loader with our test path manager
    loader = ToolConfigLoader(env_manager=test_configs["env_manager"])
    raw_configs = loader.load()

    # Verify results
    assert sorted(raw_configs) == ["echo", "new_tool"]

    echo_data = raw_configs["echo"]
    assert echo_data["config"]["description"] == "local description"
    assert echo_data["config"]["config"]["nested"] == {"a": 1, "b": 2}
    assert echo_data["config"]["config"]["extra"] is True
    assert echo_data["config"]["source"] == "external"
    assert echo_data["base_dir"] == test_configs["config_dir"]
    assert echo_data["_source"]["config_type"] == "local"
    assert str(Path(echo_data["_source"]["file"]).resolve()) == str(
        test_configs["local"].resolve()
    )

    new_tool_data = raw_configs["new_tool"]
    assert new_tool_data["config"]["enabled"] is False
    assert new_tool_data["base_dir"] == test_configs["config_dir"]
    assert new_tool_data["_source"]["config_type"] == "local"


def test_tool_loader_load_tool_configs_invokes_subcomponents(monkeypatch):
    template_raw_configs = {
        "sample": {
            "config": {
                "description": "sample tool",
                "source": "builtin",
                "module": "json",
            },
            "base_dir": None,
        }
    }

    raw_calls: list[dict[str, dict[str, object]]] = []
    enrich_calls: list[dict[str, dict[str, object]]] = []
    tool_config_paths_used = []

    def fake_load(self, tool_config_paths=None):
        data = copy.deepcopy(template_raw_configs)
        raw_calls.append(data)
        tool_config_paths_used.append(tool_config_paths)
        return data

    class DummyTool:
        class Input:  # pragma: no cover - structural placeholder
            pass

        class Output:  # pragma: no cover - structural placeholder
            pass

        def run(self):  # pragma: no cover - structural placeholder
            return None

    def fake_enrich(self, configs):
        enrich_calls.append(copy.deepcopy(configs))
        configs["sample"]["tool_class"] = DummyTool
        configs["sample"]["config"]["available"] = True

    def fake_convert(self, configs):
        assert "tool_class" in configs["sample"]
        return {
            "sample": ToolConfig(
                id="sample",
                description="sample tool",
                source="builtin",
                available=True,
            )
        }

    monkeypatch.setattr(ToolConfigLoader, "load", fake_load)
    monkeypatch.setattr(ToolModuleLoader, "enrich_with_modules", fake_enrich)
    monkeypatch.setattr(ToolConfigConverter, "convert", fake_convert)

    loader = ToolLoader()
    result = loader.load_tool_configs()

    assert len(raw_calls) == 1
    assert len(enrich_calls) == 1
    assert "sample" in result
    assert isinstance(result["sample"], ToolConfig)
    assert result["sample"].available is True


def test_tool_config_converter_convert_mcp_tool_validates_required_fields():
    converter = ToolConfigConverter()

    with pytest.raises(ConfigError):
        converter._convert_mcp_tool(
            "remote", {"description": "missing endpoint", "provider": "test"}
        )

    with pytest.raises(ConfigError):
        converter._convert_mcp_tool(
            "remote", {"description": "missing provider", "endpoint": "https://example"}
        )

    result = converter._convert_mcp_tool(
        "remote",
        {
            "description": "Remote MCP tool",
            "endpoint": "https://example",
            "provider": "test",
            "source": "mcp",
        },
    )

    assert isinstance(result, ToolConfig)
    assert result.id == "remote"
    assert result.endpoint == "https://example"
    assert result.provider == "test"
    assert result.is_async is True
    assert result.available is True


def test_retrieve_parameters_from_input_model():
    """Test extracting parameters from Input model using model_json_schema."""
    from pydantic import BaseModel

    class TestInputModel(BaseModel):
        """Test input model with various field types."""

        name: str
        age: int
        is_active: bool = True
        tags: list[str] = []
        metadata: dict[str, Any] | None = None

    class TestTool:
        """Test tool with Input model."""

        Input = TestInputModel

        def run(self, **kwargs):
            pass

    loader = ToolModuleLoader()
    parameters = loader._retrieve_parameters(TestTool)

    assert parameters["type"] == "object"
    assert set(parameters["required"]) == {"name", "age"}
    assert parameters["properties"]["name"]["type"] == "string"
    assert parameters["properties"]["age"]["type"] == "integer"
    assert parameters["properties"]["is_active"]["type"] == "boolean"
    assert parameters["properties"]["tags"]["type"] == "array"
    assert parameters["properties"]["metadata"]["type"] == "object"


class TestNormalizeType:
    """Tests for ToolModuleLoader.normalize_type method."""

    @pytest.fixture
    def loader(self):
        return ToolModuleLoader()

    def test_primitive_types(self, loader):
        """Test primitive type normalization."""
        assert loader.normalize_type(str) == {"type": "string"}
        assert loader.normalize_type(int) == {"type": "integer"}
        assert loader.normalize_type(float) == {"type": "number"}
        assert loader.normalize_type(bool) == {"type": "boolean"}

    def test_optional_types(self, loader):
        """Test Optional/Union with None types."""
        # Using Optional
        assert loader.normalize_type(Optional[str]) == {
            "type": "string",
            "nullable": True,
        }

        # Using Union with None
        assert loader.normalize_type(Union[str, None]) == {
            "type": "string",
            "nullable": True,
        }

        # Using | operator (Python 3.10+)
        assert loader.normalize_type(str | None) == {"type": "string", "nullable": True}

    def test_collection_types(self, loader):
        """Test list and dict types."""
        # List with type parameter
        assert loader.normalize_type(list[str]) == {
            "type": "array",
            "items": {"type": "string"},
        }

        # Dict with type parameters
        assert loader.normalize_type(dict[str, int]) == {"type": "object"}

        # Nested collections
        assert loader.normalize_type(list[dict[str, int]]) == {
            "type": "array",
            "items": {"type": "object"},
        }

    def test_union_types(self, loader):
        """Test Union types (non-None)."""
        # Union of primitives
        result = loader.normalize_type(Union[str, int])
        assert result["type"] == "string"  # First non-null type

        # Union with multiple types
        result = loader.normalize_type(Union[str, int, bool])
        assert result["type"] == "string"  # First non-null type

    def test_pydantic_model(self, loader):
        """Test with Pydantic model types."""

        class TestModel(BaseModel):
            name: str
            age: int

        result = loader.normalize_type(TestModel)
        # Check for required schema fields
        assert result["type"] == "object"
        assert "properties" in result
        assert "name" in result["properties"]
        assert "age" in result["properties"]
        assert "required" in result
        assert set(result["required"]) == {"name", "age"}

    def test_any_type(self, loader):
        """Test Any type handling."""
        assert loader.normalize_type(Any) == {"type": "string"}

    def test_unknown_type(self, loader):
        """Test fallback for unknown types."""

        class CustomType:
            pass

        assert loader.normalize_type(CustomType) == {"type": "string"}

    def test_nested_optional(self, loader):
        """Test nested optional types."""
        # Optional list of strings
        result = loader.normalize_type(Optional[list[str]])
        assert result == {
            "type": "array",
            "items": {"type": "string"},
            "nullable": True,
        }

        # List of optional strings
        result = loader.normalize_type(list[str | None])
        assert result == {
            "type": "array",
            "items": {"type": "string", "nullable": True},
        }


def test_retrieve_parameters_from_run_method():
    """Test extracting parameters from run method signature."""

    class TestTool:
        """Test tool with run method parameters."""

        def run(
            self, name: str, age: int, is_active: bool = True, tags: list[str] = None
        ):
            pass

    loader = ToolModuleLoader()
    parameters = loader._retrieve_parameters(TestTool)

    assert parameters["type"] == "object"
    assert set(parameters["required"]) == {"name", "age"}
    assert parameters["properties"]["name"]["type"] == "string"
    assert parameters["properties"]["age"]["type"] == "integer"
    assert parameters["properties"]["is_active"]["type"] == "boolean"
    assert parameters["properties"]["tags"]["type"] == "array"


def test_retrieve_parameters_no_input_or_run():
    """Test behavior when neither Input model nor run method is present."""

    class EmptyTool:
        pass

    loader = ToolModuleLoader()
    parameters = loader._retrieve_parameters(EmptyTool)

    assert parameters == {"type": "object", "properties": {}, "required": []}


def test_retrieve_parameters_with_invalid_input_model():
    """Test behavior when Input model exists but has no model_json_schema."""

    class TestTool:
        class Input:
            """Invalid input model without model_json_schema."""

            pass

        def run(self):
            pass

    loader = ToolModuleLoader()
    parameters = loader._retrieve_parameters(TestTool)

    # Should fall back to run method parameters (none in this case)
    assert parameters == {"type": "object", "properties": {}, "required": []}


def test_retrieve_parameters_with_coroutine_run():
    """Test with an async run method."""
    import asyncio

    class AsyncTestTool:
        """Test tool with async run method."""

        async def run(self, name: str, count: int = 1):
            await asyncio.sleep(0)
            return {"result": f"Hello {name}" * count}

    loader = ToolModuleLoader()
    parameters = loader._retrieve_parameters(AsyncTestTool)

    assert parameters["type"] == "object"
    assert set(parameters["required"]) == {"name"}
    assert parameters["properties"]["name"]["type"] == "string"
    assert parameters["properties"]["count"]["type"] == "integer"
