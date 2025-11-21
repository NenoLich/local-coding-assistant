"""Unit tests for the tool configuration loader module."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

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


def test_tool_config_loader_merges_default_and_local_configs(tmp_path, monkeypatch):
    default_payload = {
        "tools": [
            {
                "id": "echo",
                "description": "default description",
                "path": "tools/echo_tool.py",
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

    default_path = tmp_path / "tools.default.yaml"
    local_path = tmp_path / "tools.local.yaml"
    default_path.write_text(yaml.safe_dump(default_payload), encoding="utf-8")
    local_path.write_text(yaml.safe_dump(local_payload), encoding="utf-8")

    path_map: dict[str, Path] = {"default": default_path, "local": local_path}

    def fake_get_config_path(self, config_type: str) -> Path | None:  # noqa: D401
        return path_map.get(config_type)

    monkeypatch.setattr(
        ToolConfigLoader,
        "_get_config_path",
        fake_get_config_path,
        raising=False,
    )

    loader = ToolConfigLoader()
    raw_configs = loader.load()

    assert sorted(raw_configs) == ["echo", "new_tool"]

    echo_data = raw_configs["echo"]
    assert echo_data["config"]["description"] == "local description"
    assert echo_data["config"]["config"]["nested"] == {"a": 1, "b": 2}
    assert echo_data["config"]["config"]["extra"] is True
    assert echo_data["config"]["source"] == "external"
    assert echo_data["base_dir"] == local_path.parent
    assert echo_data["_source"]["config_type"] == "local"
    assert echo_data["_source"]["file"] == str(local_path)

    new_tool_data = raw_configs["new_tool"]
    assert new_tool_data["config"]["enabled"] is False
    assert new_tool_data["base_dir"] == local_path.parent
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

    def fake_load(self):  # noqa: D401
        data = copy.deepcopy(template_raw_configs)
        raw_calls.append(data)
        return data

    class DummyTool:
        class Input:  # pragma: no cover - structural placeholder
            pass

        class Output:  # pragma: no cover - structural placeholder
            pass

        def run(self):  # pragma: no cover - structural placeholder
            return None

    def fake_enrich(self, configs):  # noqa: D401
        enrich_calls.append(copy.deepcopy(configs))
        configs["sample"]["tool_class"] = DummyTool
        configs["sample"]["config"]["available"] = True

    def fake_convert(self, configs):  # noqa: D401
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
        converter._convert_mcp_tool("remote", {"description": "missing endpoint", "provider": "test"})

    with pytest.raises(ConfigError):
        converter._convert_mcp_tool("remote", {"description": "missing provider", "endpoint": "https://example"})

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
