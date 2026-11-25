"""Deterministic tool schema tests for tools_platform contracts.

Tests that tool schemas are deterministic, stable, and suitable for LLM consumption.
"""

import json

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    OperationEffect,
    ToolDefinition,
    ToolRiskClass,
)


class TestDeterministicSchemas:
    """Test that tool schemas are deterministic and stable."""

    def test_tool_definition_schema_is_deterministic(self):
        """Test that tool definition produces consistent JSON schema."""
        definition = ToolDefinition(
            id="test_tool",
            version="1.0.0",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            output_schema={"type": "object"},
        )

        # Serialize to JSON multiple times
        schema1 = json.dumps(definition.input_schema, sort_keys=True)
        schema2 = json.dumps(definition.input_schema, sort_keys=True)

        # Should be identical
        assert schema1 == schema2

    def test_tool_definition_schema_stability(self):
        """Test that tool definition schema is stable across instances."""
        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }

        definition1 = ToolDefinition(
            id="write_tool",
            version="1.0.0",
            description="Write a file",
            input_schema=schema,
            output_schema={"type": "object"},
        )

        definition2 = ToolDefinition(
            id="write_tool",
            version="1.0.0",
            description="Write a file",
            input_schema=schema,
            output_schema={"type": "object"},
        )

        # Schemas should be identical
        assert definition1.input_schema == definition2.input_schema

    def test_tool_schema_is_llm_friendly(self):
        """Test that tool schemas are suitable for LLM consumption."""
        definition = ToolDefinition(
            id="search_tool",
            version="1.0.0",
            description="Search for text in files",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to search in",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether search is case sensitive",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "line": {"type": "integer"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                    "count": {"type": "integer"},
                },
            },
        )

        # Should have clear descriptions
        assert definition.description != ""
        assert "description" in definition.input_schema["properties"]["query"]

        # Should have required fields marked
        assert "required" in definition.input_schema
        assert "query" in definition.input_schema["required"]

        # Output schema should be structured
        assert "type" in definition.output_schema
        assert definition.output_schema["type"] == "object"

    def test_tool_schema_order_independence(self):
        """Test that schema property order doesn't affect equality."""
        schema1 = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
                "c": {"type": "boolean"},
            },
        }

        schema2 = {
            "type": "object",
            "properties": {
                "c": {"type": "boolean"},
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
        }

        definition1 = ToolDefinition(
            id="test",
            version="1.0.0",
            description="Test",
            input_schema=schema1,
            output_schema={"type": "object"},
        )

        definition2 = ToolDefinition(
            id="test",
            version="1.0.0",
            description="Test",
            input_schema=schema2,
            output_schema={"type": "object"},
        )

        # Schemas should be equal despite different property order
        assert definition1.input_schema == definition2.input_schema

    def test_tool_schema_serialization_roundtrip(self):
        """Test that schema serialization roundtrip preserves data."""
        original_schema = {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["file_path"],
        }

        definition = ToolDefinition(
            id="read_tool",
            version="1.0.0",
            description="Read a file",
            input_schema=original_schema,
            output_schema={"type": "object"},
        )

        # Serialize
        data = definition.model_dump()

        # Deserialize
        restored = ToolDefinition.model_validate(data)

        # Schema should be preserved
        assert restored.input_schema == original_schema
        assert restored.input_schema["properties"]["start_line"]["minimum"] == 1

    def test_tool_definition_metadata_stability(self):
        """Test that tool definition metadata is stable."""
        definition = ToolDefinition(
            id="stable_tool",
            version="1.0.0",
            description="A stable tool",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            permissions=["read"],
            risk_class=ToolRiskClass.LOW,
            effects=[OperationEffect.READ],
            supported_backends=[BackendType.HOST],
        )

        # All metadata should be preserved
        assert definition.id == "stable_tool"
        assert definition.version == "1.0.0"
        assert definition.permissions == ["read"]
        assert definition.risk_class == ToolRiskClass.LOW
        assert len(definition.effects) == 1
        assert definition.effects[0] == OperationEffect.READ
        assert len(definition.supported_backends) == 1
        assert definition.supported_backends[0] == BackendType.HOST

    def test_tool_schema_no_ambiguous_types(self):
        """Test that tool schemas avoid ambiguous types."""
        # Good schema: specific types
        good_schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "name": {"type": "string"},
                "enabled": {"type": "boolean"},
            },
        }

        definition = ToolDefinition(
            id="typed_tool",
            version="1.0.0",
            description="Tool with specific types",
            input_schema=good_schema,
            output_schema={"type": "object"},
        )

        # All properties should have specific types
        for prop_name, prop_def in definition.input_schema["properties"].items():
            assert "type" in prop_def
            assert prop_def["type"] in [
                "string",
                "integer",
                "boolean",
                "object",
                "array",
            ]

    def test_tool_schema_default_values(self):
        """Test that tool schemas can specify default values."""
        schema = {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 100},
                "case_sensitive": {"type": "boolean", "default": False},
                "pattern": {"type": "string"},
            },
            "required": ["pattern"],
        }

        definition = ToolDefinition(
            id="search_tool",
            version="1.0.0",
            description="Search with defaults",
            input_schema=schema,
            output_schema={"type": "object"},
        )

        # Defaults should be preserved
        assert definition.input_schema["properties"]["limit"]["default"] == 100
        assert (
            definition.input_schema["properties"]["case_sensitive"]["default"] is False
        )

        # Required fields should not have defaults in the required list
        assert "pattern" in definition.input_schema["required"]
        assert "limit" not in definition.input_schema["required"]
