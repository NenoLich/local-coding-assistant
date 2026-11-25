"""Unit tests for the ExposureManager.

Tests tool exposure policy filtering based on backend compatibility,
config exposed_tools list, and tool.exposure_policy conditions.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ToolDefinition,
    ToolRiskClass,
)
from local_coding_assistant.tools_platform.exposure_manager import ExposureManager


def _make_tool_def(
    tool_id: str,
    description: str,
    risk_class: ToolRiskClass = ToolRiskClass.LOW,
    supported_backends: list[BackendType] | None = None,
    exposure_policy: dict[str, Any] | None = None,
) -> ToolDefinition:
    """Helper to create a ToolDefinition for testing."""
    from local_coding_assistant.tools_platform.definitions import ToolBuilder

    builder = ToolBuilder(tool_id=tool_id, description=description)
    builder = builder.with_risk_class(risk_class)
    if supported_backends is not None:
        builder = builder.with_backends(*supported_backends)
    if exposure_policy is not None:
        builder = builder.with_exposure_policy(exposure_policy)
    # Add minimal implementation to satisfy build_with_implementation
    from pydantic import BaseModel

    class DummyInput(BaseModel):
        pass

    class DummyOutput(BaseModel):
        pass

    builder = builder.with_implementation(
        factory=MagicMock(),
        input_model=DummyInput,
        output_model=DummyOutput,
    )
    definition, _ = builder.build_with_implementation()
    return definition


@pytest.fixture
def exposure_manager(config_manager) -> ExposureManager:
    """ExposureManager instance for tests."""
    return ExposureManager(config_manager=config_manager)


class TestIsExposed:
    """Test is_exposed method of ExposureManager."""

    def test_exposed_all_backend_compatible_no_policy(
        self, exposure_manager: ExposureManager
    ) -> None:
        """Without exposure policy, all backend-compatible tools are exposed."""
        tool = _make_tool_def(
            "tool1", "Test tool 1", supported_backends=[BackendType.HOST]
        )
        assert exposure_manager.is_exposed(tool) is True

    def test_not_exposed_wrong_backend(self, exposure_manager: ExposureManager) -> None:
        """Tool with Docker-only backend is not exposed on HOST backend."""
        tool = _make_tool_def(
            "tool1", "Test tool 1", supported_backends=[BackendType.DOCKER]
        )
        assert exposure_manager.is_exposed(tool) is False

    def test_exposed_with_allowed_tools_list(
        self, exposure_manager: ExposureManager, config_manager
    ) -> None:
        """Tool is exposed only if in allowed_tools list from config."""
        config_manager.global_config.tools_platform.exposed_tools = ["allowed_tool"]
        tool = _make_tool_def(
            "allowed_tool", "Allowed tool", supported_backends=[BackendType.HOST]
        )
        assert exposure_manager.is_exposed(tool) is True

    def test_not_excluded_with_disallowed_tool(
        self, exposure_manager: ExposureManager, config_manager
    ) -> None:
        """Tool not in allowed list is not exposed."""
        config_manager.global_config.tools_platform.exposed_tools = ["allowed_tool"]
        tool = _make_tool_def(
            "disallowed_tool", "Disallowed tool", supported_backends=[BackendType.HOST]
        )
        assert exposure_manager.is_exposed(tool) is False

    def test_exposed_with_per_tool_exposure_policy(
        self, exposure_manager: ExposureManager
    ) -> None:
        """Tool with per-tool exposure_policy is checked (but manager only uses config policy)."""
        tool = _make_tool_def(
            "policy_tool",
            "Policy tool",
            exposure_policy={"disabled_tools": ["policy_tool"]},
            supported_backends=[BackendType.HOST],
        )
        # Note: ExposureManager only checks config-level exposure_policy,
        # not per-tool exposure_policy. Tool's own exposure_policy is stored
        # but not used for exposure determination.
        # This test documents the current behavior.
        assert exposure_manager.is_exposed(tool) is True


class TestGetExposedTools:
    """Test get_exposed_tools method."""

    def test_returns_exposed_tools(
        self, exposure_manager: ExposureManager, config_manager
    ) -> None:
        """get_exposed_tools returns only exposed tools."""
        config_manager.global_config.tools_platform.exposed_tools = "auto"
        tool1 = _make_tool_def("tool1", "Tool 1", supported_backends=[BackendType.HOST])
        tool2 = _make_tool_def(
            "tool2", "Tool 2", supported_backends=[BackendType.DOCKER]
        )
        tool3 = _make_tool_def("tool3", "Tool 3", supported_backends=[BackendType.HOST])

        # Register tools
        from local_coding_assistant.tools_platform.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(tool1, MagicMock())
        registry.register(tool2, MagicMock())
        registry.register(tool3, MagicMock())

        exposed = exposure_manager.get_exposed_tools(registry)
        # With auto policy and HOST backend, all HOST-backed tools are exposed
        assert len(exposed) == 2
        exposed_ids = {t.id for t in exposed}
        assert "tool1" in exposed_ids
        assert "tool3" in exposed_ids
        assert "tool2" not in exposed_ids

    def test_empty_registry(self, exposure_manager: ExposureManager) -> None:
        """get_exposed_tools returns empty list for empty registry."""
        from local_coding_assistant.tools_platform.registry import ToolRegistry

        registry = ToolRegistry()
        exposed = exposure_manager.get_exposed_tools(registry)
        assert exposed == []

    def test_all_tools_exposed_auto_policy(
        self, exposure_manager: ExposureManager, config_manager
    ) -> None:
        """With auto policy, all backend-compatible tools are exposed."""
        config_manager.global_config.tools_platform.exposed_tools = "auto"

        from local_coding_assistant.tools_platform.registry import ToolRegistry

        registry = ToolRegistry()
        tool1 = _make_tool_def("tool1", "Tool 1", supported_backends=[BackendType.HOST])
        tool2 = _make_tool_def(
            "tool2", "Tool 2", supported_backends=[BackendType.DOCKER]
        )
        registry.register(tool1, MagicMock())
        registry.register(tool2, MagicMock())

        exposed = exposure_manager.get_exposed_tools(registry)
        # Only HOST-backed tool should be exposed when backend is HOST
        assert len(exposed) == 1
        assert exposed[0].id == "tool1"


class TestGetExposedToolSchemas:
    """Test get_exposed_tool_schemas method."""

    def test_returns_schema_dicts(
        self, exposure_manager: ExposureManager, config_manager
    ) -> None:
        """get_exposed_tool_schemas returns list of schema dicts."""
        config_manager.global_config.tools_platform.exposed_tools = "auto"

        from local_coding_assistant.tools_platform.registry import ToolRegistry

        registry = ToolRegistry()
        tool = _make_tool_def(
            "test_tool", "Test tool", supported_backends=[BackendType.HOST]
        )
        registry.register(tool, MagicMock())

        schemas = exposure_manager.get_exposed_tool_schemas(registry)
        assert len(schemas) == 1
        assert "id" in schemas[0]
        assert "description" in schemas[0]
        assert "input_schema" in schemas[0]

    def test_schema_excludes_internal_fields(
        self, exposure_manager: ExposureManager, config_manager
    ) -> None:
        """Schema excludes internal fields via model_llm_schema."""
        config_manager.global_config.tools_platform.exposed_tools = "auto"

        from local_coding_assistant.tools_platform.registry import ToolRegistry

        registry = ToolRegistry()
        tool = _make_tool_def(
            "test_tool", "Test tool", supported_backends=[BackendType.HOST]
        )
        registry.register(tool, MagicMock())

        schemas = exposure_manager.get_exposed_tool_schemas(registry)
        schema = schemas[0]
        # model_llm_schema should exclude internal fields like file_observation
        assert "file_observation" not in str(schema.get("input_schema", {}))
        assert "file_observation" not in str(schema.get("output_schema", {}))


class TestHiddenTools:
    """Test that hidden tools are absent from LLM schemas."""

    def test_hidden_tools_not_in_schemas(
        self, exposure_manager: ExposureManager, config_manager
    ) -> None:
        """Hidden tools (not exposed) are absent from LLM schemas."""
        config_manager.global_config.tools_platform.exposed_tools = ["allowed_only"]

        from local_coding_assistant.tools_platform.registry import ToolRegistry

        registry = ToolRegistry()
        hidden_tool = _make_tool_def(
            "hidden_tool", "Hidden tool", supported_backends=[BackendType.HOST]
        )
        allowed_tool = _make_tool_def(
            "allowed_only", "Allowed only tool", supported_backends=[BackendType.HOST]
        )
        registry.register(hidden_tool, MagicMock())
        registry.register(allowed_tool, MagicMock())

        schemas = exposure_manager.get_exposed_tool_schemas(registry)
        schema_ids = {s["id"] for s in schemas}
        assert "hidden_tool" not in schema_ids
        assert "allowed_only" in schema_ids
