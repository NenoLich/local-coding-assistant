"""Unit tests for core.dependencies module."""

from unittest.mock import Mock

import pytest

from local_coding_assistant.core.dependencies import AppDependencies
from local_coding_assistant.core.protocols import IConfigManager


class TestAppDependencies:
    """Test AppDependencies functionality."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager."""
        return Mock(spec=IConfigManager)

    def test_initialization(self, mock_config_manager):
        """Test initialization with required dependencies."""
        deps = AppDependencies(config_manager=mock_config_manager)
        assert deps.config_manager is mock_config_manager
        assert deps.llm_manager is None
        assert deps.tool_manager is None
        assert deps.runtime_manager is None
        assert not deps.is_initialized()

    def test_mark_initialized(self, mock_config_manager):
        """Test marking dependencies as initialized."""
        deps = AppDependencies(config_manager=mock_config_manager)
        deps.mark_initialized()
        assert deps.is_initialized()

    def test_optional_dependencies(self, mock_config_manager):
        """Test setting optional dependencies."""
        deps = AppDependencies(
            config_manager=mock_config_manager,
            llm_manager="llm_mock",
            tool_manager="tool_mock",
            runtime_manager="runtime_mock"
        )
        assert deps.llm_manager == "llm_mock"
        assert deps.tool_manager == "tool_mock"
        assert deps.runtime_manager == "runtime_mock"
        assert not deps.is_initialized()

    def test_initialization_requires_config_manager(self):
        """Test that config_manager is required for initialization."""
        with pytest.raises(TypeError):
            AppDependencies()  # type: ignore

    def test_repr(self, mock_config_manager):
        """Test string representation of AppDependencies."""
        deps = AppDependencies(
            config_manager=mock_config_manager,
            llm_manager="llm_mock",
            tool_manager="tool_mock",
            runtime_manager="runtime_mock"
        )
        assert "AppDependencies(" in repr(deps)
        assert "config_manager=" in repr(deps)
        assert "llm_manager='llm_mock'" in repr(deps)
        assert "tool_manager='tool_mock'" in repr(deps)
        assert "runtime_manager='runtime_mock'" in repr(deps)
