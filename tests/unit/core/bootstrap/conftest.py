"""Shared fixtures for bootstrap tests."""

from unittest.mock import Mock

import pytest

from local_coding_assistant.core.protocols import IConfigManager


@pytest.fixture
def mock_config_manager():
    """Create a mock config manager."""
    mock = Mock(spec=IConfigManager)
    mock.load_global_config.return_value = {"logging": {"level": "INFO"}}
    mock.get_tools.return_value = {}
    mock.global_config = {}
    mock.session_overrides = {}
    return mock


@pytest.fixture
def mock_llm_manager():
    """Create a mock LLM manager."""
    mock = Mock()
    mock.generate.return_value = "Generated response"
    return mock


@pytest.fixture
def mock_tool_manager():
    """Create a mock tool manager."""
    mock = Mock()
    mock.list_tools.return_value = []
    return mock


@pytest.fixture
def mock_runtime_manager():
    """Create a mock runtime manager."""
    mock = Mock()
    mock.orchestrate.return_value = "Test response"
    return mock
