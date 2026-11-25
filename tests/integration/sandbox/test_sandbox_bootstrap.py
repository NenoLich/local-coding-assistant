"""Integration tests for sandbox bootstrap and configuration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.sandbox.manager import SandboxManager


@pytest.fixture
def mock_env_manager(tmp_path):
    with patch("local_coding_assistant.core.bootstrap.get_env_manager") as mock_get:
        mock_instance = MagicMock()
        mock_get.return_value = mock_instance

        # Configure path_manager on the env_manager mock
        mock_path_manager = MagicMock()
        mock_instance.path_manager = mock_path_manager

        def side_effect(path, **kwargs):
            if isinstance(path, str) and path.startswith("@"):
                # Handle special paths by just returning a temp path or ignoring
                return tmp_path / path.replace("@", "").replace("/", "_")
            return Path(path)

        mock_path_manager.resolve_path.side_effect = side_effect
        yield mock_instance


@pytest.fixture
def mock_dependencies():
    with (
        patch(
            "local_coding_assistant.core.bootstrap._initialize_llm_service"
        ) as mock_llm,
        patch(
            "local_coding_assistant.core.bootstrap._initialize_runtime_manager"
        ) as mock_runtime,
        patch(
            "local_coding_assistant.config.tool_loader.ToolLoader"
        ) as mock_tool_loader_cls,
        patch(
            "local_coding_assistant.sandbox.manager.SandboxManager.ensure_availability"
        ) as mock_availability,
    ):
        # Configure mocks
        mock_tool_loader = MagicMock()
        mock_tool_loader_cls.return_value = mock_tool_loader
        mock_tool_loader.load_tool_configs.return_value = {}

        # Mock availability to return True (Docker is available)
        mock_availability.return_value = True

        yield {
            "llm": mock_llm,
            "runtime": mock_runtime,
            "tool_loader_cls": mock_tool_loader_cls,
        }
