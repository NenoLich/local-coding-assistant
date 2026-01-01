"""Integration tests for sandbox bootstrap and configuration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.sandbox.manager import SandboxManager


@pytest.fixture
def mock_env_manager(tmp_path):
    with patch("local_coding_assistant.core.bootstrap.EnvManager") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.create.return_value = mock_instance

        # Configure path_manager on the env_manager mock
        mock_path_manager = MagicMock()
        mock_instance.path_manager = mock_path_manager

        def side_effect(path, **kwargs):
            if isinstance(path, str) and path.startswith("@"):
                # Handle special paths by just returning a temp path or ignoring
                return tmp_path / path.replace("@", "").replace("/", "_")
            return Path(path)

        mock_path_manager.resolve_path.side_effect = side_effect
        yield mock_cls


@pytest.fixture
def mock_dependencies():
    with (
        patch(
            "local_coding_assistant.core.bootstrap._initialize_llm_manager"
        ) as mock_llm,
        patch(
            "local_coding_assistant.core.bootstrap._initialize_runtime_manager"
        ) as mock_runtime,
        patch("local_coding_assistant.tools.tool_manager.ToolManager") as mock_tool_cls,
        patch(
            "local_coding_assistant.config.tool_loader.ToolLoader"
        ) as mock_tool_loader_cls,
    ):
        # Configure mocks
        mock_tool_manager = MagicMock()
        mock_tool_cls.return_value = mock_tool_manager

        mock_tool_loader = MagicMock()
        mock_tool_loader_cls.return_value = mock_tool_loader
        mock_tool_loader.load_tool_configs.return_value = {}

        yield {
            "llm": mock_llm,
            "runtime": mock_runtime,
            "tool_cls": mock_tool_cls,
            "tool_instance": mock_tool_manager,
            "tool_loader_cls": mock_tool_loader_cls,
        }


def test_sandbox_bootstrap_initialization(
    mock_env_manager, mock_dependencies, tmp_path
):
    """Verify that SandboxManager is initialized and injected correctly during bootstrap."""

    # Create a temporary config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
sandbox:
  enabled: true
  image: "custom-sandbox:latest"
    """)

    # Run bootstrap
    ctx = bootstrap(config_path=str(config_file))

    # Verify SandboxManager is registered in context
    sandbox_manager = ctx.get("sandbox")
    assert isinstance(sandbox_manager, SandboxManager)
    assert sandbox_manager.config.enabled is True
    assert sandbox_manager.config.image == "custom-sandbox:latest"

    # Verify PathManager is injected
    # Verify PathManager is injected
    assert sandbox_manager.path_manager is not None
    # Cannot check isinstance with mock, so check if it has resolve_path
    assert hasattr(sandbox_manager.path_manager, "resolve_path")

    # Verify ToolManager was initialized with sandbox_manager
    mock_dependencies["tool_cls"].assert_called_once()
    call_kwargs = mock_dependencies["tool_cls"].call_args.kwargs
    assert call_kwargs.get("sandbox_manager") is sandbox_manager


def test_sandbox_defaults_loading(mock_env_manager, mock_dependencies):
    """Verify that sandbox defaults are loaded correctly."""

    # Run bootstrap without config file (should use defaults)
    ctx = bootstrap()

    sandbox_manager = ctx.get("sandbox")
    assert isinstance(sandbox_manager, SandboxManager)

    # Check default values from defaults.yaml
    assert sandbox_manager.config.enabled is False
    assert sandbox_manager.config.image == "locca-sandbox:latest"
    assert sandbox_manager.config.memory_limit == "512m"
