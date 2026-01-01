"""Pytest configuration and fixtures for e2e CLI tests."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner


@pytest.fixture
def cli_runner():
    """Provide a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_providers_config(test_configs):
    """Create a temporary providers configuration file."""
    config_dir = test_configs["config_dir"]
    config_dir.mkdir(parents=True, exist_ok=True)

    providers_file = config_dir / "providers.local.yaml"
    providers_data = {
        "test_openai": {
            "driver": "openai_chat",
            "api_key_env": "OPENAI_API_KEY",
            "models": {"gpt-4": {}, "gpt-3.5-turbo": {}},
        },
        "test_google": {
            "driver": "openai_chat",
            "api_key": "fake-key",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "models": {"gemini-pro": {}},
        },
    }

    with open(providers_file, "w") as f:
        yaml.dump(providers_data, f, default_flow_style=False)

    return providers_file


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    test_env = {
        "LOCCA_TEST_MODE": "true",
        "OPENAI_API_KEY": "test-openai-key",
        "LOCCA_LOG_LEVEL": "INFO",
        "LOCCA_MODEL": "gpt-4",
    }

    with patch.dict(os.environ, test_env):
        yield test_env


from typing import Any, cast


@pytest.fixture
def patch_tool_bootstrap():
    """Patch the tool CLI bootstrap helper."""

    with patch("local_coding_assistant.cli.commands.tool.bootstrap") as mock_bootstrap:
        yield mock_bootstrap


@pytest.fixture
def mock_bootstrap_success():
    """Mock successful bootstrap for CLI commands that use it."""
    # Patch the bootstrap where it's actually imported and used
    with patch("local_coding_assistant.cli.commands.run.bootstrap") as mock_bootstrap:
        # Create mock context with proper typing
        mock_runtime = MagicMock()
        mock_llm = MagicMock()
        mock_tools = [MagicMock(name="test_tool")]

        # Create typed mock context
        mock_ctx: dict[str, Any] = {
            "runtime": mock_runtime,
            "llm": mock_llm,
            "tools": mock_tools,
        }

        # Set return value first to establish the type
        mock_bootstrap.return_value = mock_ctx

        # Create an async function for the orchestrate mock
        async def mock_orchestrate_async(*args, **kwargs):
            return {"message": "[LLMManager] Echo: test query"}

        # Use cast to help the type checker understand the type of mock_runtime
        runtime = cast(MagicMock, mock_ctx["runtime"])
        runtime.orchestrate = AsyncMock(side_effect=mock_orchestrate_async)

        yield mock_bootstrap, mock_ctx


@pytest.fixture
def mock_bootstrap_llm_manager():
    """Mock bootstrap with LLM manager for provider commands."""
    # Patch the bootstrap where it's actually imported and used
    with patch(
        "local_coding_assistant.cli.commands.provider.bootstrap"
    ) as mock_bootstrap:
        # Create mock LLM manager
        mock_llm_manager = MagicMock()
        mock_llm_manager.provider_manager = MagicMock()
        mock_llm_manager.provider_manager.list_providers.return_value = [
            "openai",
            "google",
        ]
        mock_llm_manager.provider_manager.get_provider_source.side_effect = (
            lambda name: {"openai": "global", "google": "local"}.get(name)
        )
        mock_llm_manager.get_provider_status_list.return_value = [
            {"name": "openai", "source": "global", "status": "available", "models": 2},
            {"name": "google", "source": "local", "status": "available", "models": 1},
        ]
        mock_llm_manager.reload_providers = MagicMock()

        mock_ctx = {"llm": mock_llm_manager}
        mock_bootstrap.return_value = mock_ctx

        yield mock_bootstrap, mock_llm_manager


@pytest.fixture
def mock_bootstrap_tools():
    """Mock bootstrap with tools registry for list-tools command."""
    # Patch the bootstrap where it's actually imported and used
    with patch(
        "local_coding_assistant.cli.commands.list_tools.bootstrap"
    ) as mock_bootstrap:
        # Create mock tools registry with proper attribute access
        mock_tools = []

        # Create search_web tool mock
        search_web_tool = MagicMock()
        search_web_tool.name = "search_web"
        search_web_tool.__name__ = "search_web"
        search_web_tool.__class__.__name__ = "WebSearchTool"
        mock_tools.append(search_web_tool)

        # Create read_file tool mock
        read_file_tool = MagicMock()
        read_file_tool.name = "read_file"
        read_file_tool.__name__ = "read_file"
        read_file_tool.__class__.__name__ = "FileReadTool"
        mock_tools.append(read_file_tool)

        mock_ctx = {"tools": mock_tools}
        mock_bootstrap.return_value = mock_ctx

        yield mock_bootstrap, mock_tools


@pytest.fixture
def mock_bootstrap_config():
    """Mock bootstrap for config commands."""
    # Note: config commands don't actually use bootstrap, but keeping fixture for consistency
    mock_ctx = {"config": MagicMock()}
    return MagicMock(), mock_ctx


@pytest.fixture
def mock_bootstrap_serve():
    """Mock bootstrap for serve command."""
    # Patch the bootstrap where it's actually imported and used
    with patch("local_coding_assistant.cli.commands.serve.bootstrap") as mock_bootstrap:
        mock_ctx = {"runtime": MagicMock()}
        mock_bootstrap.return_value = mock_ctx
        yield mock_bootstrap, mock_ctx
