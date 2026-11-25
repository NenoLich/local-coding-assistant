"""Pytest configuration and fixtures for e2e CLI tests."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner


class _MockAsyncContextManager:
    """Async context manager that yields a fixed context dict."""

    def __init__(self, ctx):
        self.ctx = ctx

    async def __aenter__(self):
        return self.ctx

    async def __aexit__(self, *args):
        return False


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


@pytest.fixture
def mock_bootstrap_success():
    """Mock bootstrap + execute_query for the run command.

    The refactored ``run query`` uses ``async with bootstrap(...) as ctx`` and
    delegates execution to ``run_flow.execute_query``.
    """
    with (
        patch("local_coding_assistant.cli.commands.run.bootstrap") as mock_bootstrap,
        patch(
            "local_coding_assistant.cli.commands.run.execute_query"
        ) as mock_execute_query,
    ):
        mock_runtime = MagicMock()
        mock_runtime.event_pipeline = MagicMock()
        mock_ctx = {"runtime": mock_runtime}

        def _execute_query(runtime, text, options, console):
            report = MagicMock()
            report.final_answer = f"[LLMService] Echo: {text}"
            report.status = "success"
            return report, MagicMock(id="sess-1")

        mock_execute_query.side_effect = _execute_query
        mock_bootstrap.return_value = _MockAsyncContextManager(mock_ctx)

        yield mock_bootstrap, mock_ctx


@pytest.fixture
def mock_bootstrap_llm_service():
    """Mock bootstrap with LLM service for provider commands."""
    with patch(
        "local_coding_assistant.cli.commands.provider.bootstrap"
    ) as mock_bootstrap:
        mock_llm_service = MagicMock()
        mock_llm_service._provider_manager = MagicMock()
        mock_llm_service._provider_manager.list_providers.return_value = [
            "openai",
            "google",
        ]
        mock_llm_service._provider_manager.get_provider_source.side_effect = (
            lambda name: {"openai": "global", "google": "local"}.get(name)
        )
        mock_llm_service.get_provider_status_list.return_value = [
            {"name": "openai", "source": "global", "status": "available", "models": 2},
            {"name": "google", "source": "local", "status": "available", "models": 1},
        ]
        mock_llm_service.reload_providers = MagicMock()

        mock_ctx = {"llm": mock_llm_service}
        mock_bootstrap.return_value = _MockAsyncContextManager(mock_ctx)

        yield mock_bootstrap, mock_llm_service


@pytest.fixture
def mock_bootstrap_serve():
    """Mock bootstrap for serve command."""
    with (
        patch("local_coding_assistant.cli.commands.serve.bootstrap") as mock_bootstrap,
        patch("uvicorn.Server.serve", new_callable=AsyncMock),
    ):
        mock_store = MagicMock()
        mock_ctx = {"runtime": MagicMock(), "execution_store": mock_store}
        mock_bootstrap.return_value = _MockAsyncContextManager(mock_ctx)
        yield mock_bootstrap, mock_ctx


@pytest.fixture
def patch_tool_context():
    """Patch ToolCLIContext.create for tool commands.

    The refactored tool commands use ``ToolCLIContext.create(...)`` instead of
    ``bootstrap``.
    """
    from local_coding_assistant.cli.commands.tool import ToolCLIContext

    with patch.object(ToolCLIContext, "create") as mock_create:

        class _MockToolContextCM:
            def __init__(self, tools):
                self.tools = tools

            async def __aenter__(self):
                ctx = MagicMock()
                ctx.tool_manager = MagicMock()
                ctx.tool_manager.tools = self.tools
                return ctx

            async def __aexit__(self, *args):
                return False

        mock_create.return_value = _MockToolContextCM([])
        yield mock_create
