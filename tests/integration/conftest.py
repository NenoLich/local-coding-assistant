import pytest
from typer.testing import CliRunner
from unittest.mock import AsyncMock, MagicMock, patch
from local_coding_assistant.agent.llm_manager import (
    LLMConfig,
    LLMManager,
    LLMRequest,
    LLMResponse,
)
from local_coding_assistant.cli.main import app as cli_app
from local_coding_assistant.config.loader import load_config
from local_coding_assistant.config.schemas import RuntimeConfig
from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.tools.builtin import SumTool
from local_coding_assistant.tools.tool_manager import ToolManager


@pytest.fixture(scope="function")
def ctx():
    """Fresh context per test to avoid state bleed."""
    return bootstrap()


@pytest.fixture(scope="session")
def app():
    """CLI app is static; session scope is fine."""
    return cli_app


@pytest.fixture(scope="function")
def cli_runner():
    """Provide a CliRunner instance for invoking CLI commands."""
    return CliRunner()


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response for testing."""
    return LLMResponse(
        content="[LLMManager] Echo: Received request with tool outputs",
        model_used="gpt-5-mini",
        tokens_used=50,
        tool_calls=None,
    )


@pytest.fixture
def mock_llm_manager(mock_llm_response):
    """
    Create a mock LLM manager that doesn't make real API calls.
    This mock supports both generate() and update_config().
    """
    from local_coding_assistant.core.exceptions import AgentError

    # Create a comprehensive mock without spec constraints
    mock_manager = MagicMock()
    mock_manager.config = LLMConfig(model_name="gpt-5-mini", provider="openai")

    async def mock_generate(request: LLMRequest) -> LLMResponse:
        # Use the model from the config if available, otherwise use the mock default
        model_used = (
            mock_manager.config.model_name
            if hasattr(mock_manager, "config") and mock_manager.config.model_name
            else "gpt-5-mini"
        )

        # Check if tool outputs are present and modify response accordingly
        if request.tool_outputs:
            content = "[LLMManager] Echo: Received request with tool outputs"
        else:
            content = f"[LLMManager] Echo: Received request with model {model_used}"

        return LLMResponse(
            content=content,
            model_used=model_used,
            tokens_used=50,
            tool_calls=None,
        )

    mock_manager.generate = AsyncMock(side_effect=mock_generate)

    def mock_update_config(
        *,
        model_name: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
    ) -> None:
        """Mock update_config with proper validation."""
        updates = {}
        if model_name is not None:
            updates["model_name"] = model_name
        if provider is not None:
            updates["provider"] = provider
        if temperature is not None:
            updates["temperature"] = temperature
        if max_tokens is not None:
            updates["max_tokens"] = max_tokens
        if api_key is not None:
            updates["api_key"] = api_key

        if not updates:
            return

        # Validate the new configuration (same logic as real LLM manager)
        try:
            old_config_dict = mock_manager.config.model_dump()
            new_config_dict = {**old_config_dict, **updates}
            # This will raise ValidationError if invalid
            LLMConfig(**new_config_dict)
        except Exception as e:
            raise AgentError(f"Configuration update validation failed: {e}") from e

        # If validation passes, update the mock config
        if model_name is not None:
            mock_manager.config.model_name = model_name
        if provider is not None:
            mock_manager.config.provider = provider
        if temperature is not None:
            mock_manager.config.temperature = temperature
        if max_tokens is not None:
            mock_manager.config.max_tokens = max_tokens
        if api_key is not None:
            mock_manager.config.api_key = api_key

    mock_manager.update_config = MagicMock(side_effect=mock_update_config)
    return mock_manager


@pytest.fixture
def ctx_with_mocked_llm(mock_llm_manager):
    """
    Create a context with mocked LLM manager to avoid API quota issues.
    """
    config = load_config()
    runtime_config = RuntimeConfig(
        persistent_sessions=False,
        max_session_history=100,
        enable_logging=True,
        log_level="INFO",
    )
    runtime = RuntimeManager(
        llm_manager=mock_llm_manager,
        tool_manager=ToolManager(),
        config=runtime_config,
    )
    # Register the SumTool for testing
    runtime._tool_manager.register_tool(SumTool)
    ctx = AppContext()
    ctx.register("llm", mock_llm_manager)
    ctx.register("tools", runtime._tool_manager)
    ctx.register("runtime", runtime)
    return ctx
