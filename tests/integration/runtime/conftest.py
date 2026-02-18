"""
Fixtures and test utilities for handler integration testing.
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock

from local_coding_assistant.agent.llm import LLMResult
from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.runtime.handlers.handler_types import HandlerOutput


class MockLLMDriver:
    """Mock LLM driver that can be configured to return specific responses."""

    def __init__(self, responses: List[LLMResult]):
        self.responses = responses
        self.call_count = 0
        self.generate_calls = []

    async def generate(self, prompt: str, **kwargs) -> LLMResult:
        """Return next configured response."""
        self.generate_calls.append({"prompt": prompt, "kwargs": kwargs})

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            # Default to last response if we run out
            response = self.responses[-1]

        self.call_count += 1
        return response

    async def stream(self, prompt: str, **kwargs):
        """Mock streaming - not used in these tests."""
        yield "mock_stream_response"


class MockToolManager:
    """Mock tool manager that can be configured to return specific responses."""

    def __init__(self, responses: Dict[str, List[Any]]):
        self.responses = responses
        self.call_counts = {}
        self.execution_calls = []

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Return next configured response for tool."""
        if tool_name not in self.call_counts:
            self.call_counts[tool_name] = 0

        self.execution_calls.append({"tool_name": tool_name, "tool_args": tool_args})

        tool_responses = self.responses.get(tool_name, [])
        if self.call_counts[tool_name] < len(tool_responses):
            response = tool_responses[self.call_counts[tool_name]]
        else:
            # Default to last response if we run out
            response = (
                tool_responses[-1]
                if tool_responses
                else MockToolResponse.success_response()
            )

        self.call_counts[tool_name] += 1
        return response


def create_test_session() -> SessionState:
    """Create a test session state."""
    return SessionState(
        id="test_session_123",
        current_task="Test task for handler integration",
        history=[],
        tool_calls=[],
    )


def assert_handler_output(
    output: HandlerOutput,
    expected_status: ExecutionStatus,
    should_retry: Optional[bool] = None,
    has_continuation: Optional[bool] = None,
    has_adjusted_options: Optional[bool] = None,
    expected_strategy: Optional[str] = None,
):
    """Assert handler output has expected characteristics."""
    assert output.status == expected_status, (
        f"Expected status {expected_status}, got {output.status}"
    )

    if should_retry is not None:
        assert output.should_retry == should_retry, (
            f"Expected should_retry {should_retry}, got {output.should_retry}"
        )

    if has_continuation is not None:
        has_cont = output.template_path is not None
        assert has_cont == has_continuation, (
            f"Expected template path {has_continuation}, got {has_cont}"
        )

    if has_adjusted_options is not None:
        has_opts = output.adjusted_llm_options is not None
        assert has_opts == has_adjusted_options, (
            f"Expected adjusted options {has_adjusted_options}, got {has_opts}"
        )

    if expected_strategy is not None:
        # Strategy is embedded in template path, not a separate field
        assert expected_strategy in (output.template_path or "").lower(), (
            f"Expected strategy '{expected_strategy}' in template path"
        )


def assert_session_continuation(
    session: SessionState,
    should_have_continuation: bool = True,
    expected_keywords: Optional[List[str]] = None,
):
    """Assert session has been updated with handler context."""
    if should_have_continuation:
        assert "handler_context" in session.metadata, (
            "Session should have handler context in metadata"
        )

        handler_context = session.metadata["handler_context"]
        assert "template_path" in handler_context, (
            "Handler context should have template path"
        )

        if expected_keywords:
            template_path = handler_context["template_path"].lower()
            for keyword in expected_keywords:
                assert keyword in template_path, (
                    f"Expected keyword '{keyword}' in template path"
                )
    else:
        # Should not have added handler context
        assert "handler_context" not in session.metadata, (
            "Session should not have handler context in metadata"
        )


def capture_execution_calls(
    mock_tool_manager: MockToolManager,
) -> Dict[str, List[Dict[str, Any]]]:
    """Capture and return all tool execution calls."""
    return {
        tool_name: [
            call
            for call in mock_tool_manager.execution_calls
            if call["tool_name"] == tool_name
        ]
        for tool_name in mock_tool_manager.call_counts.keys()
    }


def verify_llm_options_adjustment(
    adjusted_options: Optional[Dict[str, Any]],
    expected_max_tokens_increase: bool = False,
    expected_min_max_tokens: Optional[int] = None,
):
    """Verify LLM options were adjusted correctly."""
    if expected_max_tokens_increase:
        assert adjusted_options is not None, (
            "Expected adjusted options for max tokens increase"
        )
        # Check for both possible key formats
        has_max_tokens = (
            "max_tokens" in adjusted_options or "llm.max_tokens" in adjusted_options
        )
        assert has_max_tokens, "Expected max_tokens in adjusted options"

        if expected_min_max_tokens:
            # Extract max_tokens value from either key format
            max_tokens_value = adjusted_options.get(
                "max_tokens"
            ) or adjusted_options.get("llm.max_tokens")
            assert max_tokens_value >= expected_min_max_tokens, (
                f"Expected max_tokens >= {expected_min_max_tokens}, got {max_tokens_value}"
            )
    else:
        assert adjusted_options is None or not any(
            key in adjusted_options for key in ["max_tokens", "llm.max_tokens"]
        ), "Expected no max_tokens adjustment"


# Import mock response factories
from tests.integration.runtime.mock_llm_responses import MockLLMResponse
from tests.integration.runtime.mock_tool_responses import MockToolResponse, MockToolCall


@pytest.fixture
def mock_llm_driver():
    """Mock LLM driver for handler integration tests."""
    return MockLLMDriver([])


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager for handler integration tests."""
    from tests.integration.runtime.conftest import MockToolManager

    return MockToolManager({})


@pytest.fixture
def test_session():
    """Test session for handler integration tests."""
    from tests.integration.runtime.conftest import create_test_session

    return create_test_session()
