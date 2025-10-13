import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from local_coding_assistant.agent.llm_manager import (
    LLMManager,
    LLMConfig,
    LLMRequest,
    LLMResponse,
)


def make_test_llm_manager():
    """Create a test LLM manager setup for integration testing."""
    config = LLMConfig(
        model_name="gpt-3.5-turbo", provider="openai", temperature=0.5, max_tokens=100
    )

    with patch.object(LLMManager, "_setup_openai_client"):
        # Mock successful OpenAI response
        mock_response = MagicMock()
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage.total_tokens = 25
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response

        llm = LLMManager.__new__(LLMManager)
        llm.config = config
        llm.client = mock_client

        return llm


class TestLLMManagerIntegration:
    """Integration tests for LLMManager with real async behavior."""

    @pytest.mark.asyncio
    async def test_full_request_response_cycle(self):
        """Test complete request-response cycle."""
        config = LLMConfig(
            model_name="gpt-5-mini", provider="openai", temperature=0.5, max_tokens=100
        )

        with patch.object(LLMManager, "_setup_openai_client"):
            # Mock successful OpenAI response for NEW Responses API
            mock_response = MagicMock()
            mock_response.configure_mock(
                model="gpt-5-mini",
                usage=MagicMock(output_tokens=25),
                output=[MagicMock()],
            )
            # Configure the output structure for new Responses API
            mock_response.output[0].configure_mock(
                content=[MagicMock(text="Integration test response")]
            )

            mock_client = AsyncMock()
            mock_client.responses.create.return_value = mock_response

            llm = LLMManager.__new__(LLMManager)
            llm.config = config
            llm.client = mock_client

            request = LLMRequest(
                prompt="Integration test prompt",
                context={"test": "data"},
                system_prompt="You are a test assistant",
            )

            response = await llm.generate(request)

            # Verify response structure
            assert isinstance(response, LLMResponse)
            assert response.content == "Integration test response"
            assert response.model_used == "gpt-5-mini"
            assert response.tokens_used == 25

            # Verify OpenAI API was called correctly
            call_args = mock_client.responses.create.call_args
            messages = call_args[1]["input"]
            assert len(messages) == 3  # system + context + prompt
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "user"
            assert call_args[1]["temperature"] == 0.5
            assert call_args[1]["model"] == "gpt-5-mini"

    @pytest.mark.asyncio
    async def test_llm_tool_calls_are_executed(self):
        """Test that LLM-initiated tool calls are properly handled."""
        llm = make_test_llm_manager()

        # Mock LLM response with tool call
        original_generate = llm.generate

        async def mock_generate(request):
            if "sum" in request.prompt:
                return LLMResponse(
                    content="I'll calculate that for you.",
                    model_used="fake-model",
                    tokens_used=50,
                    tool_calls=[
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "sum",
                                "arguments": '{"a": 10, "b": 15}',
                            },
                        }
                    ],
                )
            return await original_generate(request)

        llm.generate = mock_generate

        # Test that LLM can generate tool calls
        request = LLMRequest(
            prompt="Calculate 10 + 15 using the sum tool",
            context={},
            system_prompt="You are a helpful assistant",
        )

        response = await llm.generate(request)

        # Should have generated tool call
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "sum"
        assert response.content == "I'll calculate that for you."
