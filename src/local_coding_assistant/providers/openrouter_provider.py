"""
OpenRouter provider implementation
"""

from __future__ import annotations

from typing import Any

from local_coding_assistant.config.schemas import ModelConfig
from local_coding_assistant.providers.base import BaseProvider
from local_coding_assistant.providers.provider_manager import register_provider
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("providers.openrouter")


@register_provider("openrouter")
class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter API"""

    def __init__(
        self,
        api_key: str | None = None,
        api_key_env: str | None = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1",
        models: list[ModelConfig] | list[str] | dict[str, dict] | None = None,
        driver: str = "openai_responses",  # Use OpenAI responses driver
        **kwargs: Any,
    ):
        # Default models if not specified
        default_models = (
            [
                ModelConfig(
                    name="qwen/qwen3-coder:free",
                    supported_parameters=[
                        "tools",
                        "tool_choice",
                        "temperature",
                        "max_tokens",
                        "top_p",
                        "stop",
                        "presence_penalty",
                        "top_k",
                        "frequency_penalty",
                        "seed",
                    ],
                ),
                ModelConfig(
                    name="meta-llama/llama-3.3-70b-instruct:free",
                    supported_parameters=[
                        "tools",
                        "tool_choice",
                        "temperature",
                        "max_tokens",
                        "top_p",
                        "top_k",
                        "stop",
                        "presence_penalty",
                        "frequency_penalty",
                    ],
                ),
            ]
            if models is None
            else models
        )

        # Set provider name if not provided
        name = kwargs.pop("name", "openrouter")

        super().__init__(
            name=name,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            models=default_models,
            driver=driver,
            **kwargs,
        )

    def _create_driver_instance(self) -> Any:
        """Create and return a driver instance for OpenRouter.

        Returns:
            An instance of a BaseDriver configured for OpenRouter

        Raises:
            RuntimeError: If the driver cannot be initialized
        """
        try:
            # Use the parent's _initialize_driver helper method
            return self._initialize_driver()
        except Exception as e:
            logger.error(
                "Failed to create OpenRouter driver", error=str(e), exc_info=True
            )
            raise RuntimeError(f"Failed to initialize OpenRouter driver: {e!s}") from e
