"""
Google Gemini provider implementation
"""

from __future__ import annotations

from typing import Any

from local_coding_assistant.config.schemas import ModelConfig
from local_coding_assistant.providers import base
from local_coding_assistant.providers.provider_manager import register_provider
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("providers.google")


@register_provider("google_gemini")
class GoogleGeminiProvider(base.BaseProvider):
    """Provider for Google Gemini API"""

    def __init__(
        self,
        api_key: str | None = None,
        api_key_env: str | None = "GEMINI_API_KEY",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        models: list[ModelConfig] | list[str] | dict[str, dict] | None = None,
        driver: str = "openai_chat",  # Use OpenAI chat driver
        **kwargs: Any,
    ):
        # Default models if not specified
        default_models = (
            [
                ModelConfig(
                    name="gemini-2.5-flash",
                    supported_parameters=[
                        "temperature",
                        "max_tokens",
                        "top_p",
                        "top_k",
                        "stop",
                        "presence_penalty",
                        "frequency_penalty",
                    ],
                )
            ]
            if models is None
            else models
        )

        # Set provider name if not provided
        name = kwargs.pop("name", "google_gemini")

        super().__init__(
            name=name,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            models=default_models,
            driver=driver,
            **kwargs,
        )

    def _create_driver_instance(self) -> base.BaseDriver:
        """Create and return a driver instance for Google Gemini.

        Returns:
            An instance of a BaseDriver configured for Google Gemini
        """
        try:
            # Use the parent's _initialize_driver helper method
            return self._initialize_driver()
        except Exception as e:
            logger.error(
                "Failed to create Google Gemini driver", error=str(e), exc_info=True
            )
            raise
