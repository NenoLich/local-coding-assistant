"""
OpenRouter provider implementation
"""

from local_coding_assistant.providers.base import (
    BaseProvider,
)
from local_coding_assistant.providers.provider_manager import register_provider


@register_provider("openrouter")
class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter API"""

    def __init__(
        self,
        api_key: str | None = None,
        api_key_env: str | None = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1",
        models: list | None = None,
        driver: str = "openai_responses",  # Use OpenAI responses driver
        **kwargs,
    ):
        # Default models if not specified
        default_models = (
            [
                "qwen/qwen3-coder:free",
                "qwen/qwen3-235b-a22b:free",
                "moonshotai/kimi-dev-72b:free",
            ]
            if models is None
            else models
        )

        super().__init__(
            name="openrouter",
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            models=default_models,
            driver=driver,
            **kwargs,
        )

    def _create_driver_instance(self):
        """Create and return a driver instance for OpenRouter.

        Returns:
            An instance of a BaseDriver configured for OpenRouter
        """
        # Use the parent's _initialize_driver helper method
        return self._initialize_driver()
