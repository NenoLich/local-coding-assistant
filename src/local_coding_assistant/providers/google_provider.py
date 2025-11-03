"""
Google Gemini provider implementation
"""

import local_coding_assistant.providers.base as base
from local_coding_assistant.providers.provider_manager import register_provider


@register_provider("google_gemini")
class GoogleGeminiProvider(base.BaseProvider):
    """Provider for Google Gemini API"""

    def __init__(
        self,
        api_key: str | None = None,
        api_key_env: str | None = "GEMINI_API_KEY",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        models: list | None = None,
        driver: str = "openai_chat",  # Use OpenAI chat driver
        **kwargs,
    ):
        # Default models if not specified
        default_models = ["gemini-2.5-flash"] if models is None else models

        super().__init__(
            name="google_gemini",
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
        # Use the parent's _initialize_driver helper method
        return self._initialize_driver()
