"""
Google Gemini provider implementation
"""

from collections.abc import AsyncGenerator

from .base import (
    BaseProvider,
    ProviderLLMRequest,
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
)
from .provider_manager import register_provider


@register_provider("google_gemini")
class GoogleGeminiProvider(BaseProvider):
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

    async def health_check(self) -> bool:
        """Check Google Gemini health"""
        return await self.driver_instance.health_check()

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate response using Google Gemini"""
        return await self.driver_instance.generate(request)

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Generate response using Google Gemini with streaming"""
        async for delta in self.driver_instance.stream(request):
            yield delta
