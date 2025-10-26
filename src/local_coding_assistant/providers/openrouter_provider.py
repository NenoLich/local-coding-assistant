"""
OpenRouter provider implementation
"""

from collections.abc import AsyncGenerator

from .base import (
    BaseProvider,
    ProviderLLMRequest,
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
)
from .provider_manager import register_provider


@register_provider("openrouter")
class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter API"""

    def __init__(
        self,
        api_key: str | None = None,
        api_key_env: str | None = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1/responses",
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

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate response using OpenRouter"""
        return await self.driver_instance.generate(request)

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Generate response using OpenRouter with streaming"""
        async for delta in self.driver_instance.stream(request):
            yield delta

    async def health_check(self) -> bool:
        """Check OpenRouter health"""
        return await self.driver_instance.health_check()
