"""
Local provider implementation for offline/local models
"""

from .base import BaseProvider, ProviderLLMRequest, ProviderLLMResponse
from .provider_manager import register_provider


@register_provider("local")
class LocalProvider(BaseProvider):
    """Provider for local/offline models"""

    def __init__(
        self,
        api_key: str | None = None,  # Not needed for local
        api_key_env: str | None = None,  # Not needed for local
        base_url: str = "http://localhost:11434",  # Default Ollama URL
        models: list | None = None,
        driver: str = "local",  # Use local driver
        **kwargs,
    ):
        # Default models if not specified
        default_models = (
            ["llama3-8b", "llama3-70b", "codellama", "mistral"]
            if models is None
            else models
        )

        # For local providers, use a dummy API key since they're typically free
        # and don't require authentication
        api_key = api_key or "dummy_key_for_local"

        super().__init__(
            name="local",
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            models=default_models,
            driver=driver,
            **kwargs,
        )

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate response using local model"""
        return await self.driver_instance.generate(request)

    async def health_check(self) -> bool:
        """Check local model health"""
        return await self.driver_instance.health_check()
