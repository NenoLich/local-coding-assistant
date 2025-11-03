"""
Local provider implementation for offline/local models
"""

from local_coding_assistant.providers.base import BaseProvider
from local_coding_assistant.providers.provider_manager import register_provider


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

        if kwargs["name"]:
            name = kwargs.pop("name")
        else:
            name = "local"

        super().__init__(
            name=name,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            models=default_models,
            driver=driver,
            **kwargs,
        )

    def _create_driver_instance(self):
        """Create and return a driver instance for local models.

        Returns:
            An instance of a BaseDriver configured for local models
        """
        # Use the parent's _initialize_driver helper method
        return self._initialize_driver()

    async def health_check(self) -> bool:
        """Check local model health"""
        return await self.driver_instance.health_check()
