"""
Generic Provider that can be configured from a dictionary.
"""

import os
from typing import Any

from local_coding_assistant.providers.base import BaseProvider
from local_coding_assistant.providers.compatible_drivers import (
    OpenAIChatCompletionsDriver,
    OpenAIResponsesDriver,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("providers.generic")

driver_map = {
    "openai_chat": OpenAIChatCompletionsDriver,
    "openai_responses": OpenAIResponsesDriver,
}


class GenericProvider(BaseProvider):
    """A generic provider that is configured from a dictionary."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the generic provider.

        Args:
            **kwargs: Configuration parameters including:
                - name: Name of the provider (required)
                - base_url: Base URL for the provider's API (required)
                - driver: Driver to use (must be one of the supported drivers)
                - models: Optional list of model configurations
                - api_key: Optional API key
                - api_key_env: Optional environment variable name for API key
        """
        # Extract required parameters
        name = kwargs.get("name")
        base_url = kwargs.get("base_url")

        if not name:
            raise ValueError("name is required for GenericProvider")
        if not base_url:
            raise ValueError(f"base_url is required for GenericProvider - {name}")

        # Extract models if provided, otherwise use empty list
        models = kwargs.pop("models", [])

        # Initialize the base provider
        super().__init__(
            name=name,
            base_url=base_url,
            models=models,  # Pass models to BaseProvider
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["name", "base_url", "models"]
            },
        )

        # Initialize the driver
        driver_name = kwargs.get("driver")
        if not driver_name or driver_name not in driver_map:
            raise ValueError(
                f"Invalid or missing driver for provider {self.name}: {driver_name}"
            )

        try:
            driver_class = driver_map[driver_name]
            api_key = kwargs.get("api_key")
            if not api_key and self.api_key_env:
                api_key = os.getenv(self.api_key_env)

            driver_kwargs = {"provider_name": self.name}
            self.driver_instance = driver_class(
                api_key=api_key, base_url=self.base_url, **driver_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to initialize driver for {self.name}: {e!s}")
            raise

    def _create_driver_instance(self) -> Any:
        """Create and return a driver instance for the provider.

        Returns:
            An instance of a BaseDriver configured for the provider

        Raises:
            RuntimeError: If the driver cannot be initialized
        """
        try:
            # Use the parent's _initialize_driver helper method
            return self._initialize_driver()
        except Exception as e:
            logger.error(f"Failed to create driver instance for {self.name}: {e!s}")
            raise RuntimeError(f"Failed to initialize driver: {e!s}") from e
