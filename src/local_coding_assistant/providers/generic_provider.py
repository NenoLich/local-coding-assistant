"""
Generic Provider that can be configured from a dictionary.
"""

import os

from local_coding_assistant.providers.base import (
    BaseProvider,
)
from local_coding_assistant.providers.compatible_drivers import (
    OpenAIChatCompletionsDriver,
    OpenAIResponsesDriver,
)

driver_map = {
    "openai_chat": OpenAIChatCompletionsDriver,
    "openai_responses": OpenAIResponsesDriver,
}


class GenericProvider(BaseProvider):
    """A generic provider that is configured from a dictionary."""

    def __init__(self, **kwargs):
        if "name" not in kwargs:
            raise ValueError("name is required for GenericProvider")
        if "base_url" not in kwargs:
            raise ValueError(
                f"base_url is required for GenericProvider - {kwargs['name']}"
            )
        super().__init__(
            name=kwargs["name"],
            base_url=kwargs["base_url"],
            **{k: v for k, v in kwargs.items() if k not in ["name", "base_url"]},
        )

        driver_name = kwargs.get("driver")
        if not driver_name or driver_name not in driver_map:
            raise ValueError(
                f"Invalid or missing driver for provider {self.name}: {driver_name}"
            )

        driver_class = driver_map[driver_name]
        api_key = kwargs.get("api_key")
        if not api_key and self.api_key_env:
            api_key = os.getenv(self.api_key_env)

        driver_kwargs = {"provider_name": self.name}
        self.driver_instance = driver_class(
            api_key=api_key, base_url=self.base_url, **driver_kwargs
        )

    def _create_driver_instance(self):
        """Create and return a driver instance for local models.

        Returns:
            An instance of a BaseDriver configured for local models
        """
        # Use the parent's _initialize_driver helper method
        return self._initialize_driver()
