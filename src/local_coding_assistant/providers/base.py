"""
Base classes for LLM providers
"""

import abc
import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from local_coding_assistant.config.schemas import ModelConfig
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("providers.base")


class OptionalParameters(BaseModel):
    """Optional parameters that need to be validated against model support."""

    stream: bool = False
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    max_tokens: int | None = Field(None, gt=0)
    top_p: float | None = Field(None, gt=0, le=1)
    presence_penalty: float | None = Field(None, ge=-2, le=2)
    frequency_penalty: float | None = Field(None, ge=-2, le=2)


class ProviderLLMRequest(BaseModel):
    """Standardized request format for all providers with validation."""

    messages: list[dict[str, Any]] = Field(
        description="List of message objects with role and content"
    )
    model: str = Field(
        description="Model identifier to use for generation", min_length=1
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for response generation",
    )
    tool_outputs: dict[str, Any] | None = Field(
        None, description="Tool outputs for response generation"
    )
    parameters: OptionalParameters = Field(default_factory=OptionalParameters)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        """Ensure messages are properly formatted"""
        if not v:
            raise ValueError("messages cannot be empty")
        for msg in v:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
        return v

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Flatten the structure for API compatibility."""
        data = super().model_dump(**kwargs)
        # Merge parameters into the root level
        params = data.pop("parameters", {})
        for key, value in params.items():
            if value is not None and value != ([] if key == "tools" else False):
                data[key] = value
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any) -> Any:
        """Handle both flattened and nested parameter structures.

        This validator runs before model validation and ensures that all optional parameters
        are properly nested under the 'parameters' field, regardless of whether they were
        provided at the root level or already nested.
        """
        if isinstance(data, dict):
            data = data.copy()
            param_fields = set(OptionalParameters.model_fields.keys())
            params = {}

            # Extract parameters from root level
            for field in param_fields:
                if field in data:
                    params[field] = data.pop(field)

            # If parameters are nested, merge them
            if "parameters" in data and isinstance(data["parameters"], dict):
                params.update(data.pop("parameters"))

            # Only add parameters if we found any
            if params:
                data["parameters"] = params

        return data

    def validate_against_model(self, supported_parameters: list[str]) -> None:
        """Validate that all used parameters are supported by the model."""
        if not supported_parameters:
            return

        # Check each optional parameter that has a non-default value
        for field in self.parameters.model_fields_set:
            if field not in supported_parameters:
                raise ValueError(
                    f"Parameter '{field}' is not supported by model '{self.model}'. "
                    f"Supported parameters: {', '.join(supported_parameters)}"
                )


class ProviderLLMResponse(BaseModel):
    """Standardized response format from all providers with validation"""

    content: str = Field(description="Generated text content")
    model: str = Field(description="Model that generated the response", min_length=1)
    tokens_used: int | None = Field(
        default=None, ge=0, description="Number of tokens used in the request"
    )
    finish_reason: str | None = Field(
        default=None, description="Reason why generation finished"
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool calls made by the model"
    )
    usage: dict[str, Any] | None = Field(
        default=None, description="Detailed usage statistics"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata about the response"
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        """Ensure content is a string"""
        if not isinstance(v, str):
            raise ValueError("content must be a string")
        return v


class ProviderLLMResponseDelta(BaseModel):
    """Standardized streaming response delta format"""

    content: str = Field(
        default="", description="Content chunk from streaming response"
    )
    role: str | None = Field(
        default=None, description="Role of the message (for streaming)"
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool calls in this delta"
    )
    finish_reason: str | None = Field(
        default=None, description="Finish reason for this chunk"
    )
    metadata: dict[str, Any] | None = Field(
        default={}, description="Additional metadata for this chunk"
    )


class BaseDriver(abc.ABC):
    """Base class for LLM drivers that handle the actual API calls"""

    def __init__(self, api_key: str | None, base_url: str, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs

    @abc.abstractmethod
    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate a response from the LLM"""
        pass

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Generate a streaming response from the LLM"""
        # Default implementation - should be overridden by subclasses
        response = await self.generate(request)
        # Convert response to single delta (for non-streaming providers)
        yield ProviderLLMResponseDelta(
            content=response.content,
            finish_reason=response.finish_reason,
            metadata=response.metadata,
        )

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible"""
        pass

    async def generate_with_retry(
        self,
        request: ProviderLLMRequest,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> ProviderLLMResponse:
        """Generate with retry logic"""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await self.generate(request)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(
                        retry_delay * (2**attempt)
                    )  # Exponential backoff

        raise last_error or RuntimeError(f"Failed after {max_retries} attempts")

    async def stream_with_retry(
        self,
        request: ProviderLLMRequest,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Stream with retry logic"""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                async for delta in self.stream(request):
                    yield delta
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(
                        retry_delay * (2**attempt)
                    )  # Exponential backoff

        raise last_error or RuntimeError(f"Failed after {max_retries} attempts")


class BaseProvider(abc.ABC):
    """Base class for LLM providers

    Subclasses must implement the _create_driver_instance method to provide
    a driver instance for the specific provider implementation.
    """

    @abc.abstractmethod
    def _create_driver_instance(self) -> BaseDriver:
        """Create and return a driver instance for this provider.

        This is a required method that must be implemented by all provider subclasses.
        It should return an instance of a class that inherits from BaseDriver.

        Returns:
            An instance of a BaseDriver subclass specific to this provider
        """
        raise NotImplementedError("Subclasses must implement _create_driver_instance")

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        models: list[ModelConfig] | list[str] | dict[str, dict] | None = None,
        driver: str = "openai_chat",
        health_check_endpoint: str | None = None,
        health_check_method: str = "GET",
        health_check_timeout: float = 5.0,
        **kwargs,
    ):
        self.name = name
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.driver = driver
        self.health_check_endpoint = health_check_endpoint
        self.health_check_method = health_check_method.upper()
        self.health_check_timeout = health_check_timeout
        self.kwargs = kwargs

        # Convert models to list of ModelConfig
        self.models: list[ModelConfig] = self._normalize_models(models)

        # Resolve API key
        self.api_key = api_key or self._get_api_key()

        if not self.api_key:
            from local_coding_assistant.providers.exceptions import ProviderError

            raise ProviderError(f"No API key provided for provider {name}")

        # Initialize driver instance immediately
        self.driver_instance = self._create_driver_instance()

    def _normalize_models(
        self, models: list[ModelConfig] | list[str] | dict[str, dict] | None
    ) -> list[ModelConfig]:
        """Convert various model input formats to a list of ModelConfig objects.

        Args:
            models: Can be:
                - List[ModelConfig]: List of already initialized ModelConfig objects
                - List[str]: List of model names (with default config)
                - Dict[str, dict]: Dict mapping model names to their configs
                - None: Empty list will be returned

        Returns:
            List of ModelConfig objects

        Raises:
            ValueError: If models list contains mixed types or invalid configurations
            TypeError: If models is not a list, dict, or None
        """
        if models is None:
            return []

        if isinstance(models, list):
            if not models:
                return []

            # Handle case where all items are already ModelConfig objects
            if all(isinstance(m, ModelConfig) for m in models):
                return models  # type: ignore

            # Handle case where all items are strings (model names)
            if all(isinstance(m, str) for m in models):
                return [ModelConfig(name=m) for m in models]  # type: ignore

            # Handle case where items are mixed or invalid types
            if any(not isinstance(m, ModelConfig | str) for m in models):
                raise ValueError(
                    "Models list must contain only strings or ModelConfig objects"
                )

            # Handle case where we have a mix of strings and ModelConfig objects
            normalized_models = []
            for model in models:
                if isinstance(model, str):
                    normalized_models.append(ModelConfig(name=model))
                else:
                    normalized_models.append(model)
            return normalized_models

        if isinstance(models, dict):
            return [
                ModelConfig(
                    name=name,
                    supported_parameters=config.get("supported_parameters", []),
                )
                for name, config in models.items()
            ]

        raise TypeError(f"Models must be a list or dict, got {type(models).__name__}")

    def _get_api_key(self) -> str | None:
        """Get API key from environment variable if specified"""
        if self.api_key_env:
            import os

            return os.getenv(self.api_key_env)
        return None

    async def _handle_auth_error(self, error: Exception) -> None:
        """Handle authentication errors and try to recover with API key from env var if available"""
        from local_coding_assistant.providers.exceptions import ProviderAuthError

        if not isinstance(error, ProviderAuthError):
            raise error

        # If we have an API key env var, and it's different from the current key, try with that
        if (
            self.api_key_env
            and (env_key := self._get_api_key())
            and env_key != self.api_key
        ):
            logger.info(
                f"Authentication failed, trying with key from {self.api_key_env}"
            )
            self.api_key = env_key
            # Recreate the driver instance with the new API key
            self.driver_instance = self._create_driver_instance()
            return

        # If we get here, we couldn't recover from the auth error
        raise error

    def _initialize_driver(self):
        """
        Initialize driver instance based on driver type.

        This is a helper method that can be used by subclasses to create
        standard driver instances. Subclasses can override this or implement
        _create_driver_instance directly.
        """
        driver_mapping = {
            "openai_chat": "OpenAIChatCompletionsDriver",
            "openai_responses": "OpenAIResponsesDriver",
            "local": "LocalDriver",
        }

        driver_name = driver_mapping.get(self.driver)
        if not driver_name:
            from local_coding_assistant.providers.exceptions import ProviderError

            raise ProviderError(f"Unknown driver type: {self.driver}")

        # Import the driver class locally to avoid circular imports
        from local_coding_assistant.providers.compatible_drivers import (
            LocalDriver,
            OpenAIChatCompletionsDriver,
            OpenAIResponsesDriver,
        )

        driver_classes = {
            "OpenAIChatCompletionsDriver": OpenAIChatCompletionsDriver,
            "OpenAIResponsesDriver": OpenAIResponsesDriver,
            "LocalDriver": LocalDriver,
        }

        driver_class = driver_classes.get(driver_name)
        if not driver_class:
            from local_coding_assistant.providers.exceptions import ProviderError

            raise ProviderError(f"Unknown driver type: {self.driver}")

        # Avoid passing provider_name twice if it's already in kwargs
        driver_kwargs = self.kwargs.copy()
        if "provider_name" not in driver_kwargs:
            driver_kwargs["provider_name"] = self.name

        return driver_class(
            api_key=self.api_key,
            base_url=self.base_url,
            **driver_kwargs,
        )

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate a response using this provider"""
        try:
            return await self.driver_instance.generate(request)
        except Exception as e:
            await self._handle_auth_error(e)
            # If we get here, _handle_auth_error didn't raise, so we can retry
            return await self.driver_instance.generate(request)

    async def generate_with_retry(
        self,
        request: ProviderLLMRequest,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> ProviderLLMResponse:
        """Generate a response with retry logic"""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await self.generate(request)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(
                        retry_delay * (2**attempt)
                    )  # Exponential backoff

        raise last_error or RuntimeError(f"Failed after {max_retries} attempts")

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Generate a streaming response using this provider"""
        try:
            if self.driver_instance:
                async for delta in self.driver_instance.stream(request):
                    yield delta
            else:
                # Fallback to non-streaming response as single delta
                response = await self.generate(request)
                yield ProviderLLMResponseDelta(
                    content=response.content,
                    finish_reason=response.finish_reason,
                    metadata=response.metadata,
                )
        except Exception as e:
            await self._handle_auth_error(e)
            # If we get here, _handle_auth_error didn't raise, so we can retry
            if self.driver_instance:
                async for delta in self.driver_instance.stream(request):
                    yield delta
            else:
                # Fallback to non-streaming response as single delta
                response = await self.generate(request)
                yield ProviderLLMResponseDelta(
                    content=response.content,
                    finish_reason=response.finish_reason,
                    metadata=response.metadata,
                )

    async def stream_with_retry(
        self,
        request: ProviderLLMRequest,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Stream with retry logic"""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                async for delta in self.stream(request):
                    yield delta
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(
                        retry_delay * (2**attempt)
                    )  # Exponential backoff

        raise last_error or RuntimeError(f"Failed after {max_retries} attempts")

    def _get_health_check_url(self) -> str:
        """Get the full URL for the health check endpoint."""
        if not self.health_check_endpoint:
            return self.base_url.rstrip("/")

        if self.health_check_endpoint.startswith(("http://", "https://")):
            return self.health_check_endpoint

        return f"{self.base_url.rstrip('/')}/{self.health_check_endpoint.lstrip('/')}"

    async def _make_health_request(self, url: str, api_key: str) -> bool:
        """Make the actual health check request."""
        import httpx

        from local_coding_assistant.providers.exceptions import (
            ProviderAuthError,
            ProviderError,
        )

        try:
            async with httpx.AsyncClient(timeout=self.health_check_timeout) as client:
                response = await client.request(
                    method=self.health_check_method,
                    url=url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=self.health_check_timeout,
                )
                response.raise_for_status()
                return True

        except httpx.HTTPStatusError as err:
            if err.response.status_code == 401:
                raise ProviderAuthError(
                    f"Authentication failed for {self.name}"
                ) from err
            logger.warning(
                f"Health check failed for {self.name} with status {err.response.status_code}"
            )
            return False

        except Exception as err:
            error_msg = f"Health check error for {self.name}: {err!s}"
            logger.error(error_msg)
            raise ProviderError(error_msg) from err

    async def _try_env_api_key(self, url: str, original_error: Exception) -> bool:
        """Try using API key from environment variable if available."""
        if not (
            self.api_key_env
            and (env_key := os.getenv(self.api_key_env))
            and env_key != self.api_key
        ):
            raise original_error

        logger.info(
            f"Authentication failed with current API key, trying with key from {self.api_key_env}"
        )
        from local_coding_assistant.providers import ProviderAuthError

        try:
            result = await self._make_health_request(url, env_key)
            self._update_api_key(env_key)
            return result
        except ProviderAuthError:
            raise original_error from None

    def _update_api_key(self, new_key: str) -> None:
        """Update the API key and recreate the driver instance."""
        self.api_key = new_key
        self.driver_instance = self._create_driver_instance()
        logger.info(
            f"Successfully updated API key for {self.name} from environment variable"
        )

    async def health_check(self) -> bool | str:
        """Check provider health.

        Returns:
            bool: True if health check passed, False if failed
            str: 'unavailable' if health check is not implemented or not configured

        Raises:
            ProviderError: If there's an error during health check
        """
        if not self.health_check_endpoint:
            return "unavailable"

        from local_coding_assistant.providers.exceptions import ProviderAuthError

        url = self._get_health_check_url()

        try:
            if self.api_key is None:
                raise ProviderAuthError("No API key provided")
            return await self._make_health_request(url, self.api_key)

        except ProviderAuthError as e:
            return await self._try_env_api_key(url, e)

    def get_model_config(self, model_name: str) -> ModelConfig | None:
        """Get configuration for a specific model by name.

        Args:
            model_name: Name of the model to get config for

        Returns:
            ModelConfig if found, None otherwise

        Raises:
            ValueError: If model_name is not a string or is empty
        """
        if not isinstance(model_name, str):
            raise ValueError(
                f"model_name must be a string, got {type(model_name).__name__}"
            )

        if not model_name:
            raise ValueError("model_name cannot be empty")

        return next((model for model in self.models if model.name == model_name), None)

    def supports_model(self, model_name: str) -> bool:
        """Check if this provider supports the given model.

        Args:
            model_name: Name of the model to check

        Returns:
            bool: True if the model is supported, False otherwise

        Raises:
            ValueError: If model_name is not a string or is empty
        """
        if not model_name:
            return False

        return self.get_model_config(model_name) is not None

    def is_parameter_supported(self, model_name: str, param: str) -> bool:
        """Check if a parameter is supported by the specified model.

        Args:
            model_name: Name of the model to check
            param: Name of the parameter to check

        Returns:
            bool: True if the parameter is supported, False otherwise

        Raises:
            ValueError: If model_name or param is not a string or is empty
        """
        if not isinstance(param, str) or not param:
            raise ValueError("param must be a non-empty string")

        model = self.get_model_config(model_name)
        if not model:
            return False

        # If no supported_parameters are defined, assume all parameters are supported
        return model.supported_parameters is None or param in model.supported_parameters

    def get_supported_parameters(self, model_name: str) -> list[str]:
        """Get list of supported parameters for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            List of supported parameter names. Returns an empty list if the model
            is not found or if no specific parameters are defined.

        Raises:
            ValueError: If model_name is not a string or is empty
        """
        model = self.get_model_config(model_name)
        if not model:
            return []

        # Return a copy to prevent modification of the original list
        return list(model.supported_parameters) if model.supported_parameters else []

    def get_available_models(self) -> list[str]:
        """Get list of available model names for this provider.

        Returns:
            List of model names as strings
        """
        return [model.name for model in self.models]

    def validate_request(self, request: ProviderLLMRequest) -> None:
        """Validate that the request is compatible with the provider and model.

        This method validates that:
        1. The model is specified and supported by this provider
        2. All used parameters are supported by the model

        Args:
            request: The request to validate

        Raises:
            ProviderValidationError: If the request is not compatible with the provider or model
        """
        from local_coding_assistant.providers.exceptions import ProviderValidationError

        try:
            if not request.model:
                raise ValueError("Request must specify a model")

            if not self.supports_model(request.model):
                raise ValueError(
                    f"Model '{request.model}' is not supported by provider '{self.name}'. "
                    f"Supported models: {', '.join(self.get_available_models())}"
                )

            if not hasattr(request, "model_dump"):
                return  # Not a Pydantic model, skip validation

            # Get the model config and supported parameters
            model_config = self.get_model_config(request.model)

            # If we have a model config with supported parameters, validate against them
            if model_config and model_config.supported_parameters:
                request.validate_against_model(model_config.supported_parameters)

        except ValueError as e:
            raise ProviderValidationError(
                str(e), provider=self.name, model=getattr(request, "model", None)
            ) from e
