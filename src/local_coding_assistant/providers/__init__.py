"""
LLM Provider System

This module provides a dynamic provider system for LLM integrations.
Supports multiple providers with automatic fallback and retry capabilities.
"""

# Import provider modules to trigger decorator registration
from . import (
    google_provider,  # noqa: F401
    local_provider,  # noqa: F401
    openrouter_provider,  # noqa: F401
)
from .base import (
    BaseProvider,
    OptionalParameters,
    ProviderLLMRequest,
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
)
from .exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderValidationError,
)
from .health import ProviderHealthManager
from .provider_manager import (
    ProviderManager,
    ProviderSource,
    list_providers,
    register_provider,
)
from .resolver import ProviderResolver

__all__ = [
    "BaseProvider",
    "OptionalParameters",
    "ProviderAuthError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderHealthManager",
    "ProviderLLMRequest",
    "ProviderLLMResponse",
    "ProviderLLMResponseDelta",
    "ProviderManager",
    "ProviderNotFoundError",
    "ProviderRateLimitError",
    "ProviderResolver",
    "ProviderSource",
    "ProviderTimeoutError",
    "ProviderValidationError",
    "list_providers",
    "register_provider",
]
