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
)
from .provider_manager import (
    ProviderManager,
    ProviderSource,
    list_providers,
    register_provider,
)
from .router import ProviderRouter

__all__ = [
    "BaseProvider",
    "ProviderAuthError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderLLMRequest",
    "ProviderLLMResponse",
    "ProviderLLMResponseDelta",
    "ProviderManager",
    "ProviderNotFoundError",
    "ProviderRateLimitError",
    "ProviderRouter",
    "ProviderSource",
    "ProviderTimeoutError",
    "list_providers",
    "register_provider",
]
