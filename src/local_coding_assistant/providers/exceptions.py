"""
Provider-specific exceptions
"""

from local_coding_assistant.core.exceptions import LLMError


class ProviderError(LLMError):
    """Base exception for provider-related errors"""

    def __init__(self, message: str, provider: str | None = None, **kwargs):
        self.provider = provider
        self.message = message
        super().__init__(
            f"[{provider or 'unknown'}] {message}" if provider else message, **kwargs
        )


class ProviderConnectionError(ProviderError):
    """Raised when connection to provider fails"""

    def __init__(self, message: str, provider: str | None = None, **kwargs):
        super().__init__(message, provider, **kwargs)


class ProviderAuthError(ProviderError):
    """Raised when authentication with provider fails"""

    def __init__(self, message: str, provider: str | None = None, **kwargs):
        super().__init__(message, provider, **kwargs)


class ProviderRateLimitError(ProviderError):
    """Raised when rate limit is exceeded"""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        retry_after: int | None = None,
        **kwargs,
    ):
        self.retry_after = retry_after
        super().__init__(message, provider, **kwargs)


class ProviderTimeoutError(ProviderError):
    """Raised when request times out"""

    def __init__(self, message: str, provider: str | None = None, **kwargs):
        super().__init__(message, provider, **kwargs)


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not available"""

    def __init__(self, message: str, provider: str | None = None, **kwargs):
        super().__init__(message, provider, **kwargs)
