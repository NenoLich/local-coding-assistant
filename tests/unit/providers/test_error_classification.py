"""Unit tests for structured provider error classification.

Covers the classification precedence (typed exceptions -> status codes ->
cause chain -> message patterns -> permanent default) and the traps that
made the previous substring-based classifier unreliable.
"""

from __future__ import annotations

import pytest

from local_coding_assistant.core.exceptions import (
    LLMContentError,
    LLMError,
    LLMInfrastructureError,
)
from local_coding_assistant.providers.error_classification import (
    PERMANENT_STATUS_CODES,
    TRANSIENT_STATUS_CODES,
    ProviderErrorKind,
    classify_llm_error,
    extract_retry_after,
    extract_status_code,
)
from local_coding_assistant.providers.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderValidationError,
)


class DummyResponse:
    """Minimal requests-like response object."""

    def __init__(self, status_code: int, headers: dict | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.text = '{"error": "boom"}'


class TestTypedClassification:
    """Typed provider exceptions are the authoritative signal."""

    @pytest.mark.parametrize(
        ("error", "expected_kind", "expected_transient"),
        [
            (
                ProviderRateLimitError("Rate limit exceeded", provider="p"),
                ProviderErrorKind.RATE_LIMIT,
                True,
            ),
            (
                ProviderTimeoutError("timed out", provider="p"),
                ProviderErrorKind.TIMEOUT,
                True,
            ),
            (
                ProviderConnectionError("connection reset", provider="p"),
                ProviderErrorKind.NETWORK,
                True,
            ),
            (
                ProviderAuthError("Invalid API key", provider="p"),
                ProviderErrorKind.AUTH,
                False,
            ),
            (
                ProviderValidationError("Invalid request", provider="p"),
                ProviderErrorKind.VALIDATION,
                False,
            ),
            (LLMContentError("malformed json"), ProviderErrorKind.CONTENT, False),
        ],
    )
    def test_typed_errors_are_authoritative(
        self,
        error: Exception,
        expected_kind: ProviderErrorKind,
        expected_transient: bool,
    ) -> None:
        result = classify_llm_error(error)

        assert result.kind is expected_kind
        assert result.transient is expected_transient
        assert result.decided_by == "typed"
        if expected_kind is not ProviderErrorKind.CONTENT:
            assert result.provider == "p"

    def test_rate_limit_carries_retry_after(self) -> None:
        result = classify_llm_error(
            ProviderRateLimitError("slow down", provider="p", retry_after=30)
        )

        assert result.transient is True
        assert result.status_code == 429
        assert result.retry_after == 30.0

    def test_connection_error_with_5xx_is_server_kind(self) -> None:
        result = classify_llm_error(
            ProviderConnectionError(
                "Server error (HTTP 503)", provider="p", status_code=503
            )
        )

        assert result.kind is ProviderErrorKind.SERVER
        assert result.transient is True
        assert result.status_code == 503

    def test_bare_provider_error_falls_through_to_message(self) -> None:
        result = classify_llm_error(
            ProviderError("API error: HTTP 418 error: teapot", provider="p")
        )

        # 418 is not in either map -> no structured signal -> permanent
        assert result.transient is False
        assert result.decided_by == "default"


class TestCauseChain:
    """Wrapper exceptions must not hide the typed original error."""

    def test_infrastructure_wrapper_preserves_classification(self) -> None:
        cause = ProviderRateLimitError(
            "Rate limit exceeded", provider="p", retry_after=12
        )
        wrapper = LLMInfrastructureError(
            "LLM streaming failed after exhausting failover attempts",
            classification=classify_llm_error(cause),
        )
        try:
            raise wrapper from cause
        except LLMInfrastructureError as e:
            result = classify_llm_error(e)

        assert result.kind is ProviderErrorKind.RATE_LIMIT
        assert result.transient is True
        assert result.retry_after == 12.0
        assert result.decided_by == "typed"

    def test_generic_wrapper_without_classification_attr(self) -> None:
        """The pre-refactor service.py shape: generic message, typed cause."""
        cause = ProviderConnectionError(
            "Server error (HTTP 503)", provider="p", status_code=503
        )
        try:
            raise LLMError(
                "LLM streaming failed after exhausting failover attempts"
            ) from cause
        except LLMError as e:
            result = classify_llm_error(e)

        assert result.transient is True
        assert result.kind is ProviderErrorKind.SERVER

    def test_wrapper_message_does_not_mask_typed_cause(self) -> None:
        """A permanent phrase in a wrapper message must not win over a typed cause."""
        cause = ProviderConnectionError("Server error (HTTP 503)", provider="p")
        try:
            raise LLMError("unauthorized to retry: not found") from cause
        except LLMError as e:
            result = classify_llm_error(e)

        assert result.transient is True
        assert result.decided_by == "typed"

    def test_cause_cycle_is_safe(self) -> None:
        first = LLMError("first")
        second = LLMError("second")
        try:
            raise first from second
        except LLMError:
            pass
        try:
            raise second from first
        except LLMError:
            pass

        result = classify_llm_error(first)
        assert result.transient is False


class TestStatusCodeExtraction:
    """Duck-typed SDK exceptions expose status codes."""

    def test_status_code_attribute(self) -> None:
        class SDKError(Exception):
            status_code = 429

        result = classify_llm_error(SDKError("rate limited"))

        assert result.kind is ProviderErrorKind.RATE_LIMIT
        assert result.transient is True
        assert result.decided_by == "status_code"

    def test_status_code_on_response(self) -> None:
        class SDKError(Exception):
            response = DummyResponse(status_code=500)

        result = classify_llm_error(SDKError("internal error"))

        assert result.kind is ProviderErrorKind.SERVER
        assert result.transient is True

    def test_status_code_extraction_helper(self) -> None:
        class SDKError(Exception):
            response = DummyResponse(status_code=504)

        assert extract_status_code(SDKError("x")) == 504
        assert extract_status_code(ValueError("no code")) is None
        assert extract_status_code(None) is None

    def test_retry_after_from_response_headers(self) -> None:
        class SDKError(Exception):
            response = DummyResponse(status_code=429, headers={"retry-after": "45"})

        assert extract_retry_after(SDKError("x")) == 45.0

    def test_retry_after_from_exception_attribute(self) -> None:
        class SDKError(Exception):
            retry_after = 7

        assert extract_retry_after(SDKError("x")) == 7.0

    def test_unmapped_status_is_not_a_signal(self) -> None:
        class SDKError(Exception):
            status_code = 418

        result = classify_llm_error(SDKError("teapot"))

        assert result.transient is False
        assert result.decided_by == "default"


class TestMessagePatterns:
    """Message regexes are the last resort and must be tight."""

    @pytest.mark.parametrize(
        ("message", "expected_transient"),
        [
            ("Request timed out after 30s", True),
            ("connection reset by peer", True),
            ("service temporarily unavailable", True),
            ("internal server error", True),
            ("model overloaded", True),
            ("Request took 14.29s for completion", False),
            ("cannot open /cache/504/data.bin: permission denied", False),
            ("resource not found in namespace cache", False),
            ("quota exceeded for organization", False),
            ("something odd happened", False),
        ],
    )
    def test_message_traps(self, message: str, expected_transient: bool) -> None:
        result = classify_llm_error(OSError(message))

        assert result.transient is expected_transient
        assert result.decided_by in {"message", "default"}

    def test_timeout_exception_type_name(self) -> None:
        class ModelTimeoutError(Exception):
            pass

        result = classify_llm_error(ModelTimeoutError("gave up"))

        assert result.transient is True

    def test_permission_error_type_name_is_permanent(self) -> None:
        result = classify_llm_error(PermissionError("denied"))

        assert result.transient is False

    def test_transient_code_matches_only_whole_numbers(self) -> None:
        """'429' inside a latency value must not trigger a rate-limit match."""
        result = classify_llm_error(ValueError("request took 14.29s"))

        assert result.transient is False


class TestDefaults:
    def test_unknown_error_is_permanent(self) -> None:
        result = classify_llm_error(Exception("totally unknown failure"))

        assert result.transient is False
        assert result.kind is ProviderErrorKind.UNKNOWN
        assert result.decided_by == "default"

    def test_status_maps_cover_common_codes(self) -> None:
        assert {408, 409, 425, 429, 500, 502, 503, 504, 529} == set(
            TRANSIENT_STATUS_CODES
        )
        assert {400, 401, 402, 403, 404, 413, 422} == set(PERMANENT_STATUS_CODES)
