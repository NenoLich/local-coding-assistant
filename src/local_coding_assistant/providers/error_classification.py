"""Structured error classification for provider/LLM failures.

Single source of truth for deciding whether an error is retryable
(transient) or permanent (non-transient).

Classification precedence (strict, global across the cause chain):
1. Typed exceptions (``ProviderError`` subclasses, ``LLMContentError``)
2. Duck-typed HTTP status codes (SDK errors expose ``status_code`` or
   ``response.status_code``)
3. The ``__cause__``/``__context__`` chain (failover wrappers hide the
   original typed error there)
4. Tightened message regexes (word boundaries, explicit phrases only)
5. Default: permanent (fail safe)

Tool errors carry structured ``retryable`` flags on the contract /
``ToolPlatformError`` and are classified by ``classify_tool_error``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from local_coding_assistant.core.exceptions import LLMContentError
from local_coding_assistant.providers.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderValidationError,
)

# HTTP status codes considered retryable by provider semantics.
TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
PERMANENT_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 413, 422})

# Message patterns used only when no typed/structured signal exists.
# Codes are matched with word boundaries so "429" does not match "14.29s".
_TRANSIENT_CODE_RE = re.compile(r"\b(408|409|425|429|500|502|503|504|529)\b")
_TRANSIENT_PHRASE_RE = re.compile(
    r"rate limit|too many requests|timed out|timeout|"
    r"connection (?:refused|reset|error|aborted)|"
    r"temporarily unavailable|service unavailable|"
    r"internal server error|overloaded",
    re.IGNORECASE,
)
_PERMANENT_CODE_RE = re.compile(r"\b(400|401|402|403|404|413|422)\b")
_PERMANENT_PHRASE_RE = re.compile(
    r"authentication failed|unauthorized|invalid api key|"
    r"access denied|permission denied|bad request|invalid request|"
    r"validation failed|not found|quota exceeded|billing|"
    r"context length exceeded",
    re.IGNORECASE,
)

_MAX_CAUSE_DEPTH = 10


def _next_cause(exc: BaseException) -> BaseException | None:
    """Next link in the cause chain, or None.

    Only follows actual exceptions: SDK response objects and other
    non-exception attributes must never be treated as chain links.
    """
    for candidate in (exc.__cause__, exc.__context__):
        if isinstance(candidate, BaseException):
            return candidate
    return None


def _walk_cause_chain(exc: BaseException | None):
    """Yield the exception and its causes, cycle- and depth-safe."""
    seen: set[int] = set()
    current: BaseException | None = exc if isinstance(exc, BaseException) else None
    depth = 0
    while current is not None and id(current) not in seen and depth < _MAX_CAUSE_DEPTH:
        seen.add(id(current))
        depth += 1
        yield current
        current = _next_cause(current)


class ProviderErrorKind(str, Enum):
    """Coarse category of a provider/LLM failure (descriptive only)."""

    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTH = "auth"
    VALIDATION = "validation"
    BILLING = "billing"
    CONTENT = "content"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassifiedError:
    """Structured classification of a provider/LLM error.

    ``transient`` is the authoritative retry signal; ``kind`` is
    descriptive metadata for logging and telemetry.
    """

    kind: ProviderErrorKind
    transient: bool
    status_code: int | None = None
    retry_after: float | None = None
    provider: str | None = None
    decided_by: str = "default"
    original: BaseException | None = None


def extract_status_code(exc: BaseException | None) -> int | None:
    """Extract an HTTP status code from a duck-typed SDK exception.

    Handles the two common SDK shapes:
    - ``exc.status_code`` (openai, anthropic, httpx)
    - ``exc.response.status_code`` (requests, older openai)

    The cause chain is followed, so wrapper exceptions are transparent.
    """
    for current in _walk_cause_chain(exc):
        status = getattr(current, "status_code", None)
        if status is not None:
            try:
                return int(str(status))
            except (ValueError, TypeError):
                pass
        response = getattr(current, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            if status is not None:
                try:
                    return int(str(status))
                except (ValueError, TypeError):
                    pass
    return None


def retry_after_from_response(response: object) -> float | None:
    """Extract a Retry-After hint (seconds) from an HTTP response object."""
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    try:
        return _parse_retry_after(getter("retry-after"))
    except Exception:
        return None


def extract_retry_after(exc: BaseException | None) -> float | None:
    """Extract a Retry-After hint (seconds) from a duck-typed SDK exception."""
    for current in _walk_cause_chain(exc):
        candidates = [
            getattr(current, "retry_after", None),
            getattr(current, "retry_after_seconds", None),
        ]
        response = getattr(current, "response", None)
        if response is not None:
            candidates.append(retry_after_from_response(response))
        for candidate in candidates:
            seconds = _parse_retry_after(candidate)
            if seconds is not None:
                return seconds
    return None


def _parse_retry_after(value: Any) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(str(value).strip())
    except (ValueError, TypeError):
        return None
    if seconds < 0:
        return None
    return seconds


def _kind_for_status_code(status_code: int) -> ProviderErrorKind:
    if status_code == 429:
        return ProviderErrorKind.RATE_LIMIT
    if status_code == 401 or status_code == 403:
        return ProviderErrorKind.AUTH
    if status_code == 402:
        return ProviderErrorKind.BILLING
    if status_code in {400, 404, 413, 422}:
        return ProviderErrorKind.VALIDATION
    if status_code in {500, 502, 503, 504, 529}:
        return ProviderErrorKind.SERVER
    if status_code in {408, 409, 425}:
        return ProviderErrorKind.NETWORK
    return ProviderErrorKind.UNKNOWN


def _classified(
    kind: ProviderErrorKind,
    transient: bool,
    exc: BaseException,
    decided_by: str,
) -> ClassifiedError:
    return ClassifiedError(
        kind=kind,
        transient=transient,
        status_code=extract_status_code(exc),
        retry_after=extract_retry_after(exc),
        provider=getattr(exc, "provider", None),
        decided_by=decided_by,
        original=exc,
    )


def _classify_typed(exc: BaseException) -> ClassifiedError | None:
    """Classify by exception type hierarchy (authoritative when available)."""
    if isinstance(exc, ProviderRateLimitError):
        return ClassifiedError(
            kind=ProviderErrorKind.RATE_LIMIT,
            transient=True,
            status_code=429,
            retry_after=getattr(exc, "retry_after", None),
            provider=getattr(exc, "provider", None),
            decided_by="typed",
            original=exc,
        )
    if isinstance(exc, ProviderAuthError):
        return ClassifiedError(
            kind=ProviderErrorKind.AUTH,
            transient=False,
            status_code=401,
            provider=getattr(exc, "provider", None),
            decided_by="typed",
            original=exc,
        )
    if isinstance(exc, ProviderTimeoutError):
        return _classified(ProviderErrorKind.TIMEOUT, True, exc, "typed")
    if isinstance(exc, ProviderConnectionError):
        # Connection errors cover server-side 5xx raised by drivers
        status_code = extract_status_code(exc)
        if status_code is not None and 500 <= status_code < 600:
            return _classified(ProviderErrorKind.SERVER, True, exc, "typed")
        return _classified(ProviderErrorKind.NETWORK, True, exc, "typed")
    if isinstance(exc, ProviderValidationError):
        return _classified(ProviderErrorKind.VALIDATION, False, exc, "typed")
    if isinstance(exc, LLMContentError):
        return _classified(ProviderErrorKind.CONTENT, False, exc, "typed")
    # Bare ProviderError / other ProviderError subclasses fall through to
    # the structured signals below.
    return None


def _classify_by_status_code(exc: BaseException) -> ClassifiedError | None:
    status_code = extract_status_code(exc)
    if status_code is None:
        return None
    if status_code in TRANSIENT_STATUS_CODES:
        return _classified(_kind_for_status_code(status_code), True, exc, "status_code")
    if status_code in PERMANENT_STATUS_CODES:
        return _classified(
            _kind_for_status_code(status_code), False, exc, "status_code"
        )
    # Unmapped status (e.g. 418): no signal, fall through.
    return None


def _classify_by_message(exc: BaseException) -> ClassifiedError | None:
    message = str(exc)
    type_name = type(exc).__name__.lower()

    # Explicit permanent signals win: retrying them cannot succeed.
    if _PERMANENT_CODE_RE.search(message) or _PERMANENT_PHRASE_RE.search(message):
        return _classified(ProviderErrorKind.UNKNOWN, False, exc, "message")
    if _TRANSIENT_CODE_RE.search(message) or _TRANSIENT_PHRASE_RE.search(message):
        return _classified(ProviderErrorKind.UNKNOWN, True, exc, "message")
    # Well-known exception class names (whole name, not substring)
    if any(name in type_name for name in ("timeouterror", "ratelimiterror")):
        return _classified(ProviderErrorKind.TIMEOUT, True, exc, "message")
    if "connectionerror" in type_name or "networkerror" in type_name:
        return _classified(ProviderErrorKind.NETWORK, True, exc, "message")
    if "autherror" in type_name or "permissionerror" in type_name:
        return _classified(ProviderErrorKind.AUTH, False, exc, "message")
    return None


def _classify_chain(exc: BaseException) -> ClassifiedError | None:
    """Strict precedence across the whole cause chain.

    All links are checked with the strongest signal (typed exceptions)
    before weaker signals (status codes, messages) are considered, so a
    wrapper's generic message can never mask a typed cause.
    """
    for classifier in (_classify_typed, _classify_by_status_code):
        for link in _walk_cause_chain(exc):
            result = classifier(link)
            if result is not None:
                return result
    for link in _walk_cause_chain(exc):
        result = _classify_by_message(link)
        if result is not None:
            return result
    return None


def classify_llm_error(exc: Exception) -> ClassifiedError:
    """Classify an exception raised anywhere in the LLM/provider path.

    Args:
        exc: The exception to classify. The cause chain is followed, so
            wrapper exceptions (e.g. failover ``LLMError``) are fine.

    Returns:
        ClassifiedError. ``transient`` is only True for failures that are
        safe to retry; unknown errors default to permanent.
    """
    if not isinstance(exc, BaseException):
        return ClassifiedError(
            kind=ProviderErrorKind.UNKNOWN, transient=False, decided_by="invalid"
        )

    classified = _classify_chain(exc)
    if classified is not None:
        return classified

    return ClassifiedError(
        kind=ProviderErrorKind.UNKNOWN,
        transient=False,
        status_code=extract_status_code(exc),
        provider=getattr(exc, "provider", None),
        decided_by="default",
        original=exc,
    )


__all__ = [
    "PERMANENT_STATUS_CODES",
    "TRANSIENT_STATUS_CODES",
    "ClassifiedError",
    "ProviderErrorKind",
    "classify_llm_error",
    "extract_retry_after",
    "extract_status_code",
]
