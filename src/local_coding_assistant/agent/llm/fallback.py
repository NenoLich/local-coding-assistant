"""Fallback strategy abstractions for LLM provider orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import LLMPolicy


class FallbackStrategy(Protocol):
    """Defines how the service should behave after a failed attempt."""

    def next_delay(self, attempt: int, *, base_delay: float) -> float:
        """Return delay (seconds) before the next attempt."""

    def should_continue(self, attempt: int, *, max_attempts: int) -> bool:
        """Return True if another attempt should be made."""


@dataclass(slots=True)
class SequentialFallback(FallbackStrategy):
    """Basic strategy that retries sequentially across policy routes."""

    max_failovers: int | None = None

    def next_delay(self, attempt: int, *, base_delay: float) -> float:
        return base_delay

    def should_continue(self, attempt: int, *, max_attempts: int) -> bool:
        return attempt < max_attempts


@dataclass(slots=True)
class ExponentialBackoffFallback(FallbackStrategy):
    """Strategy that increases delay exponentially."""

    max_failovers: int | None = None
    multiplier: float = 2.0

    def next_delay(self, attempt: int, *, base_delay: float) -> float:
        return base_delay * (self.multiplier ** (attempt - 1))

    def should_continue(self, attempt: int, *, max_attempts: int) -> bool:
        return attempt < max_attempts


@dataclass(slots=True)
class NoFallback(FallbackStrategy):
    """No fallback across policy routes."""

    def next_delay(self, attempt: int, *, base_delay: float) -> float:
        return 0.0

    def should_continue(self, attempt: int, *, max_attempts: int) -> bool:
        return attempt == 0


def get_strategy(policy: LLMPolicy | None = None) -> FallbackStrategy:
    """Factory selecting the right fallback strategy for a policy."""
    if policy:
        name = policy.fallback_strategy.lower()
        if name == "exponential":
            return ExponentialBackoffFallback(max_failovers=policy.max_failovers)
        if name == "sequential":
            return SequentialFallback(max_failovers=policy.max_failovers)
    return NoFallback()
