"""Telemetry helpers for the LLM service pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from local_coding_assistant.utils.logging import get_logger


@dataclass(slots=True)
class TelemetryEmitter:
    """Structured logging hooks for generation attempts and failovers."""

    logger_name: str = "agent.llm.telemetry"
    _logger: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = get_logger(self.logger_name)

    def attempt_start(
        self,
        *,
        mode: str,
        attempt: int,
        provider: str,
        model: str,
        policy: str,
        task_metadata: dict[str, Any],
        options_metadata: dict[str, Any],
    ) -> None:
        self._logger.info(
            "llm_attempt_start",
            mode=mode,
            attempt=attempt,
            provider=provider,
            model=model,
            policy=policy,
            task_metadata=task_metadata,
            options_metadata=options_metadata,
        )

    def attempt_success(
        self,
        *,
        mode: str,
        attempt: int,
        provider: str,
        model: str,
        tokens_used: int | None,
    ) -> None:
        self._logger.info(
            "llm_attempt_success",
            mode=mode,
            attempt=attempt,
            provider=provider,
            model=model,
            tokens_used=tokens_used,
        )

    def attempt_failure(
        self,
        *,
        mode: str,
        attempt: int,
        provider: str,
        model: str,
        error: Exception,
    ) -> None:
        self._logger.warning(
            "llm_attempt_failure",
            mode=mode,
            attempt=attempt,
            provider=provider,
            model=model,
            error=str(error),
        )

    def stream_chunk(
        self,
        *,
        provider: str,
        model: str,
        is_final: bool,
    ) -> None:
        self._logger.debug(
            "llm_stream_chunk",
            provider=provider,
            model=model,
            is_final=is_final,
        )

    def failover_exhausted(
        self,
        *,
        mode: str,
        attempts: int,
        error: Exception | None,
    ) -> None:
        self._logger.error(
            "llm_failover_exhausted",
            mode=mode,
            attempts=attempts,
            error=str(error) if error else None,
        )
