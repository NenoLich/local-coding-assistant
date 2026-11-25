"""OpenTelemetry setup for runtime tracing.

Configures the global tracer provider once per process so the runtime
``OTelSubscriber`` exports spans over OTLP/gRPC (Jaeger-compatible
collectors). Everything here is synchronous and side-effect free when
telemetry is disabled.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.core.protocols import IConfigManager

logger = get_logger("core.telemetry")

DEFAULT_SERVICE_NAME = "locca-runtime"
_ENDPOINT_ENV_VAR = "OTEL_EXPORTER_OTLP_ENDPOINT"

_configured = False


def _resolve_settings(config_manager: IConfigManager) -> dict[str, Any]:
    """Resolve telemetry settings from config, with env overrides.

    ``OTEL_EXPORTER_OTLP_ENDPOINT`` takes precedence over the config value
    and implicitly enables export.

    Args:
        config_manager: Config manager with loaded global config

    Returns:
        Dict with ``enabled``, ``service_name``, ``otlp_endpoint`` and
        ``console_export`` keys.
    """
    settings: dict[str, Any] = {
        "enabled": False,
        "service_name": DEFAULT_SERVICE_NAME,
        "otlp_endpoint": None,
        "console_export": False,
    }

    try:
        telemetry = config_manager.global_config.telemetry
        settings["enabled"] = bool(telemetry.enabled)
        settings["service_name"] = telemetry.service_name or DEFAULT_SERVICE_NAME
        settings["otlp_endpoint"] = telemetry.otlp_endpoint
        settings["console_export"] = bool(telemetry.console_export)
    except Exception as e:
        logger.debug("Telemetry config unavailable, using defaults", error=str(e))

    endpoint_env = os.getenv(_ENDPOINT_ENV_VAR)
    if endpoint_env:
        settings["otlp_endpoint"] = endpoint_env
        settings["enabled"] = True

    return settings


def is_enabled(config_manager: IConfigManager) -> bool:
    """Check whether telemetry export is enabled for this run."""
    return bool(_resolve_settings(config_manager)["enabled"])


def configure_telemetry(config_manager: IConfigManager) -> bool:
    """Configure the global OTel tracer provider.

    Safe to call multiple times; only the first call takes effect. When
    telemetry is disabled, no provider is created and spans become no-ops
    through the OTel API default provider.

    Args:
        config_manager: Config manager with loaded global config

    Returns:
        True if a tracer provider was configured, False otherwise
    """
    global _configured
    if _configured:
        return True

    settings = _resolve_settings(config_manager)
    if not settings["enabled"]:
        logger.debug("Telemetry disabled, OTel spans will not be exported")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        provider = TracerProvider(
            resource=Resource.create({"service.name": settings["service_name"]})
        )
        trace.set_tracer_provider(provider)

        endpoint = settings["otlp_endpoint"]
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
                )
                logger.info("OTel OTLP exporter configured", endpoint=endpoint)
            except ImportError:
                logger.warning(
                    "OTLP gRPC exporter not installed; spans will not be exported"
                )

        if settings["console_export"]:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logger.info("OTel console span export configured")

        _configured = True
        return True

    except ImportError:
        logger.warning("OpenTelemetry SDK not installed, telemetry disabled")
        return False
    except Exception as e:
        logger.warning("Failed to configure telemetry", error=str(e))
        return False


def get_tracer(name: str = "locca") -> Any:
    """Get a tracer from the (possibly default no-op) global provider."""
    from opentelemetry import trace

    return trace.get_tracer(name)


def shutdown_telemetry() -> None:
    """Flush and shut down the global tracer provider, if configured."""
    global _configured
    if not _configured:
        return

    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
        logger.debug("Telemetry shut down")
    except Exception as e:
        logger.warning("Failed to shut down telemetry", error=str(e))
    finally:
        _configured = False
