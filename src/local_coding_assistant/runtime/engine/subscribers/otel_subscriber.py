"""OTel/Jaeger subscriber for distributed tracing.

Creates one span hierarchy per turn (turn -> frame -> llm/tool) using the
tracer provider configured by ``core.telemetry``. Falls back to a minimal
provider when used stand-alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from local_coding_assistant.runtime.events.contracts import (
    FrameCompleteEvent,
    FrameStartEvent,
    LLMCompleteEvent,
    LLMStartEvent,
    RuntimeEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

logger = get_logger("runtime.subscribers.otel")


class OTelSubscriber:
    """Subscriber that emits OpenTelemetry spans for runtime events."""

    def __init__(
        self,
        service_name: str = "locca-runtime",
        otel_endpoint: str | None = None,
    ) -> None:
        self._service_name = service_name
        # Kept for backward compatibility; export is configured centrally
        # by core.telemetry, so this is only used as a fallback hint.
        self._otel_endpoint = otel_endpoint
        self._tracer: Tracer | None = None
        self._initialized = False

        self._turn_spans: dict[str, Any] = {}
        self._frame_spans: dict[str, Any] = {}
        self._llm_spans: dict[str, Any] = {}
        self._tool_spans: dict[str, Any] = {}

    def _initialize(self) -> None:
        """Resolve the tracer from the globally configured provider."""
        if self._initialized:
            return

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider

            provider = trace.get_tracer_provider()
            if not isinstance(provider, TracerProvider):
                provider = None

            if provider is None:
                from opentelemetry.sdk.resources import Resource

                provider = TracerProvider(
                    resource=Resource.create({"service.name": self._service_name})
                )
                trace.set_tracer_provider(provider)
                logger.debug("OTel subscriber created a standalone tracer provider")

            self._tracer = provider.get_tracer(self._service_name)
            self._initialized = True

        except ImportError:
            logger.warning("OpenTelemetry SDK not installed, OTel subscriber disabled")
            self._initialized = True  # Don't retry

    async def on_event(self, event: RuntimeEvent) -> None:
        """Process event and create/update spans."""
        if not self._initialized:
            self._initialize()

        if not self._tracer:
            return

        try:
            self._dispatch_event(event)
        except Exception as e:
            logger.error("OTel subscriber error", error=str(e), exc_info=True)

    def _dispatch_event(self, event: RuntimeEvent) -> None:
        """Dispatch event to appropriate handler."""
        match event:
            case TurnStartEvent():
                self._handle_turn_start(event)
            case FrameStartEvent():
                self._handle_frame_start(event)
            case LLMStartEvent():
                self._handle_llm_start(event)
            case LLMCompleteEvent():
                self._handle_llm_complete(event)
            case ToolStartEvent():
                self._handle_tool_start(event)
            case ToolResultEvent():
                self._handle_tool_result(event)
            case FrameCompleteEvent():
                self._handle_frame_complete(event)
            case TurnCompleteEvent():
                self._handle_turn_complete(event)

    def _handle_turn_start(self, event: TurnStartEvent) -> None:
        """Start the root turn span."""
        assert self._tracer is not None
        span = self._tracer.start_span(
            f"turn.{event.turn_index}",
            attributes={
                "turn.id": event.turn_id,
                "turn.index": event.turn_index,
                "session.id": event.session_id,
                "trace.id": event.trace_id,
                "user.input.length": len(event.user_input),
            },
        )
        self._turn_spans[event.turn_id] = span

    def _handle_frame_start(self, event: FrameStartEvent) -> None:
        """Start frame span as child of the turn span."""
        assert self._tracer is not None
        parent = self._turn_spans.get(event.turn_id)
        context = self._parent_context(parent)
        span = self._tracer.start_span(
            f"frame.{event.iteration}",
            context=context,
            attributes={
                "frame.id": event.frame_id,
                "frame.iteration": event.iteration,
                "turn.id": event.turn_id,
                "session.id": event.session_id,
                "trace.id": event.trace_id,
                "context.snapshot.id": event.context_snapshot_id,
            },
        )
        self._frame_spans[event.frame_id or ""] = span

    def _handle_llm_start(self, event: LLMStartEvent) -> None:
        """Start LLM span as child of the frame span."""
        assert self._tracer is not None
        parent = self._frame_spans.get(event.frame_id or "")
        span = self._tracer.start_span(
            "llm.request",
            context=self._parent_context(parent),
            attributes={
                "llm.model": event.model or "unknown",
                "frame.id": event.frame_id or "unknown",
                "turn.id": event.turn_id,
            },
        )
        self._llm_spans[event.frame_id or ""] = span

    def _handle_llm_complete(self, event: LLMCompleteEvent) -> None:
        """Complete the LLM span for this frame."""
        span = self._llm_spans.pop(event.frame_id or "")
        if span is None:
            return

        span.set_attributes(
            {
                "llm.input_tokens": int(event.input_tokens or 0),
                "llm.output_tokens": int(event.output_tokens or 0),
                "llm.finish_reason": event.finish_reason or "unknown",
                "llm.latency_ms": int(event.latency_ms) if event.latency_ms else 0,
            }
        )
        if event.finish_reason and event.finish_reason != "stop":
            span.set_attribute("error", True)
            span.set_attribute("error.message", f"finish_reason: {event.finish_reason}")
        span.end()

    def _handle_tool_start(self, event: ToolStartEvent) -> None:
        """Start tool span as child of the frame span."""
        assert self._tracer is not None
        parent = self._frame_spans.get(event.frame_id or "")
        span = self._tracer.start_span(
            f"tool.{event.tool_name}",
            context=self._parent_context(parent),
            attributes={
                "tool.name": event.tool_name,
                "tool.call_id": event.tool_call_id,
                "frame.id": event.frame_id or "unknown",
                "turn.id": event.turn_id,
            },
        )
        self._tool_spans[event.tool_call_id] = span

    def _handle_tool_result(self, event: ToolResultEvent) -> None:
        """Complete the tool span for this call."""
        span = self._tool_spans.pop(event.tool_call_id)
        if span is None:
            return

        span.set_attributes(
            {
                "tool.success": event.success,
                "tool.output.length": len(str(event.output)),
            }
        )
        if event.error:
            span.set_attribute("error", True)
            span.set_attribute(
                "error.message", event.error.get("message", str(event.error))
            )
        span.end()

    def _handle_frame_complete(self, event: FrameCompleteEvent) -> None:
        """Complete the frame span (ending any unfinished child spans)."""
        span = self._frame_spans.pop(event.frame_id or "")
        if span is None:
            return

        orphaned_llm = self._llm_spans.pop(event.frame_id or "")
        if orphaned_llm is not None:
            orphaned_llm.set_attribute("error", True)
            orphaned_llm.set_attribute("error.message", "frame completed before llm")
            orphaned_llm.end()

        span.set_attributes(
            {
                "frame.status": event.status,
                "frame.duration_ms": event.duration_ms or 0,
                "frame.input_tokens": int(event.total_input_tokens or 0),
                "frame.output_tokens": int(event.total_output_tokens or 0),
            }
        )
        if event.error_message:
            span.set_attribute("error", True)
            span.set_attribute("error.message", event.error_message)
        span.end()

    def _handle_turn_complete(self, event: TurnCompleteEvent) -> None:
        """Complete the turn span (ending any unfinished child spans)."""
        span = self._turn_spans.pop(event.turn_id)
        if span is None:
            return

        for frame_span in list(self._frame_spans.values()):
            frame_span.set_attribute("error", True)
            frame_span.set_attribute("error.message", "turn completed before frame")
            frame_span.end()
        self._frame_spans.clear()

        span.set_attributes(
            {
                "turn.status": event.status,
                "turn.input_tokens": int(event.total_input_tokens or 0),
                "turn.output_tokens": int(event.total_output_tokens or 0),
                "turn.total_frames": event.total_frames or 0,
            }
        )
        if event.status not in ("completed", "partial"):
            span.set_attribute("error", True)
            span.set_attribute("error.message", f"status: {event.status}")
        span.end()

    def _parent_context(self, parent: Any) -> Any:
        """Build an OTel context carrying the parent span, if any."""
        if parent is None:
            return None
        from opentelemetry import trace

        return trace.set_span_in_context(parent)

    async def close(self) -> None:
        """End any leaked spans and flush the tracer provider."""
        for span in list(self._tool_spans.values()):
            span.end()
        for span in list(self._llm_spans.values()):
            span.end()
        for span in list(self._frame_spans.values()):
            span.end()
        for span in list(self._turn_spans.values()):
            span.end()
        self._tool_spans.clear()
        self._llm_spans.clear()
        self._frame_spans.clear()
        self._turn_spans.clear()

        try:
            from opentelemetry import trace

            provider = trace.get_tracer_provider()
            force_flush = getattr(provider, "force_flush", None)
            if callable(force_flush):
                force_flush()
        except Exception as e:
            logger.error("OTel close error", error=str(e))
