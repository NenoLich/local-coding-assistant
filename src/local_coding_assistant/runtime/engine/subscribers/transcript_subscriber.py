"""TranscriptSubscriber - mirrors the projected transcript into SessionState.

The store adapter persists the same projection atomically with each event;
this subscriber keeps the in-memory session (used for prompt resolution)
in lockstep. Messages are deduplicated per (session, message key) derived
from the source event, so replaying an event never appends twice.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from local_coding_assistant.runtime.events.contracts import RuntimeEvent
from local_coding_assistant.runtime.events.projector import MessageService
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.runtime.session import SessionState

logger = get_logger("runtime.engine.subscribers.transcript_subscriber")

# Session lookup: returns the live cached session, or None when this process
# does not hold it (the store still persists the projection).
SessionLookup = Callable[[str], "SessionState | None"]


class TranscriptSubscriber:
    """Event subscriber that projects transcript messages into SessionState."""

    def __init__(self, session_lookup: SessionLookup) -> None:
        """Initialize the subscriber.

        Args:
            session_lookup: Callable resolving a session_id to its live
                SessionState, or None when unavailable
        """
        self._session_lookup = session_lookup
        self._message_service = MessageService()
        self._seen_keys: dict[str, set[str]] = {}

    async def on_event(self, event: RuntimeEvent) -> None:
        """Project one event into the live session history.

        Args:
            event: Runtime event to project
        """
        session = self._session_lookup(event.session_id)
        if session is None:
            # Session not live in this process; persistence is handled by
            # the store adapter, the in-memory mirror is best effort
            return

        built = self._message_service.build_message(event)
        if built is None:
            return

        message, dedupe_key = built
        seen = self._seen_keys.setdefault(event.session_id, set())
        if dedupe_key in seen:
            return

        if message.role == "tool" and message.tool_call_id:
            # Also dedupe against history loaded from the store on cold start
            if any(
                m.role == "tool" and m.tool_call_id == message.tool_call_id
                for m in session.history
            ):
                seen.add(dedupe_key)
                return

        session.append_transcript_message(message)
        seen.add(dedupe_key)
        logger.debug(
            "Transcript message appended",
            session_id=event.session_id,
            role=message.role,
            dedupe_key=dedupe_key,
        )

    async def close(self) -> None:
        """Cleanup resources."""
        self._seen_keys.clear()
