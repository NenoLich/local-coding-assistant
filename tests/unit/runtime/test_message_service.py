"""Tests for SessionState transcript append (the live mirror's seq source)."""

from local_coding_assistant.runtime.context.profile import (
    ContextProfile,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.session import Message, SessionState


def _session(**kwargs: object) -> SessionState:
    payload = {
        "context_profile": ContextProfile(
            runtime_context_config=RuntimeContextConfig(),
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
        ),
    }
    payload.update(kwargs)
    return SessionState(**payload)


def test_transcript_append_assigns_monotonic_sequences() -> None:
    session = _session(
        history=[
            Message(role="user", content="Earlier", seq=40),
            Message(role="assistant", content="Earlier answer", seq=41),
        ],
        history_sequence=41,
    )

    session.append_transcript_message(Message(role="user", content="Continue"))
    session.append_transcript_message(Message(role="assistant", content="Working"))

    assert [message.seq for message in session.history] == [40, 41, 42, 43]
    assert session.history_sequence == 43
    assert "seq" not in session.history[-1].model_dump()
