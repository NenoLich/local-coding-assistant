"""Unit tests for the CLI run flow adapter.

Locks the two contracts between the CLI and the runtime:
- config-affecting run options are delegated to
  ``RuntimeManager.apply_run_options`` (no config key knowledge in the CLI)
- one query dispatches to exactly one runtime entrypoint, selected by the
  message/selectors combination (run/Chat use the message-shaped
  entrypoints; text-less resume re-enters an interrupted turn)
"""

from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

from local_coding_assistant.cli.run_flow import RunOptions, execute_query
from local_coding_assistant.core.exceptions import CLIError
from local_coding_assistant.runtime.context.profile import (
    ContextProfile,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.options import RunOptions as RunConfigOptions
from local_coding_assistant.runtime.resume import ResumeValidationError
from local_coding_assistant.runtime.session import SessionState


class _FakePipeline:
    """Minimal event pipeline recording subscribe/unsubscribe."""

    def __init__(self) -> None:
        self.subscribed: list[Any] = []
        self.unsubscribed: list[Any] = []

    def subscribe(self, subscriber: Any) -> None:
        self.subscribed.append(subscriber)

    def unsubscribe(self, subscriber: Any) -> None:
        self.unsubscribed.append(subscriber)


class _FakeRuntime:
    """RuntimeManager stand-in recording entrypoint and option calls."""

    def __init__(self) -> None:
        self.event_pipeline = _FakePipeline()
        self.applied_options: list[RunConfigOptions] = []
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def apply_run_options(self, options: RunConfigOptions) -> None:
        self.applied_options.append(options)

    async def start_session(self, text: str, **_kwargs: Any) -> SessionState:
        self.calls.append(("start_session", (text,), _kwargs))
        return _make_session()

    async def resume_session_with_message(
        self, session_id: str, text: str, **_kwargs: Any
    ) -> SessionState:
        self.calls.append(("resume_session_with_message", (session_id, text), _kwargs))
        return _make_session()

    async def resume_session(self, session_id: str, **_kwargs: Any) -> SessionState:
        self.calls.append(("resume_session", (session_id,), _kwargs))
        return _make_session()

    async def respond_approval(
        self, session_id: str, turn_id: str, responses: dict[str, str]
    ) -> SessionState:
        self.calls.append(("respond_approval", (session_id, turn_id, responses), {}))
        return _make_session()


def _make_session() -> SessionState:
    return SessionState(
        context_profile=ContextProfile(
            runtime_context_config=RuntimeContextConfig(),
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
        )
    )


def _make_console() -> Console:
    return Console(record=True, width=100, color_system=None)


class TestExecuteQueryDispatch:
    """One query -> exactly one runtime entrypoint."""

    async def test_new_session_dispatches_to_start_session(self) -> None:
        runtime = _FakeRuntime()

        report, session = await execute_query(
            runtime,  # type: ignore[arg-type]
            "hello",
            RunOptions(),
            _make_console(),
        )

        assert [name for name, _, _ in runtime.calls] == ["start_session"]
        assert runtime.calls[0][1] == ("hello",)
        assert session is not None
        assert report is None  # no events streamed in the fake run

    async def test_continue_session_dispatches_to_resume_with_message(self) -> None:
        runtime = _FakeRuntime()

        await execute_query(
            runtime,  # type: ignore[arg-type]
            "follow up",
            RunOptions(session_id="sess-1"),
            _make_console(),
        )

        assert runtime.calls == [
            (
                "resume_session_with_message",
                ("sess-1", "follow up"),
                {"turn_id": None, "turn_index": None},
            )
        ]

    async def test_no_message_dispatches_to_resume_session(self) -> None:
        runtime = _FakeRuntime()

        # Crash recovery: continue the interrupted turn as-is.
        await execute_query(
            runtime,  # type: ignore[arg-type]
            None,
            RunOptions(session_id="sess-1"),
            _make_console(),
        )

        assert runtime.calls == [
            (
                "resume_session",
                ("sess-1",),
                {"turn_id": None, "turn_index": None, "frame_id": None},
            )
        ]

    async def test_no_message_with_selector_dispatches_to_resume_session(
        self,
    ) -> None:
        runtime = _FakeRuntime()

        # A past turn: fork-and-continue (no new message).
        await execute_query(
            runtime,  # type: ignore[arg-type]
            None,
            RunOptions(
                session_id="sess-1",
                resume_turn_id="turn-9",
                resume_turn_index=9,
                resume_frame_id="frame-1",
            ),
            _make_console(),
        )

        assert runtime.calls == [
            (
                "resume_session",
                ("sess-1",),
                {"turn_id": "turn-9", "turn_index": 9, "frame_id": "frame-1"},
            )
        ]

    async def test_message_with_selector_branches_with_new_turn(
        self,
    ) -> None:
        runtime = _FakeRuntime()

        await execute_query(
            runtime,  # type: ignore[arg-type]
            "try again differently",
            RunOptions(session_id="sess-1", resume_turn_index=2),
            _make_console(),
        )

        assert runtime.calls == [
            (
                "resume_session_with_message",
                ("sess-1", "try again differently"),
                {"turn_id": None, "turn_index": 2},
            )
        ]

    async def test_blank_message_is_treated_as_recovery(self) -> None:
        runtime = _FakeRuntime()

        await execute_query(
            runtime,  # type: ignore[arg-type]
            "   ",
            RunOptions(session_id="sess-1"),
            _make_console(),
        )

        assert runtime.calls == [
            (
                "resume_session",
                ("sess-1",),
                {"turn_id": None, "turn_index": None, "frame_id": None},
            )
        ]

    async def test_no_message_without_session_id_fails(self) -> None:
        runtime = _FakeRuntime()

        with pytest.raises(CLIError):
            await execute_query(runtime, None, RunOptions(), _make_console())  # type: ignore[arg-type]

    async def test_resume_validation_error_becomes_cli_error(self) -> None:
        runtime = _FakeRuntime()

        async def _reject(session_id: str, **_kwargs: Any) -> SessionState:
            raise ResumeValidationError(
                "Session sess-1 is idle (no interrupted turn). "
                "Use resume_session_with_message to add input."
            )

        runtime.resume_session = _reject  # type: ignore[method-assign]

        with pytest.raises(CLIError) as exc:
            await execute_query(
                runtime, None, RunOptions(session_id="sess-1"), _make_console()
            )  # type: ignore[arg-type]

        assert "is idle" in str(exc.value)
        assert "resume_session_with_message" in str(exc.value)


class TestExecuteQueryOptions:
    """Run options are delegated to the runtime, not applied by the CLI."""

    async def test_options_forwarded_to_runtime(self) -> None:
        runtime = _FakeRuntime()
        options = RunOptions(
            model="gpt-5",
            agent_mode="frame",
            tool_call_mode="ptc",
            sandbox_session="shared",
            session_id="sess-1",
            verbose=True,
        )

        await execute_query(runtime, "hello", options, _make_console())  # type: ignore[arg-type]

        assert runtime.applied_options == [options]
        # The CLI must not touch the config manager directly.
        assert not hasattr(runtime, "config_manager")

    async def test_resume_selectors_are_local_options_only(self) -> None:
        """Selector fields must not leak into the runtime run-options contract."""
        run_options = RunOptions(session_id="sess-1", resume_turn_index=2)
        config_options = RunConfigOptions()

        assert run_options.to_session_overrides() == {}
        assert config_options.to_session_overrides() == {}
