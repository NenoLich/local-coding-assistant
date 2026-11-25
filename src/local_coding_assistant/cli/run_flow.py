"""CLI-side adapter mapping CLI flags onto RuntimeManager entrypoints.

The runtime entrypoints (``start_session``, ``resume_session_with_message``,
``resume_session``, ``respond_approval``) are the only execution API; this
module adds the CLI-specific plumbing: option forwarding, a transient event
subscriber for streaming, the in-process approval loop, and report building.
Config-key translation for run options lives in the runtime layer
(``RuntimeManager.apply_run_options``).

Command mapping:

- ``locca run query TEXT`` / ``locca chat``:
  message present, no session -> ``start_session``
- ``locca run query TEXT --session-id X`` / next chat input:
  message present, session -> ``resume_session_with_message``
- ``locca sessions resume X`` (no TEXT):
  no message -> ``resume_session`` (crash recovery / fork-and-continue)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from rich.console import Console

from local_coding_assistant.cli.runtime_stream import (
    EventStreamRenderer,
    QueueSubscriber,
    build_report,
    prompt_approvals,
    pump_stream,
)
from local_coding_assistant.core.exceptions import CLIError
from local_coding_assistant.runtime.events.reporting import RunReport
from local_coding_assistant.runtime.manager import RuntimeManager
from local_coding_assistant.runtime.options import RunOptions as RunConfigOptions
from local_coding_assistant.runtime.resume import ResumeValidationError
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("cli.run_flow")

MAX_APPROVAL_ROUNDS = 8


@dataclass
class RunOptions(RunConfigOptions):
    """CLI-level options for one query execution.

    Extends the runtime run-options contract (config-affecting flags, applied
    by ``RuntimeManager.apply_run_options``) with CLI dispatch and rendering
    concerns. The ``resume_*`` selectors address a turn inside an existing
    session: given a message they branch/extend it, without one they re-enter
    it where it stopped.
    """

    session_id: str | None = None
    resume_turn_id: str | None = None
    resume_turn_index: int | None = None
    resume_frame_id: str | None = None
    verbose: bool = False


async def _dispatch_query(
    runtime: RuntimeManager, message: str | None, options: RunOptions
) -> SessionState:
    """Route a query to the matching runtime entrypoint.

    Args:
        runtime: Initialized RuntimeManager
        message: New user message, or None to re-enter the session where it
            stopped (recovery: the interrupted turn is continued as-is)
        options: CLI-level options

    Returns:
        SessionState after the entrypoint returns (branched session when the
        selected turn was not the last one)

    Raises:
        CLIError: If no session is given but no message either
        ResumeValidationError: If the selected turn cannot be resumed
    """
    if options.session_id is None:
        if message is None:
            raise CLIError("A query text is required to start a new session")
        return await runtime.start_session(message)

    if message is None or not message.strip():
        # No message: continue the interrupted turn from where it paused
        # (crash recovery), optionally at a past turn or frame. A past turn
        # branches a new session at that point.
        return await runtime.resume_session(
            options.session_id,
            turn_id=options.resume_turn_id,
            turn_index=options.resume_turn_index,
            frame_id=options.resume_frame_id,
        )

    # With a message: one new turn in the session (branched when the
    # selector points before the last turn).
    return await runtime.resume_session_with_message(
        options.session_id,
        message,
        turn_id=options.resume_turn_id,
        turn_index=options.resume_turn_index,
    )


async def execute_query(
    runtime: RuntimeManager,
    text: str | None,
    options: RunOptions,
    console: Console,
) -> tuple[RunReport | None, SessionState]:
    """Execute one query with console streaming and approval handling.

    Subscribes a transient queue subscriber to the runtime pipeline for the
    duration of the call, so events are rendered live and accumulated for the
    report without changing the runtime API.

    Args:
        runtime: Initialized RuntimeManager
        text: User query, or None to resume an interrupted session (see
            ``_dispatch_query``)
        options: CLI-level options
        console: Rich console for streaming output

    Returns:
        Tuple of (RunReport or None, resulting SessionState)

    Raises:
        CLIError: On invalid input or a resume the runtime refuses
        RuntimeError: If the runtime has no event pipeline
    """
    pipeline = runtime.event_pipeline
    if pipeline is None:
        raise RuntimeError(
            "Runtime event pipeline unavailable (missing store/pipeline wiring)"
        )

    # Config-affecting options are translated and applied by the runtime,
    # keeping config key knowledge out of the front-end.
    runtime.apply_run_options(options)

    renderer = EventStreamRenderer(console, verbose=options.verbose)
    subscriber = QueueSubscriber()
    pipeline.subscribe(subscriber)
    pump = asyncio.create_task(pump_stream(subscriber, renderer))

    try:
        try:
            session = await _dispatch_query(runtime, text, options)
        except ResumeValidationError as exc:
            # Surface the planner's actionable message as a clean CLI error
            raise CLIError(str(exc)) from exc

        # Approval rounds: the turn pauses with in-memory pending approvals;
        # respond in this process to resume the same turn.
        for _ in range(MAX_APPROVAL_ROUNDS):
            if not renderer.pending_approvals:
                break
            pending = renderer.pending_approvals
            renderer.pending_approvals = []
            last_turn = session.get_last_turn()
            if last_turn is None:
                logger.warning("Approval requested but no turn found in session")
                break
            console.print(f"[dim]turn {last_turn.turn_id} paused for approval[/dim]")
            responses = await prompt_approvals(pending)
            session = await runtime.respond_approval(
                session.id, last_turn.turn_id, responses
            )
        else:
            logger.warning("Approval round limit reached, stopping")

    finally:
        await subscriber.close()
        await pump
        pipeline.unsubscribe(subscriber)

    return build_report(renderer), session
