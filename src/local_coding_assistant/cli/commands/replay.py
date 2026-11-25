"""Reconstruct and display a stored session/turn/frame from durable data.

Read-only: loads the durable session graph (turns -> frames -> events and
messages) via the store replay API. Unresolvable context references are
reported explicitly instead of being silently reconstructed.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import typer

from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.storage.contracts import SessionReplay, TurnReplay
from local_coding_assistant.storage.execution_store_reader import ExecutionStoreReader
from local_coding_assistant.utils.logging import get_logger, setup_logging

app = typer.Typer(name="replay", help="Replay a stored session or turn (read-only)")
log = get_logger("cli.replay")

_DEFAULT_CONTENT_MAX_LEN = 400


@app.command()
@safe_entrypoint("cli.replay.show")
def show(
    session_id: str = typer.Argument(help="Session id to replay"),
    turn_id: str = typer.Option(None, "--turn", help="Limit to one turn"),
    frame_id: str = typer.Option(None, "--frame", help="Limit to one frame"),
    events: bool = typer.Option(
        False, "--events", help="Include per-frame event timelines"
    ),
    output_format: str = typer.Option(
        "plain", "--format", help="Output format: plain or json"
    ),
    content_max_len: int = typer.Option(
        _DEFAULT_CONTENT_MAX_LEN,
        "--content-max-len",
        help="Max characters of content to display (0 = no truncation)",
    ),
    store_db_path: str = typer.Option(
        "data/execution_store.db",
        "--store-db-path",
        help="Path to the execution store database",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Replay a session (or a single turn) from durable storage."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    setup_logging(level=level)

    async def _main() -> int:
        db_path = Path(store_db_path)
        if not db_path.exists():
            typer.echo(f"Error: Execution store not found: {db_path}", err=True)
            return 1

        store = ExecutionStoreReader(db_path)
        await store.init()
        try:
            replay = await store.load_replay_batch(session_id)
            if replay is None:
                typer.echo(f"Session not found: {session_id}", err=True)
                return 1

            if turn_id:
                replay = _filter_turn(replay, turn_id)
                if replay is None:
                    typer.echo(f"Turn not found: {turn_id}", err=True)
                    return 1

            if output_format.lower() == "json":
                typer.echo(
                    json.dumps(
                        _render_json(
                            replay,
                            with_events=events,
                            content_max_len=content_max_len,
                            frame_id=frame_id,
                        ),
                        indent=2,
                        default=str,
                    )
                )
                return 0

            _render_plain(
                replay,
                frame_id=frame_id,
                with_events=events,
                content_max_len=content_max_len,
            )
        finally:
            await store.close()
        return 0

    raise typer.Exit(code=asyncio.run(_main()))


def _filter_turn(replay: SessionReplay, turn_id: str) -> SessionReplay | None:
    """Return a copy of the replay limited to one turn."""
    turns = [t for t in replay.turns if t.turn.id == turn_id]
    if not turns:
        return None
    return SessionReplay(session=replay.session, turns=turns)


def _truncate(text: str | None, max_len: int) -> str:
    """Truncate text to max_len chars; 0 or negative means no truncation."""
    if text is None:
        return ""
    if max_len <= 0:
        return text
    return str(text)[:max_len]


def _render_plain(
    replay: SessionReplay,
    *,
    frame_id: str | None,
    with_events: bool,
    content_max_len: int,
) -> None:
    """Print the replay graph as an indented trace tree."""
    session = replay.session
    typer.echo(f"session {session.id}  status={session.status}")
    typer.echo(f"input: {_truncate(session.user_input, content_max_len)}")
    typer.echo("")

    for turn in replay.turns:
        _render_turn(
            turn,
            frame_id=frame_id,
            with_events=with_events,
            content_max_len=content_max_len,
        )


def _render_turn(
    turn: TurnReplay, *, frame_id: str | None, with_events: bool, content_max_len: int
) -> None:
    """Print one turn with frames, messages and optional events."""
    stored_turn = turn.turn
    typer.echo(
        f"turn {stored_turn.id} (index={stored_turn.turn_index}) "
        f"status={stored_turn.status} input_tokens={int(stored_turn.total_input_tokens) or 0} output_tokens={int(stored_turn.total_output_tokens) or 0}"
    )
    typer.echo(f"input: {_truncate(stored_turn.user_input, content_max_len)}")

    snapshot = turn.snapshot
    if snapshot is None:
        typer.echo("  context snapshot: MISSING (INCOMPLETE)")
    else:
        missing = turn.missing_refs
        incomplete = missing.tool_ids or missing.template_ids or missing.repo_hashes
        note = " (INCOMPLETE: missing refs)" if incomplete else ""
        typer.echo(
            f"  context snapshot {snapshot.id} (hash={snapshot.content_hash}){note}"
        )
        if incomplete:
            typer.echo(f"    missing refs: {_missing_refs_summary(missing)}")

    if not turn.frames:
        typer.echo("  no frames recorded")
    for frame in turn.frames:
        if frame_id and frame.id != frame_id:
            continue
        typer.echo(
            f"  frame {frame.id} (iteration={frame.iteration}) "
            f"status={frame.status} input_tokens={int(frame.input_tokens) or 0} output_tokens={int(frame.output_tokens) or 0}"
        )
        if frame.error_message:
            typer.echo(f"    error: {frame.error_message}")
        if with_events:
            for event in turn.frame_events.get(frame.id, []):
                payload = event.get("data_json", "{}")
                payload = _truncate(payload, content_max_len)
                typer.echo(f"    [{event.get('seq')}] {event.get('type')} {payload}")

    typer.echo("  messages:")
    for message in turn.messages:
        content = str(message.get("content", ""))
        typer.echo(
            f"    [{message.get('seq')}] {message.get('role')}: {_truncate(content, content_max_len)}"
        )
    typer.echo("")


def _missing_refs_summary(missing: Any) -> str:
    """Summarize missing refs for display."""
    parts: list[str] = []
    if missing.tool_ids:
        parts.append(f"tools={missing.tool_ids}")
    if missing.template_ids:
        parts.append(f"templates={missing.template_ids}")
    if missing.repo_hashes:
        parts.append(f"repo={missing.repo_hashes}")
    return ", ".join(parts) if parts else "none"


def _parse_json_field(value: str | None) -> Any:
    """Parse a stored JSON string into an object for indented rendering."""
    if not value:
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def _render_json(
    replay: SessionReplay,
    *,
    with_events: bool,
    content_max_len: int,
    frame_id: str | None = None,
) -> dict[str, Any]:
    """Convert the replay graph to a JSON-serializable structure."""
    session = replay.session
    return {
        "session": {
            "id": session.id,
            "status": session.status,
            "user_input": _truncate(session.user_input, content_max_len),
            "model_name": session.model_name,
            "input_tokens": int(session.total_input_tokens or 0),
            "output_tokens": int(session.total_output_tokens or 0),
            "total_turns": session.total_turns,
            "total_frames": session.total_frames,
        },
        "turns": [
            {
                "id": turn.turn.id,
                "turn_index": turn.turn.turn_index,
                "status": turn.turn.status,
                "user_input": _truncate(turn.turn.user_input, content_max_len),
                "trace_id": turn.turn.trace_id,
                "input_tokens": int(turn.turn.total_input_tokens or 0),
                "output_tokens": int(turn.turn.total_output_tokens or 0),
                "snapshot": (
                    {
                        "id": turn.snapshot.id,
                        "content_hash": turn.snapshot.content_hash,
                    }
                    if turn.snapshot
                    else None
                ),
                "missing_refs": {
                    "tool_ids": turn.missing_refs.tool_ids,
                    "template_ids": turn.missing_refs.template_ids,
                    "repo_hashes": turn.missing_refs.repo_hashes,
                },
                "frames": [
                    {
                        "id": frame.id,
                        "iteration": frame.iteration,
                        "status": frame.status,
                        "input_tokens": int(frame.input_tokens or 0),
                        "output_tokens": int(frame.output_tokens or 0),
                        "error_message": frame.error_message,
                        "events": (
                            [
                                _render_event(event, content_max_len)
                                for event in turn.frame_events.get(frame.id, [])
                            ]
                            if with_events
                            else []
                        ),
                    }
                    for frame in turn.frames
                    if not frame_id or frame.id == frame_id
                ],
                "messages": [
                    {
                        "id": message.get("id"),
                        "seq": message.get("seq"),
                        "role": message.get("role"),
                        "content": _truncate(
                            message.get("content", ""), content_max_len
                        ),
                        "tool_call_id": message.get("tool_call_id"),
                        "tool_name": message.get("tool_name"),
                        "metadata_json": _parse_json_field(
                            message.get("metadata_json")
                        ),
                        "dedupe_key": message.get("dedupe_key"),
                        "created_at": message.get("created_at"),
                    }
                    for message in turn.messages
                ],
            }
            for turn in replay.turns
        ],
    }


def _render_event(event: dict[str, Any], content_max_len: int) -> dict[str, Any]:
    """Render one stored event with indented data_json and optional truncation."""
    data = _parse_json_field(event.get("data_json"))
    if isinstance(data, str) and content_max_len > 0:
        data = _truncate(data, content_max_len)
    return {
        "id": event.get("id"),
        "trace_id": event.get("trace_id"),
        "session_id": event.get("session_id"),
        "turn_id": event.get("turn_id"),
        "frame_id": event.get("frame_id"),
        "seq": event.get("seq"),
        "type": event.get("type"),
        "data_json": data,
        "timestamp_ms": event.get("timestamp_ms"),
        "timestamp": event.get("timestamp"),
    }
