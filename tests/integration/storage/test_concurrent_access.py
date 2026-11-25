"""Integration tests for concurrent access to ExecutionStore.

Covers:
- Concurrent process_event: seq assignment per turn/frame, no collisions
- Concurrent append_message: dedupe_key prevents duplicates
- Multiple concurrent writers: _write_lock serializes batches
- Read during write: WAL mode allows concurrent readers
"""

from __future__ import annotations

import asyncio
import uuid

from local_coding_assistant.runtime.events.contracts import (
    FrameCompleteEvent,
    FrameStartEvent,
    LLMCompleteEvent,
    SessionStartEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.storage.execution_store import ExecutionStore


def _new_id() -> str:
    return f"t_{uuid.uuid4().hex[:12]}"


async def _fire_session_start(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    user_input: str = "hello",
) -> None:
    await store.process_event(
        SessionStartEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            user_input=user_input,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )
    )


async def _fire_turn_start(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    turn_index: int = 0,
    user_input: str = "turn query",
) -> None:
    await store.process_event(
        TurnStartEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            user_input=user_input,
            turn_index=turn_index,
            config_snapshot_json="{}",
        )
    )


async def _fire_frame_start(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    iteration: int = 0,
) -> None:
    await store.process_event(
        FrameStartEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            frame_id=frame_id,
            iteration=iteration,
            context_snapshot_id="snap_default",
            context_hash="hash_default",
        )
    )


async def _fire_llm_complete(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    content: str = "response",
    input_tokens: int = 10,
    output_tokens: int = 0,
) -> None:
    await store.process_event(
        LLMCompleteEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            frame_id=frame_id,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model="gpt-4",
        )
    )


async def _fire_frame_complete(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    status: str = "completed",
) -> None:
    await store.process_event(
        FrameCompleteEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            frame_id=frame_id,
            status=status,
            duration_ms=100,
            total_input_tokens=0,
            total_output_tokens=0,
            model_name="gpt-4",
            finish_reason="stop",
        )
    )


async def _fire_turn_complete(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    status: str = "completed",
    final_answer: str | None = None,
) -> None:
    await store.process_event(
        TurnCompleteEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            status=status,
            total_input_tokens=0,
            total_output_tokens=0,
            total_frames=0,
            final_answer=final_answer,
        )
    )


async def _setup_session_turn_frame(
    store: ExecutionStore,
) -> tuple[str, str, str]:
    """Create a session, turn and frame and flush them to the DB."""
    session_id = _new_id()
    turn_id = _new_id()
    frame_id = _new_id()
    await _fire_session_start(store, session_id, turn_id, user_input="concurrent")
    await _fire_turn_start(store, session_id, turn_id, turn_index=0)
    await _fire_frame_start(store, session_id, turn_id, frame_id, iteration=0)
    await store.flush()
    return session_id, turn_id, frame_id


class TestConcurrentAccess:
    async def test_concurrent_process_event_no_seq_collisions(
        self, execution_store: ExecutionStore
    ) -> None:
        """Concurrent process_event: seq assignment per turn/frame, no collisions."""
        store = execution_store
        session_id, turn_id, frame_id = await _setup_session_turn_frame(store)

        count = 20
        await asyncio.gather(
            *[
                _fire_llm_complete(
                    store,
                    session_id,
                    turn_id,
                    frame_id,
                    content=f"resp {i}",
                    input_tokens=5,
                    output_tokens=0,
                )
                for i in range(count)
            ]
        )

        events = await store.get_frame_events(frame_id)
        assert len(events) == count + 1  # frame_start + count llm_complete
        seqs = sorted(e["seq"] for e in events)
        # Setup consumes seqs 1 (session_start), 2 (turn_start), 3 (frame_start),
        # so the frame's events continue the per-session counter at 3..count+3.
        assert seqs == list(range(3, count + 4))
        assert len(set(seqs)) == count + 1

        frame = await store.frame_repo.get(frame_id)
        assert frame is not None
        assert frame.input_tokens == count * 5

        messages = await store.get_all_messages(session_id)
        assistant = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant) == 1  # dedupe_key "assistant:{frame_id}"

    async def test_concurrent_append_message_dedupe_no_duplicates(
        self, execution_store: ExecutionStore
    ) -> None:
        """Concurrent append_message: dedupe_key prevents duplicates."""
        store = execution_store
        session_id, turn_id, frame_id = await _setup_session_turn_frame(store)

        dedupe_key = "tool:turn_1:call_1"
        count = 15
        await asyncio.gather(
            *[
                store.append_message(
                    session_id=session_id,
                    turn_id=turn_id,
                    frame_id=frame_id,
                    seq=None,
                    role="tool",
                    content="result",
                    tool_call_id="call_1",
                    tool_name="calculator",
                    dedupe_key=dedupe_key,
                )
                for _ in range(count)
            ]
        )

        messages = await store.get_all_messages(session_id)
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1

    async def test_concurrent_writes_across_frames_serialized(
        self, execution_store: ExecutionStore
    ) -> None:
        """Multiple concurrent writers: _write_lock serializes batches, no errors."""
        store = execution_store
        session_id = _new_id()
        turn_id = _new_id()

        await _fire_session_start(store, session_id, turn_id, user_input="multi-frame")
        await _fire_turn_start(store, session_id, turn_id, turn_index=0)

        frame_ids = [_new_id() for _ in range(10)]
        for iteration, frame_id in enumerate(frame_ids):
            await _fire_frame_start(
                store, session_id, turn_id, frame_id, iteration=iteration
            )
        await store.flush()

        async def write_frame(frame_id: str) -> None:
            await asyncio.gather(
                *[
                    _fire_llm_complete(
                        store,
                        session_id,
                        turn_id,
                        frame_id,
                        input_tokens=5,
                        output_tokens=0,
                    )
                    for _ in range(5)
                ]
            )
            await _fire_frame_complete(
                store, session_id, turn_id, frame_id, status="completed"
            )

        await asyncio.gather(*[write_frame(fid) for fid in frame_ids])

        frames = await store.frame_repo.list_by_session(session_id)
        assert len(frames) == 10
        for frame in frames:
            assert frame.status == "completed"
            assert frame.input_tokens == 25

        events = await store.event_repo.get_by_session(session_id)
        assert len(events) == 72  # 12 setup + 10 * (5 llm + 1 frame_complete)

        by_frame: dict[str, list[int]] = {}
        for event in events:
            if event["frame_id"]:
                by_frame.setdefault(event["frame_id"], []).append(event["seq"])
        for frame_id, frame_seqs in by_frame.items():
            assert len(frame_seqs) == len(set(frame_seqs)), (
                f"seq collisions for frame {frame_id}"
            )

    async def test_read_during_write_wal_consistency(
        self, execution_store: ExecutionStore
    ) -> None:
        """Read during write: WAL mode allows concurrent readers without errors."""
        store = execution_store
        session_id, turn_id, frame_id = await _setup_session_turn_frame(store)

        read_errors: list[Exception] = []

        async def reader() -> None:
            try:
                for _ in range(10):
                    await store.get_session(session_id)
                    await store.get_frame_events(frame_id)
                    await store.frame_repo.get(frame_id)
            except Exception as exc:  # collect read failures
                read_errors.append(exc)

        async def writer() -> None:
            await asyncio.gather(
                *[
                    _fire_llm_complete(
                        store,
                        session_id,
                        turn_id,
                        frame_id,
                        input_tokens=3,
                        output_tokens=0,
                    )
                    for _ in range(10)
                ]
            )

        await asyncio.gather(reader(), writer())

        assert not read_errors, f"read errors during write: {read_errors}"

        events = await store.get_frame_events(frame_id)
        assert len(events) == 11  # frame_start + 10 llm_complete
        seqs = [e["seq"] for e in events]
        assert len(seqs) == len(set(seqs))
