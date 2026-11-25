"""Event repository unit tests."""

from __future__ import annotations

import uuid


class TestEventRepo:
    async def test_append_inserts_event(self, event_repo, session_repo) -> None:
        """append(): inserts event with all fields."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        frame_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())

        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=1,
            event_type="llm_start",
            data_json='{"prompt": "hello"}',
            timestamp_ms=1000,
        )

        events = await event_repo.get_by_frame(frame_id)
        assert len(events) == 1
        assert events[0]["trace_id"] == trace_id
        assert events[0]["session_id"] == session_id
        assert events[0]["turn_id"] == turn_id
        assert events[0]["frame_id"] == frame_id
        assert events[0]["seq"] == 1
        assert events[0]["type"] == "llm_start"
        assert events[0]["data_json"] == '{"prompt": "hello"}'
        assert events[0]["timestamp"] == 1000

    async def test_get_by_frame_ordered_by_id(
        self, event_repo, session_repo, turn_repo, frame_repo
    ) -> None:
        """get_by_frame(): ordered by seq."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=trace_id,
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        # Append events out of order by seq
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=3,
            event_type="llm_complete",
            data_json='{"result": "done"}',
            timestamp_ms=3000,
        )
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=1,
            event_type="llm_start",
            data_json='{"prompt": "hello"}',
            timestamp_ms=1000,
        )
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=2,
            event_type="tool_start",
            data_json='{"tool": "read"}',
            timestamp_ms=2000,
        )

        events = await event_repo.get_by_frame(frame_id)
        assert len(events) == 3
        assert events[0]["id"] == 1
        assert events[1]["id"] == 2
        assert events[2]["id"] == 3

    async def test_get_by_turn_ordered_by_id(
        self, event_repo, session_repo, turn_repo, frame_repo
    ) -> None:
        """get_by_turn(): ordered by timestamp (chronological)."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=trace_id,
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame1_id = str(uuid.uuid4())
        frame2_id = str(uuid.uuid4())

        await frame_repo.create(
            frame_id=frame1_id,
            session_id=session_id,
            trace_id=trace_id,
            iteration=1,
            status="completed",
            started_at_ms=1000,
            turn_id=turn_id,
        )
        await frame_repo.create(
            frame_id=frame2_id,
            session_id=session_id,
            trace_id=trace_id,
            iteration=2,
            status="completed",
            started_at_ms=2000,
            turn_id=turn_id,
        )

        # Append events with timestamps that differ from seq order
        # Frame 1: seq=2 happens at timestamp=2000, seq=1 at timestamp=1000
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame1_id,
            seq=2,
            event_type="llm_complete",
            data_json='{"result": "done"}',
            timestamp_ms=2000,
        )
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame1_id,
            seq=1,
            event_type="llm_start",
            data_json='{"prompt": "hello"}',
            timestamp_ms=1000,
        )
        # Frame 2: seq=3 at timestamp=3000
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame2_id,
            seq=3,
            event_type="tool_start",
            data_json='{"tool": "read"}',
            timestamp_ms=3000,
        )

        events = await event_repo.get_by_turn(turn_id)
        assert len(events) == 3
        # Should be ordered by id
        assert events[0]["id"] == 1
        assert events[1]["id"] == 2
        assert events[2]["id"] == 3

    async def test_get_by_frames_chunked_in_clause(
        self, event_repo, session_repo, turn_repo, frame_repo
    ) -> None:
        """get_by_frames(): chunked IN clause (100), ordered by timestamp."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=trace_id,
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_ids = []
        for i in range(5):
            frame_id = str(uuid.uuid4())
            frame_ids.append(frame_id)
            await frame_repo.create(
                frame_id=frame_id,
                session_id=session_id,
                trace_id=trace_id,
                iteration=i,
                status="completed",
                started_at_ms=i * 1000,
                turn_id=turn_id,
            )
            await event_repo.append(
                trace_id=trace_id,
                session_id=session_id,
                turn_id=turn_id,
                frame_id=frame_id,
                seq=i + 1,
                event_type=f"event_{i}",
                data_json=f'{{"frame": {i}}}',
                timestamp_ms=i * 1000,
            )

        events = await event_repo.get_by_frames(frame_ids)
        assert len(events) == 5
        # Verify ordered by timestamp across all frames
        assert events[0]["timestamp"] == 0
        assert events[1]["timestamp"] == 1000
        assert events[2]["timestamp"] == 2000
        assert events[3]["timestamp"] == 3000
        assert events[4]["timestamp"] == 4000

    async def test_get_by_frames_empty_list(self, event_repo) -> None:
        """get_by_frames(): returns empty list for empty frame_ids."""
        events = await event_repo.get_by_frames([])
        assert events == []

    async def test_get_by_session_ordered_by_id(
        self, event_repo, session_repo, turn_repo, frame_repo
    ) -> None:
        """get_by_session(): ordered by timestamp (chronological)."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=trace_id,
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=trace_id,
            iteration=1,
            status="completed",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        # Append events out of order by timestamp
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=3,
            event_type="llm_complete",
            data_json='{"result": "done"}',
            timestamp_ms=3000,
        )
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=1,
            event_type="llm_start",
            data_json='{"prompt": "hello"}',
            timestamp_ms=1000,
        )
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=2,
            event_type="tool_start",
            data_json='{"tool": "read"}',
            timestamp_ms=2000,
        )

        events = await event_repo.get_by_session(session_id)
        assert len(events) == 3
        # Should be ordered by id
        assert events[0]["id"] == 1
        assert events[1]["id"] == 2
        assert events[2]["id"] == 3

    async def test_get_by_trace_ordered_by_id(
        self, event_repo, session_repo, turn_repo, frame_repo
    ) -> None:
        """get_by_trace(): ordered by timestamp (chronological)."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        trace_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=trace_id,
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=trace_id,
            iteration=1,
            status="completed",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        # Append events out of order by timestamp
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=3,
            event_type="llm_complete",
            data_json='{"result": "done"}',
            timestamp_ms=3000,
        )
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=1,
            event_type="llm_start",
            data_json='{"prompt": "hello"}',
            timestamp_ms=1000,
        )
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=2,
            event_type="tool_start",
            data_json='{"tool": "read"}',
            timestamp_ms=2000,
        )

        events = await event_repo.get_by_trace(trace_id)
        assert len(events) == 3
        # Should be ordered by id
        assert events[0]["id"] == 1
        assert events[1]["id"] == 2
        assert events[2]["id"] == 3

    async def test_get_by_type_ordered_by_timestamp_desc(
        self, event_repo, session_repo, turn_repo, frame_repo
    ) -> None:
        """get_by_type(): ordered by timestamp DESC, limit."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=trace_id,
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=trace_id,
            iteration=1,
            status="completed",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        # Append events of same type with different timestamps
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=1,
            event_type="llm_start",
            data_json='{"prompt": "hello"}',
            timestamp_ms=1000,
        )
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=2,
            event_type="llm_start",
            data_json='{"prompt": "world"}',
            timestamp_ms=3000,
        )
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=3,
            event_type="llm_start",
            data_json='{"prompt": "test"}',
            timestamp_ms=2000,
        )

        events = await event_repo.get_by_type("llm_start", limit=10)
        assert len(events) == 3
        # Ordered by timestamp DESC
        assert events[0]["timestamp"] == 3000
        assert events[1]["timestamp"] == 2000
        assert events[2]["timestamp"] == 1000

    async def test_get_by_type_respects_limit(
        self, event_repo, session_repo, turn_repo, frame_repo
    ) -> None:
        """get_by_type(): respects limit parameter."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=trace_id,
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=trace_id,
            iteration=1,
            status="completed",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        for i in range(5):
            await event_repo.append(
                trace_id=trace_id,
                session_id=session_id,
                turn_id=turn_id,
                frame_id=frame_id,
                seq=i + 1,
                event_type="llm_start",
                data_json=f'{{"i": {i}}}',
                timestamp_ms=i * 1000,
            )

        events = await event_repo.get_by_type("llm_start", limit=3)
        assert len(events) == 3

    async def test_delete_by_frame_removes_events(
        self, event_repo, session_repo, turn_repo, frame_repo
    ) -> None:
        """delete_by_frame(): removes events for frame."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=trace_id,
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame1_id = str(uuid.uuid4())
        frame2_id = str(uuid.uuid4())

        await frame_repo.create(
            frame_id=frame1_id,
            session_id=session_id,
            trace_id=trace_id,
            iteration=1,
            status="completed",
            started_at_ms=1000,
            turn_id=turn_id,
        )
        await frame_repo.create(
            frame_id=frame2_id,
            session_id=session_id,
            trace_id=trace_id,
            iteration=2,
            status="completed",
            started_at_ms=2000,
            turn_id=turn_id,
        )

        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame1_id,
            seq=1,
            event_type="llm_start",
            data_json='{"frame": 1}',
            timestamp_ms=1000,
        )
        await event_repo.append(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame2_id,
            seq=2,
            event_type="llm_start",
            data_json='{"frame": 2}',
            timestamp_ms=2000,
        )

        # Delete events for frame1
        await event_repo.delete_by_frame(frame1_id)

        events = await event_repo.get_by_session(session_id)
        assert len(events) == 1
        assert events[0]["frame_id"] == frame2_id
