"""Message repository unit tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime


class TestMessageRepo:
    async def test_append_inserts_message(self, message_repo, session_repo) -> None:
        """append(): INSERT OR IGNORE on (session_id, dedupe_key), seq = MAX(seq)+1 per session."""
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

        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="user",
            content="Hello",
            dedupe_key=f"user:{turn_id}",
        )

        messages = await message_repo.get_by_session(session_id)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[0]["seq"] == 1  # Auto-incremented

    async def test_append_dedupe_key_prevents_duplicates(
        self, message_repo, session_repo
    ) -> None:
        """dedupe_key: prevents duplicate projection from same event."""
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
        dedupe_key = f"user:{turn_id}"

        # Append same message twice with same dedupe_key
        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="user",
            content="Hello",
            dedupe_key=dedupe_key,
        )
        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="user",
            content="Hello",
            dedupe_key=dedupe_key,
        )

        messages = await message_repo.get_by_session(session_id)
        assert len(messages) == 1  # Only one inserted due to INSERT OR IGNORE

    async def test_append_seq_computation_null_to_max_plus_one(
        self, message_repo, session_repo
    ) -> None:
        """seq computation: NULL seq → MAX(seq)+1 within same transaction."""
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

        # Append three messages with seq=None
        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="user",
            content="Hello",
            dedupe_key=f"user:{turn_id}:1",
        )
        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="assistant",
            content="Hi",
            dedupe_key=f"assistant:{frame_id}:1",
        )
        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="user",
            content="Bye",
            dedupe_key=f"user:{turn_id}:2",
        )

        messages = await message_repo.get_by_session(session_id)
        assert len(messages) == 3
        assert messages[0]["seq"] == 1
        assert messages[1]["seq"] == 2
        assert messages[2]["seq"] == 3

    async def test_append_explicit_seq(self, message_repo, session_repo) -> None:
        """append(): respects explicit seq when provided."""
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

        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=10,
            role="user",
            content="Hello",
            dedupe_key=f"user:{turn_id}",
        )

        messages = await message_repo.get_by_session(session_id)
        assert len(messages) == 1
        assert messages[0]["seq"] == 10

    async def test_append_tool_call_fields(self, message_repo, session_repo) -> None:
        """append(): stores tool_call_id, tool_name, metadata_json."""
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

        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="tool",
            content="Tool result",
            tool_call_id="call_123",
            tool_name="read_file",
            metadata_json='{"args": {"path": "/tmp/file.txt"}}',
            dedupe_key=f"tool:{turn_id}:call_123",
        )

        messages = await message_repo.get_by_session(session_id)
        assert len(messages) == 1
        assert messages[0]["tool_call_id"] == "call_123"
        assert messages[0]["tool_name"] == "read_file"
        assert messages[0]["metadata_json"] == '{"args": {"path": "/tmp/file.txt"}}'

    async def test_get_by_frame_ordered_by_seq(
        self, message_repo, session_repo
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
        frame_id = str(uuid.uuid4())

        # Append messages out of order by seq
        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=3,
            role="user",
            content="Third",
            dedupe_key=f"user:{turn_id}:3",
        )
        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=1,
            role="user",
            content="First",
            dedupe_key=f"user:{turn_id}:1",
        )
        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=2,
            role="assistant",
            content="Second",
            dedupe_key=f"assistant:{frame_id}:2",
        )

        messages = await message_repo.get_by_frame(frame_id)
        assert len(messages) == 3
        assert messages[0]["seq"] == 1
        assert messages[1]["seq"] == 2
        assert messages[2]["seq"] == 3

    async def test_get_by_session_ordered_by_seq(
        self, message_repo, session_repo
    ) -> None:
        """get_by_session(): ordered by seq."""
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
        frame1_id = str(uuid.uuid4())
        frame2_id = str(uuid.uuid4())

        # Append messages across frames
        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame1_id,
            seq=2,
            role="assistant",
            content="Second",
            dedupe_key=f"assistant:{frame1_id}:2",
        )
        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame1_id,
            seq=1,
            role="user",
            content="First",
            dedupe_key=f"user:{turn_id}:1",
        )
        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame2_id,
            seq=3,
            role="user",
            content="Third",
            dedupe_key=f"user:{turn_id}:2",
        )

        messages = await message_repo.get_by_session(session_id)
        assert len(messages) == 3
        assert messages[0]["seq"] == 1
        assert messages[1]["seq"] == 2
        assert messages[2]["seq"] == 3

    async def test_get_by_range_inclusive(self, message_repo, session_repo) -> None:
        """get_by_range(): start_seq <= seq <= end_seq."""
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

        for i in range(1, 6):
            await message_repo.append(
                session_id=session_id,
                turn_id=turn_id,
                frame_id=frame_id,
                seq=i,
                role="user",
                content=f"Message {i}",
                dedupe_key=f"user:{turn_id}:{i}",
            )

        messages = await message_repo.get_by_range(session_id, start_seq=2, end_seq=4)
        assert len(messages) == 3
        assert messages[0]["seq"] == 2
        assert messages[1]["seq"] == 3
        assert messages[2]["seq"] == 4

    async def test_delete_by_frame_removes_messages(
        self, message_repo, session_repo
    ) -> None:
        """delete_by_frame(): removes messages for frame."""
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
        frame1_id = str(uuid.uuid4())
        frame2_id = str(uuid.uuid4())

        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame1_id,
            seq=1,
            role="user",
            content="Frame 1 message",
            dedupe_key=f"user:{turn_id}:1",
        )
        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame2_id,
            seq=2,
            role="user",
            content="Frame 2 message",
            dedupe_key=f"user:{turn_id}:2",
        )

        # Delete messages for frame1
        await message_repo.delete_by_frame(frame1_id)

        messages = await message_repo.get_by_session(session_id)
        assert len(messages) == 1
        assert messages[0]["frame_id"] == frame2_id

    async def test_auto_fills_created_at(self, message_repo, session_repo) -> None:
        """append(): auto-fills created_at_ms when None."""
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

        before = int(datetime.now(UTC).timestamp() * 1000)
        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="user",
            content="Hello",
            dedupe_key=f"user:{turn_id}",
        )
        after = int(datetime.now(UTC).timestamp() * 1000)

        messages = await message_repo.get_by_session(session_id)
        assert len(messages) == 1
        assert messages[0]["created_at"] is not None
        assert before <= messages[0]["created_at"] <= after

    async def test_explicit_created_at(self, message_repo, session_repo) -> None:
        """append(): respects explicit created_at_ms when provided."""
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
        created_at_ms = 1234567890

        await message_repo.append(
            session_id=session_id,
            turn_id=turn_id,
            frame_id=frame_id,
            seq=None,
            role="user",
            content="Hello",
            created_at_ms=created_at_ms,
            dedupe_key=f"user:{turn_id}",
        )

        messages = await message_repo.get_by_session(session_id)
        assert len(messages) == 1
        assert messages[0]["created_at"] == created_at_ms
