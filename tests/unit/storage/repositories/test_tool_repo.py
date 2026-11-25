"""Tool repository unit tests."""

from __future__ import annotations


class TestToolRepo:
    async def test_register_deduplicates_by_content_hash(self, tool_repo) -> None:
        """register(): deduplicates by content_hash (name|schema|risk|desc|version), returns existing ID."""
        tool_id_1 = await tool_repo.register(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
            description="Read a file",
            version="1.0.0",
        )

        # Register same tool again - should return existing ID
        tool_id_2 = await tool_repo.register(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
            description="Read a file",
            version="1.0.0",
        )

        assert tool_id_1 == tool_id_2

    async def test_register_returns_new_id_for_different_content(
        self, tool_repo
    ) -> None:
        """register(): returns new ID when content differs."""
        tool_id_1 = await tool_repo.register(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
            description="Read a file",
            version="1.0.0",
        )

        tool_id_2 = await tool_repo.register(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
            description="Read a file",  # Same but different version
            version="2.0.0",
        )

        assert tool_id_1 != tool_id_2

    async def test_register_auto_fills_created_at(self, tool_repo) -> None:
        """register(): auto-fills created_at_ms when None."""
        from datetime import UTC, datetime

        before = int(datetime.now(UTC).timestamp() * 1000)
        tool_id = await tool_repo.register(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
            created_at_ms=None,
        )
        after = int(datetime.now(UTC).timestamp() * 1000)

        tool = await tool_repo.get(tool_id)
        assert tool is not None
        assert tool["created_at"] is not None
        assert before <= tool["created_at"] <= after

    async def test_get_returns_tool_dict_by_id(self, tool_repo) -> None:
        """get(): returns tool dict by ID."""
        tool_id = await tool_repo.register(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
            description="Read a file",
            version="1.0.0",
        )

        tool = await tool_repo.get(tool_id)
        assert tool is not None
        assert tool["id"] == tool_id
        assert tool["name"] == "read_file"
        assert tool["schema_json"] == '{"type": "object"}'
        assert tool["risk_class"] == "safe"
        assert tool["description"] == "Read a file"
        assert tool["version"] == "1.0.0"

    async def test_get_returns_none_for_missing_tool(self, tool_repo) -> None:
        tool = await tool_repo.get(99999)
        assert tool is None

    async def test_get_existing_ids_filters_existing(self, tool_repo) -> None:
        """get_existing_ids(): IN clause, returns subset."""
        tool_id_1 = await tool_repo.register(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
        )
        tool_id_2 = await tool_repo.register(
            name="write_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
        )

        existing = await tool_repo.get_existing_ids([tool_id_1, tool_id_2, 99999])
        assert set(existing) == {tool_id_1, tool_id_2}

    async def test_get_existing_ids_empty_list(self, tool_repo) -> None:
        """get_existing_ids(): returns empty list for empty input."""
        existing = await tool_repo.get_existing_ids([])
        assert existing == []

    async def test_get_by_name_returns_tool(self, tool_repo) -> None:
        """get_by_name(): returns tool by name."""
        await tool_repo.register(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
            description="Read a file",
            version="1.0.0",
        )

        tool = await tool_repo.get_by_name("read_file")
        assert tool is not None
        assert tool["name"] == "read_file"
        assert tool["schema_json"] == '{"type": "object"}'

    async def test_get_by_name_returns_none_for_missing_name(self, tool_repo) -> None:
        tool = await tool_repo.get_by_name("nonexistent")
        assert tool is None

    async def test_list_all_ordered_by_name(self, tool_repo) -> None:
        """list_all(): ordered by name."""
        await tool_repo.register(
            name="write_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
        )
        await tool_repo.register(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
        )
        await tool_repo.register(
            name="delete_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
        )

        tools = await tool_repo.list_all()
        assert len(tools) == 3
        assert tools[0]["name"] == "delete_file"
        assert tools[1]["name"] == "read_file"
        assert tools[2]["name"] == "write_file"

    async def test_get_session_tools_joins_session_tools(
        self, tool_repo, session_repo
    ) -> None:
        """get_session_tools(): JOIN session_tools."""
        import uuid

        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        tool_id_1 = await tool_repo.register(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
        )
        tool_id_2 = await tool_repo.register(
            name="write_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
        )

        # Associate tools with session
        await tool_repo._conn.execute(
            "INSERT INTO session_tools (session_id, tool_id) VALUES (?, ?)",
            (session_id, tool_id_1),
        )
        await tool_repo._conn.execute(
            "INSERT INTO session_tools (session_id, tool_id) VALUES (?, ?)",
            (session_id, tool_id_2),
        )

        tools = await tool_repo.get_session_tools(session_id)
        assert len(tools) == 2
        tool_names = {t["name"] for t in tools}
        assert tool_names == {"read_file", "write_file"}

    async def test_get_session_tools_ordered_by_name(
        self, tool_repo, session_repo
    ) -> None:
        """get_session_tools(): ordered by name."""
        import uuid

        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        await tool_repo.register(name="write_file", schema_json="{}", risk_class="safe")
        await tool_repo.register(name="read_file", schema_json="{}", risk_class="safe")

        # Get tool IDs
        tools = await tool_repo.list_all()
        for tool in tools:
            await tool_repo._conn.execute(
                "INSERT INTO session_tools (session_id, tool_id) VALUES (?, ?)",
                (session_id, tool["id"]),
            )

        session_tools = await tool_repo.get_session_tools(session_id)
        assert len(session_tools) == 2
        assert session_tools[0]["name"] == "read_file"
        assert session_tools[1]["name"] == "write_file"

    async def test_get_many_batch_fetch_by_ids(self, tool_repo) -> None:
        """get_many(): batch fetch by IDs."""
        tool_id_1 = await tool_repo.register(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
        )
        tool_id_2 = await tool_repo.register(
            name="write_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
        )
        tool_id_3 = await tool_repo.register(
            name="delete_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
        )

        tools = await tool_repo.get_many([tool_id_1, tool_id_3])
        assert len(tools) == 2
        tool_names = {t["name"] for t in tools}
        assert tool_names == {"read_file", "delete_file"}

    async def test_get_many_empty_list(self, tool_repo) -> None:
        """get_many(): returns empty list for empty input."""
        tools = await tool_repo.get_many([])
        assert tools == []

    async def test_register_handles_none_description(self, tool_repo) -> None:
        """register(): handles None description in content hash."""
        tool_id_1 = await tool_repo.register(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
            description=None,
            version="1.0.0",
        )

        tool_id_2 = await tool_repo.register(
            name="read_file",
            schema_json='{"type": "object"}',
            risk_class="safe",
            description=None,
            version="1.0.0",
        )

        assert tool_id_1 == tool_id_2
