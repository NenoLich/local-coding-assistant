"""Template repository unit tests."""

from __future__ import annotations


class TestTemplateRepo:
    async def test_register_deduplicates_by_content_hash(self, template_repo) -> None:
        """register(): deduplicates by content_hash, returns existing ID."""
        template_id_1 = await template_repo.register(
            path="templates/read.md",
            content="Read file template",
            content_hash="abc123",
            version="1.0.0",
        )

        # Register same template again - should return existing ID
        template_id_2 = await template_repo.register(
            path="templates/read.md",
            content="Read file template",
            content_hash="abc123",
            version="1.0.0",
        )

        assert template_id_1 == template_id_2

    async def test_register_returns_new_id_for_different_content(
        self, template_repo
    ) -> None:
        """register(): returns new ID when content differs."""
        template_id_1 = await template_repo.register(
            path="templates/read.md",
            content="Read file template",
            content_hash="abc123",
            version="1.0.0",
        )

        template_id_2 = await template_repo.register(
            path="templates/read.md",
            content="Read file template",
            content_hash="def456",  # Different hash
            version="1.0.0",
        )

        assert template_id_1 != template_id_2

    async def test_register_auto_fills_created_at(self, template_repo) -> None:
        """register(): auto-fills created_at_ms when None."""
        from datetime import UTC, datetime

        before = int(datetime.now(UTC).timestamp() * 1000)
        template_id = await template_repo.register(
            path="templates/read.md",
            content="Read file template",
            content_hash="abc123",
            created_at_ms=None,
        )
        after = int(datetime.now(UTC).timestamp() * 1000)

        template = await template_repo.get(template_id)
        assert template is not None
        assert template["created_at"] is not None
        assert before <= template["created_at"] <= after

    async def test_get_returns_template_dict_by_id(self, template_repo) -> None:
        """get(): returns template dict by ID."""
        template_id = await template_repo.register(
            path="templates/read.md",
            content="Read file template",
            content_hash="abc123",
            version="1.0.0",
        )

        template = await template_repo.get(template_id)
        assert template is not None
        assert template["id"] == template_id
        assert template["path"] == "templates/read.md"
        assert template["content"] == "Read file template"
        assert template["content_hash"] == "abc123"
        assert template["version"] == "1.0.0"

    async def test_get_returns_none_for_missing_template(self, template_repo) -> None:
        template = await template_repo.get(99999)
        assert template is None

    async def test_get_existing_ids_filters_existing(self, template_repo) -> None:
        """get_existing_ids(): IN clause."""
        template_id_1 = await template_repo.register(
            path="templates/read.md",
            content="Read file template",
            content_hash="abc123",
        )
        template_id_2 = await template_repo.register(
            path="templates/write.md",
            content="Write file template",
            content_hash="def456",
        )

        existing = await template_repo.get_existing_ids(
            [template_id_1, template_id_2, 99999]
        )
        assert set(existing) == {template_id_1, template_id_2}

    async def test_get_existing_ids_empty_list(self, template_repo) -> None:
        """get_existing_ids(): returns empty list for empty input."""
        existing = await template_repo.get_existing_ids([])
        assert existing == []

    async def test_get_by_path_returns_template(self, template_repo) -> None:
        """get_by_path(): returns template by path."""
        await template_repo.register(
            path="templates/read.md",
            content="Read file template",
            content_hash="abc123",
            version="1.0.0",
        )

        template = await template_repo.get_by_path("templates/read.md")
        assert template is not None
        assert template["path"] == "templates/read.md"
        assert template["content"] == "Read file template"
        assert template["content_hash"] == "abc123"

    async def test_get_by_path_returns_none_for_missing_path(
        self, template_repo
    ) -> None:
        template = await template_repo.get_by_path("nonexistent.md")
        assert template is None

    async def test_list_all_ordered_by_path(self, template_repo) -> None:
        """list_all(): ordered by path."""
        await template_repo.register(
            path="templates/write.md",
            content="Write file template",
            content_hash="def456",
        )
        await template_repo.register(
            path="templates/read.md",
            content="Read file template",
            content_hash="abc123",
        )
        await template_repo.register(
            path="templates/delete.md",
            content="Delete file template",
            content_hash="ghi789",
        )

        templates = await template_repo.list_all()
        assert len(templates) == 3
        assert templates[0]["path"] == "templates/delete.md"
        assert templates[1]["path"] == "templates/read.md"
        assert templates[2]["path"] == "templates/write.md"

    async def test_get_session_templates_joins_session_templates(
        self, template_repo, session_repo
    ) -> None:
        """get_session_templates(): JOIN session_templates."""
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

        template_id_1 = await template_repo.register(
            path="templates/read.md",
            content="Read file template",
            content_hash="abc123",
        )
        template_id_2 = await template_repo.register(
            path="templates/write.md",
            content="Write file template",
            content_hash="def456",
        )

        # Associate templates with session
        await template_repo._conn.execute(
            "INSERT INTO session_templates (session_id, template_id) VALUES (?, ?)",
            (session_id, template_id_1),
        )
        await template_repo._conn.execute(
            "INSERT INTO session_templates (session_id, template_id) VALUES (?, ?)",
            (session_id, template_id_2),
        )

        templates = await template_repo.get_session_templates(session_id)
        assert len(templates) == 2
        template_paths = {t["path"] for t in templates}
        assert template_paths == {"templates/read.md", "templates/write.md"}

    async def test_get_session_templates_ordered_by_path(
        self, template_repo, session_repo
    ) -> None:
        """get_session_templates(): ordered by path."""
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

        await template_repo.register(
            path="templates/write.md", content="Write", content_hash="def456"
        )
        await template_repo.register(
            path="templates/read.md", content="Read", content_hash="abc123"
        )

        # Get template IDs
        templates = await template_repo.list_all()
        for template in templates:
            await template_repo._conn.execute(
                "INSERT INTO session_templates (session_id, template_id) VALUES (?, ?)",
                (session_id, template["id"]),
            )

        session_templates = await template_repo.get_session_templates(session_id)
        assert len(session_templates) == 2
        assert session_templates[0]["path"] == "templates/read.md"
        assert session_templates[1]["path"] == "templates/write.md"

    async def test_get_many_batch_fetch_by_ids(self, template_repo) -> None:
        """get_many(): batch fetch by IDs."""
        template_id_1 = await template_repo.register(
            path="templates/read.md",
            content="Read file template",
            content_hash="abc123",
        )
        template_id_2 = await template_repo.register(
            path="templates/write.md",
            content="Write file template",
            content_hash="def456",
        )
        template_id_3 = await template_repo.register(
            path="templates/delete.md",
            content="Delete file template",
            content_hash="ghi789",
        )

        templates = await template_repo.get_many([template_id_1, template_id_3])
        assert len(templates) == 2
        template_paths = {t["path"] for t in templates}
        assert template_paths == {"templates/read.md", "templates/delete.md"}

    async def test_get_many_empty_list(self, template_repo) -> None:
        """get_many(): returns empty list for empty input."""
        templates = await template_repo.get_many([])
        assert templates == []
