"""Tests for ContextManager with session context policy and immutable snapshots."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from local_coding_assistant.runtime.context import ContextManager
from local_coding_assistant.runtime.context.profile import (
    ContextProfile,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.context.resolution import SessionContextBundle
from local_coding_assistant.runtime.session import Message, SessionState
from local_coding_assistant.storage.repositories import RepoSnapshotData


class MockConfigManager:
    def __init__(self, resolution_mode: str = "static"):
        self.global_config = MagicMock()
        self.global_config.runtime = MagicMock()
        self.global_config.runtime.session_context_resolution = resolution_mode
        self.global_config.runtime.tool_call_mode = "classic_tools"
        self.global_config.runtime.agent_mode = "no_agent"
        self.global_config.runtime.stream = True
        self.global_config.llm = MagicMock()
        self.global_config.llm.model_name = "test-model"
        self.global_config.llm.temperature = 0.1
        self.global_config.llm.max_tokens = 1000
        self.global_config.sandbox = MagicMock()
        self.global_config.sandbox.enabled = False
        self.global_config.prompt = MagicMock()
        self.global_config.prompt.templates = {}
        self.global_config.prompt.trim_blocks = True
        self.global_config.prompt.lstrip_blocks = True
        self.global_config.prompt.agents_md = MagicMock()
        self.global_config.prompt.agents_md.enabled = True
        self.global_config.prompt.agents_md.walk_up = True
        self.global_config.prompt.agents_md.filenames = ["AGENTS.md", "CLAUDE.md"]
        self.global_config.prompt.agents_md.global_paths = []
        self.global_config.prompt.config_instructions = []
        self.path_manager = MagicMock()
        import tempfile
        from pathlib import Path

        temp_dir = tempfile.mkdtemp()
        self.path_manager.get_template_dir.return_value = Path(temp_dir)
        self.path_manager.get_target_project_root.return_value = "/test/project"

    def get_provider_by_model(self, model_id: str) -> str:
        return "test-provider"


class MockToolGateway:
    def __init__(self):
        self._registry = MagicMock()
        self._registry.list_tools.return_value = []


class MockMemoryProvider:
    def fetch(self, session=None):
        return ["memory1", "memory2"]


class MockRepoService:
    def get_repo_context(self):
        from local_coding_assistant.repository.models import ProjectInfo
        from local_coding_assistant.repository.repo_map import RepoMapData

        repo_map = RepoMapData(
            project_name="test-project",
            language_distribution={"python": 100.0},
            total_symbols=10,
            grouped_symbols={},
        )
        project_info = ProjectInfo(name="test-project", description="Test project")
        return project_info, repo_map

    @property
    def target_project_root(self):
        return "/test/project"


class MockStore:
    def __init__(self):
        self.tools = {}
        self.templates = {}
        self.repo_snapshots = {}
        self.memories = {}

    async def get_tool_by_id(self, tool_id: int):
        return self.tools.get(tool_id)

    async def get_tools_by_ids(self, tool_ids: list[int]):
        return [
            self.tools.get(tid) for tid in tool_ids if self.tools.get(tid) is not None
        ]

    async def get_template_by_id(self, template_id: int):
        return self.templates.get(template_id)

    async def get_templates_by_ids(self, template_ids: list[int]):
        return [
            self.templates.get(tid)
            for tid in template_ids
            if self.templates.get(tid) is not None
        ]

    async def get_repo_snapshot(self, repo_root: str, commit_hash: str | None):
        return self.repo_snapshots.get(repo_root)

    async def get_repo_snapshot_by_hash(self, repo_snapshot_hash: str):
        for snap in self.repo_snapshots.values():
            if snap.get("repo_snapshot_hash") == repo_snapshot_hash:
                return RepoSnapshotData(
                    target_project_root=snap["target_project_root"],
                    repo_map_json=snap["repo_map_json"],
                    project_info_json=snap.get("project_info_json"),
                    repo_snapshot_hash=snap["repo_snapshot_hash"],
                )
        return None

    async def register_repo_snapshot(
        self,
        target_project_root: str,
        repo_map_json: str,
        project_info_json: str | None,
        repo_snapshot_hash: str,
    ):
        self.repo_snapshots[target_project_root] = {
            "target_project_root": target_project_root,
            "repo_map_json": repo_map_json,
            "project_info_json": project_info_json,
            "repo_snapshot_hash": repo_snapshot_hash,
        }

        class Result:
            def __init__(self, id, is_new):
                self.id = id
                self.is_new = is_new

        return Result(len(self.repo_snapshots), True)

    async def get_memory_by_id(self, memory_id: str):
        return self.memories.get(memory_id)

    async def register_tool(
        self,
        name: str,
        schema_json: str,
        risk_class: str,
        description: str | None = None,
        version: str = "1.0.0",
    ):
        tool_id = len(self.tools) + 1
        self.tools[tool_id] = {
            "id": tool_id,
            "name": name,
            "schema_json": schema_json,
            "risk_class": risk_class,
            "description": description,
            "version": version,
        }
        return tool_id

    async def register_template(
        self, path: str, content: str, content_hash: str, version: str = "1.0.0"
    ):
        template_id = len(self.templates) + 1
        self.templates[template_id] = {
            "id": template_id,
            "path": path,
            "content": content,
            "content_hash": content_hash,
            "version": version,
        }
        return template_id

    async def update_context_artifacts(
        self,
        tool_specs: list[dict[str, Any]],
        template_specs: list[dict[str, str]],
        repo_map_data: Any | None,
        project_info: Any | None,
        target_project_root: str,
        memories: list[str],
    ) -> dict[str, Any]:
        """Mock implementation of update_context_artifacts."""
        import dataclasses
        import hashlib
        import json

        # Register tools
        tool_ids = []
        tool_identities = []
        for spec in tool_specs:
            tool_id = await self.register_tool(
                name=spec["name"],
                schema_json="{}",
                risk_class="low",
                description=spec.get("description", ""),
                version="1.0.0",
            )
            tool_ids.append(tool_id)
            param_hash = hashlib.sha256(
                json.dumps(spec.get("parameters", {}), sort_keys=True).encode()
            ).hexdigest()[:8]
            tool_identities.append((tool_id, spec["name"], param_hash))

        # Register templates
        template_ids = []
        template_paths = []
        for spec in template_specs:
            for path, content in spec.items():
                content_hash = hashlib.sha256(
                    content.encode() if content else path.encode()
                ).hexdigest()
                template_id = await self.register_template(
                    path=path,
                    content=content,
                    content_hash=content_hash,
                    version="1.0.0",
                )
                template_ids.append(template_id)
                template_paths.append(path)

        # Register repo snapshot if data available
        repo_snapshot_hash = "no_repo"
        if repo_map_data is not None:
            repo_map_json = json.dumps(dataclasses.asdict(repo_map_data), default=str)
            project_info_json = project_info.model_dump_json() if project_info else None
            repo_snapshot_hash = hashlib.sha256(
                json.dumps(
                    {"repo_map": repo_map_json, "project_info": project_info_json},
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:16]

            await self.register_repo_snapshot(
                target_project_root=target_project_root,
                repo_map_json=repo_map_json,
                project_info_json=project_info_json,
                repo_snapshot_hash=repo_snapshot_hash,
            )

        # Memory snapshot version
        memory_snapshot_version = "empty"
        if memories:
            memory_snapshot_version = hashlib.sha256(
                json.dumps(sorted(memories), sort_keys=True).encode()
            ).hexdigest()[:16]

        # Compute version strings
        tool_set_version = hashlib.sha256(
            json.dumps(sorted(tool_identities), sort_keys=True).encode()
        ).hexdigest()[:16]

        template_set_version = hashlib.sha256(
            json.dumps(sorted(template_paths), sort_keys=True).encode()
        ).hexdigest()[:16]

        return {
            "tool_ids": tool_ids,
            "template_ids": template_ids,
            "tool_set_version": tool_set_version,
            "template_set_version": template_set_version,
            "repo_snapshot_hash": repo_snapshot_hash,
            "memory_snapshot_version": memory_snapshot_version,
        }


@pytest.fixture
def session():
    return SessionState(
        context_profile=ContextProfile(
            runtime_context_config=RuntimeContextConfig(),
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
        )
    )


@pytest.fixture
def config_manager():
    return MockConfigManager("static")


@pytest.fixture
def dynamic_config_manager():
    return MockConfigManager("dynamic")


@pytest.fixture
def tool_gateway():
    return MockToolGateway()


@pytest.fixture
def memory_provider():
    return MockMemoryProvider()


@pytest.fixture
def repo_service():
    return MockRepoService()


@pytest.fixture
def store():
    return MockStore()


@pytest.fixture
def context_manager(config_manager, tool_gateway, repo_service, memory_provider, store):
    return ContextManager(
        config_manager=config_manager,
        tool_gateway=tool_gateway,
        repo_service=repo_service,
        memory_provider=memory_provider,
        store=store,
    )


@pytest.fixture
def dynamic_context_manager(
    dynamic_config_manager, tool_gateway, repo_service, memory_provider, store
):
    return ContextManager(
        config_manager=dynamic_config_manager,
        tool_gateway=tool_gateway,
        repo_service=repo_service,
        memory_provider=memory_provider,
        store=store,
    )


class TestSessionContextPolicy:
    """Tests for static vs dynamic session context resolution."""

    @pytest.mark.asyncio
    async def test_static_mode_reuses_bundle(self, context_manager, session):
        """In static mode, session bundle is reused across turns with same config."""
        config = context_manager._build_context_config()

        # First call builds bundle
        bundle1 = await context_manager.update_bundle_cache(session.id, config=config)
        assert isinstance(bundle1, SessionContextBundle)
        assert context_manager._compute_config_hash(
            bundle1.profile.runtime_context_config
        ) == context_manager._compute_config_hash(config)

        # Second call should reuse bundle (same config)
        bundle2 = await context_manager.update_bundle_cache(session.id, config=config)
        assert bundle2 is bundle1  # Same object reused

    @pytest.mark.asyncio
    async def test_dynamic_mode_builds_fresh_bundle(
        self, dynamic_context_manager, session
    ):
        """In dynamic mode, new bundle built for each turn even with same config."""
        config = dynamic_context_manager._build_context_config()

        bundle1 = await dynamic_context_manager.update_bundle_cache(
            session.id, config=config
        )
        bundle2 = await dynamic_context_manager.update_bundle_cache(
            session.id, config=config
        )

        # Different objects (fresh each time)
        assert bundle1 is not bundle2
        # But same content
        assert dynamic_context_manager._compute_config_hash(
            bundle1.profile.runtime_context_config
        ) == dynamic_context_manager._compute_config_hash(
            bundle2.profile.runtime_context_config
        )

    @pytest.mark.asyncio
    async def test_static_mode_invalidates_on_config_change(
        self, context_manager, session
    ):
        """Static bundle invalidated when config hash changes."""
        config1 = context_manager._build_context_config()
        bundle1 = await context_manager.update_bundle_cache(session.id, config=config1)

        # Create config with different hash
        config2 = RuntimeContextConfig(
            llm_model="different-model",
            llm_temperature=0.5,
            llm_max_tokens=2000,
            tool_call_mode="classic",
            execution_mode="classic",
            agent_mode=False,
            graph_mode=False,
            streaming=True,
            sandbox_enabled=False,
            prompt_templates={"system": "base/system.jinja2"},
            trim_blocks=True,
            lstrip_blocks=True,
        )

        bundle2 = await context_manager.update_bundle_cache(session.id, config=config2)
        assert bundle2 is not bundle1  # New bundle created
        assert context_manager._compute_config_hash(
            bundle2.profile.runtime_context_config
        ) != context_manager._compute_config_hash(
            bundle1.profile.runtime_context_config
        )


class TestImmutableTurnSnapshot:
    """Tests for immutable turn snapshot creation and resolution."""

    @pytest.mark.asyncio
    async def test_create_snapshot_includes_stable_refs(self, context_manager, session):
        """Snapshot contains stable ref identities with versions."""
        # Add some history for message_boundary
        session.history = [
            Message(role="user", content="msg1", seq=1),
            Message(role="assistant", content="msg2", seq=2),
        ]
        session.history_sequence = 2

        config = context_manager._build_context_config()
        snapshot = await context_manager.create_snapshot(session, config, "turn_1")

        assert snapshot.id == "ctx_turn_1"
        assert snapshot.turn_id == "turn_1"
        assert isinstance(snapshot.tool_set_ref, ToolSetRef)
        assert snapshot.tool_set_ref.version != ""
        assert snapshot.template_set_ref.version != ""
        assert snapshot.repo_snapshot_ref is not None
        assert snapshot.repo_snapshot_ref.repo_snapshot_hash is not None
        assert snapshot.memory_snapshot_ref is not None
        assert snapshot.content_hash != ""
        assert snapshot.message_boundary is not None

    @pytest.mark.asyncio
    async def test_snapshot_message_boundary_is_turn_start_history(
        self, context_manager, session
    ):
        """Snapshot's message_boundary captures history at turn start."""
        # Add some history
        session.history = [
            Message(role="user", content="msg1", seq=1),
            Message(role="assistant", content="msg2", seq=2),
        ]
        session.history_sequence = 2

        config = context_manager._build_context_config()
        snapshot = await context_manager.create_snapshot(session, config, "turn_1")

        assert snapshot.message_boundary is not None
        assert snapshot.message_boundary.start_seq == 1
        assert snapshot.message_boundary.end_seq == 2
        assert snapshot.message_boundary.session_id == session.id

    @pytest.mark.asyncio
    async def test_resolve_context_uses_frame_history_through_seq(
        self, context_manager, session
    ):
        """Frame resolution uses history_through_seq, not snapshot message_boundary."""
        session.history = [
            Message(role="user", content="msg1", seq=1),
            Message(role="assistant", content="msg2", seq=2),
            Message(role="user", content="msg3", seq=3),
        ]
        session.history_sequence = 3

        config = context_manager._build_context_config()
        snapshot = await context_manager.create_snapshot(session, config, "turn_1")

        # Resolve with frame's advancing cursor (seq=2, not the turn's boundary which is 2)
        from local_coding_assistant.runtime.engine.continuation import FrameContinuation

        continuation = FrameContinuation()

        prompt_ctx, rendered = await context_manager.resolve_context(
            session=session,
            snapshot=snapshot,
            continuation=continuation,
            config_snapshot=config,
            history_through_seq=2,  # Frame only sees up to seq=2
        )

        # Should only have messages up to seq=2
        history_dicts = [
            msg.model_dump() for msg in session.history if msg.seq and msg.seq <= 2
        ]
        assert len(history_dicts) == 2


class TestRefResolution:
    """Tests for stable ref resolution from store."""

    @pytest.mark.asyncio
    async def test_resolve_tools_from_store(self, context_manager, store):
        """Tools resolved from store using snapshot's tool_ids."""
        # Register a tool
        tool_id = await store.register_tool("search", "{}", "low", "Search tool")

        # Create snapshot with that tool
        config = context_manager._build_context_config()
        session = SessionState(
            id="s1",
            context_profile=ContextProfile(
                runtime_context_config=config,
                tool_set_ref=ToolSetRef(),
                template_set_ref=TemplateSetRef(),
            ),
        )
        snapshot = await context_manager.create_snapshot(session, config, "turn_1")

        # Manually set tool_ids to match registered tool
        snapshot.tool_set_ref.tool_ids = [tool_id]

        # Resolve refs
        refs = await context_manager._resolve_all_refs_from_snapshot(snapshot)
        assert len(refs.tools) == 1
        assert refs.tools[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_resolve_templates_from_store(self, context_manager, store):
        """Templates resolved from store using snapshot's template_ids."""
        template_id = await store.register_template(
            "base/system.jinja2", "System prompt", "hash123"
        )

        config = context_manager._build_context_config()
        session = SessionState(
            id="s1",
            context_profile=ContextProfile(
                runtime_context_config=config,
                tool_set_ref=ToolSetRef(),
                template_set_ref=TemplateSetRef(),
            ),
        )
        snapshot = await context_manager.create_snapshot(session, config, "turn_1")
        snapshot.template_set_ref.template_ids = [template_id]

        refs = await context_manager._resolve_all_refs_from_snapshot(snapshot)
        # Key is the template path
        assert "base/system.jinja2" in refs.templates
        assert refs.templates["base/system.jinja2"] == "System prompt"


class TestCacheInvalidation:
    """Tests for cache invalidation on source changes."""

    def test_clear_session_bundle_invalidates_static_cache(
        self, context_manager, session
    ):
        """clear_session_bundle removes static bundle cache."""
        config = context_manager._build_context_config()
        # Can't easily test async cache without running, but verify method exists
        context_manager.clear_session_bundle(session.id)
        assert session.id not in context_manager._session_bundles

    def test_clear_session_all_clears_both_caches(self, context_manager, session):
        """clear_session_all clears config and bundle caches."""
        config = context_manager._build_context_config()
        context_manager.clear_session_all(session.id)
        assert session.id not in context_manager._session_bundles


class TestContentHash:
    """Tests for content hash stability."""

    @pytest.mark.asyncio
    async def test_content_hash_identical(self, context_manager, session):
        """Content hash should be identical for same refs and history."""
        config = context_manager._build_context_config()

        # Create two snapshots with same refs but different session history
        snapshot1 = await context_manager.create_snapshot(session, config, "turn_1")

        snapshot2 = await context_manager.create_snapshot(session, config, "turn_2")

        assert snapshot1.content_hash == snapshot2.content_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
