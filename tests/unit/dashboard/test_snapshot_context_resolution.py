"""
Unit tests for snapshot_context_resolution: resolving and rendering stored
context snapshots for the dashboard plane. The execution store is mocked
with an AsyncMock, so no real database or filesystem access is required
(AGENTS.md discovery is disabled in fixtures unless explicitly tested).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_coding_assistant.dashboard.services.snapshot_context_resolution import (
    SnapshotContextResult,
    SnapshotRefs,
    build_prompt_context,
    render_context_blocks,
    render_snapshot_context,
    resolve_snapshot_refs,
)
from local_coding_assistant.runtime.context.profile import (
    ContextSnapshot,
    MemorySnapshotRef,
    RepoSnapshotRef,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.types import ExecutionMode
from local_coding_assistant.storage.repositories.repo_snapshot_repo import (
    RepoSnapshotData,
)


def _ms() -> int:
    return int(datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)


def _snapshot(**overrides) -> ContextSnapshot:
    data = dict(
        id="snap-1",
        turn_id="turn-1",
        content_hash="hash-1",
        created_at_ms=_ms(),
        tool_set_ref=ToolSetRef(tool_ids=[1, 2], version="tools-1"),
        template_set_ref=TemplateSetRef(template_ids=[3], version="templates-1"),
        repo_snapshot_ref=RepoSnapshotRef(
            target_project_root="/proj", repo_snapshot_hash="repo-hash"
        ),
        memory_snapshot_ref=MemorySnapshotRef(memory_ids=["mem-1"], version="mem-1"),
    )
    data.update(overrides)
    return ContextSnapshot(**data)


def _config(**overrides) -> RuntimeContextConfig:
    data = dict(
        llm_model="gpt-4",
        llm_provider="openai",
        execution_mode="classic_tools",
        tool_call_mode="classic",
        sandbox_enabled=False,
        cwd="/proj",
        platform="win32",
        agents_md_enabled=False,
    )
    data.update(overrides)
    return RuntimeContextConfig(**data)


def _refs(**overrides) -> SnapshotRefs:
    data = dict(
        tools=[],
        templates={},
        memories=[],
        target_project_root="/proj",
        project_info=None,
        repo_map_data=None,
        tool_set_version="",
        template_set_version="",
        repo_snapshot_hash="",
        memory_snapshot_version="",
    )
    data.update(overrides)
    return SnapshotRefs(**data)


class TestResolveSnapshotRefs:
    @pytest.mark.asyncio
    async def test_resolves_all_refs(self, mock_store):
        mock_store.get_tools_by_ids.return_value = [
            {
                "id": 1,
                "name": "echo",
                "description": "Echo tool",
                "schema_json": '{"type": "object"}',
            }
        ]
        mock_store.get_templates_by_ids.return_value = [
            {"id": 3, "path": "default.jinja2", "content": "hi {{ name }}"}
        ]
        mock_store.get_repo_snapshot_by_hash.return_value = RepoSnapshotData(
            target_project_root="/proj",
            repo_map_json='{"nodes": []}',
            project_info_json='{"name": "demo"}',
            repo_snapshot_hash="repo-hash",
        )
        mock_store.get_memories_by_ids.return_value = ["remember X"]

        refs, warnings = await resolve_snapshot_refs(mock_store, _snapshot())

        assert warnings == []
        assert refs.tools == [
            {
                "id": 1,
                "name": "echo",
                "description": "Echo tool",
                "schema_json": '{"type": "object"}',
            }
        ]
        assert refs.templates == {"default.jinja2": "hi {{ name }}"}
        assert refs.memories == ["remember X"]
        assert refs.target_project_root == "/proj"
        assert refs.project_info == {"name": "demo"}
        assert refs.repo_map_data == {"nodes": []}
        assert refs.tool_set_version == "tools-1"
        assert refs.template_set_version == "templates-1"
        assert refs.repo_snapshot_hash == "repo-hash"
        assert refs.memory_snapshot_version == "mem-1"

    @pytest.mark.asyncio
    async def test_empty_snapshot_makes_no_store_calls(self, mock_store):
        snap = _snapshot(
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
            repo_snapshot_ref=None,
            memory_snapshot_ref=None,
        )
        refs, warnings = await resolve_snapshot_refs(mock_store, snap)

        assert warnings == []
        assert refs.tools == []
        assert refs.templates == {}
        assert refs.memories == []
        assert refs.target_project_root == ""
        assert refs.project_info is None
        assert refs.repo_map_data is None
        # Ref versions are carried even when id lists are empty
        assert refs.tool_set_version == "1.0.0"
        assert refs.template_set_version == "1.0.0"
        assert refs.repo_snapshot_hash == ""
        assert refs.memory_snapshot_version == ""
        mock_store.get_tools_by_ids.assert_not_awaited()
        mock_store.get_templates_by_ids.assert_not_awaited()
        mock_store.get_repo_snapshot_by_hash.assert_not_awaited()
        mock_store.get_memories_by_ids.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_error_produces_warning_and_continues(self, mock_store):
        mock_store.get_tools_by_ids.side_effect = RuntimeError("store down")
        mock_store.get_templates_by_ids.return_value = []
        mock_store.get_repo_snapshot_by_hash.return_value = None
        mock_store.get_memories_by_ids.return_value = []

        refs, warnings = await resolve_snapshot_refs(mock_store, _snapshot())

        assert refs.tools == []
        assert refs.templates == {}
        assert refs.memories == []
        assert warnings == ["tools: store down"]

    @pytest.mark.asyncio
    async def test_invalid_repo_json_tolerated(self, mock_store):
        mock_store.get_tools_by_ids.return_value = []
        mock_store.get_templates_by_ids.return_value = []
        mock_store.get_repo_snapshot_by_hash.return_value = RepoSnapshotData(
            target_project_root="/proj",
            repo_map_json="{not json",
            project_info_json="{not json",
            repo_snapshot_hash="repo-hash",
        )
        mock_store.get_memories_by_ids.return_value = []

        refs, warnings = await resolve_snapshot_refs(mock_store, _snapshot())

        assert warnings == []
        assert refs.target_project_root == "/proj"
        assert refs.project_info is None
        assert refs.repo_map_data is None

    @pytest.mark.asyncio
    async def test_missing_repo_snapshot_yields_empty_fields(self, mock_store):
        mock_store.get_tools_by_ids.return_value = []
        mock_store.get_templates_by_ids.return_value = []
        mock_store.get_repo_snapshot_by_hash.return_value = None
        mock_store.get_memories_by_ids.return_value = []

        refs, warnings = await resolve_snapshot_refs(mock_store, _snapshot())

        assert warnings == []
        assert refs.target_project_root == ""
        assert refs.project_info is None
        assert refs.repo_map_data is None
        assert refs.repo_snapshot_hash == "repo-hash"


class TestBuildPromptContext:
    def test_maps_refs_and_config_to_prompt_context(self):
        refs = _refs(
            tools=[
                {
                    "name": "echo",
                    "description": "Echoes input",
                    "schema_json": json.dumps({"type": "object", "properties": {}}),
                }
            ],
            memories=["remember X"],
            project_info={"name": "demo"},
            repo_map_data={"nodes": []},
            tool_set_version="tools-1",
            template_set_version="templates-1",
            repo_snapshot_hash="repo-hash",
            memory_snapshot_version="mem-1",
        )
        ctx = build_prompt_context(refs, _config(), "session-1", _ms())

        assert ctx.session_id == "session-1"
        assert ctx.execution_mode == ExecutionMode.CLASSIC_TOOLS
        assert len(ctx.tools) == 1
        assert ctx.tools[0].name == "echo"
        assert ctx.tools[0].description == "Echoes input"
        assert ctx.tools[0].parameters == {"type": "object", "properties": {}}
        assert ctx.memories == ["remember X"]
        assert ctx.workspace_root == "/proj"
        assert ctx.cwd == "/proj"
        assert ctx.platform == "win32"
        assert ctx.model_id == "gpt-4"
        assert ctx.model_provider == "openai"
        assert ctx.metadata["context_snapshot_id"] == "repo-hash"
        assert ctx.metadata["tool_set_version"] == "tools-1"
        assert ctx.metadata["template_set_version"] == "templates-1"
        assert ctx.metadata["current_date"] == _ms()
        assert ctx.project_info == {"name": "demo"}
        assert ctx.repo_map_data == {"nodes": []}

    def test_invalid_schema_json_falls_back_to_empty(self):
        refs = _refs(
            tools=[{"name": "broken", "description": "d", "schema_json": "{not json"}]
        )
        ctx = build_prompt_context(refs, _config(), "session-1", None)

        assert ctx.tools[0].parameters == {}

    def test_classic_alias_and_invalid_execution_mode(self):
        refs = _refs()
        ctx = build_prompt_context(
            refs, _config(execution_mode="classic"), "session-1", None
        )
        assert ctx.execution_mode == ExecutionMode.CLASSIC_TOOLS
        ctx = build_prompt_context(
            refs, _config(execution_mode="bogus-mode"), "session-1", None
        )
        assert ctx.execution_mode == ExecutionMode.REASONING_ONLY

    def test_defaults_cwd_platform_and_now(self):
        refs = _refs()
        ctx = build_prompt_context(
            refs, _config(cwd="", platform=""), "session-1", None
        )
        assert ctx.cwd == os.getcwd()
        assert ctx.platform == sys.platform
        assert ctx.current_date > 0

    def test_agents_md_discovery_closest_first(self, tmp_path: Path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "AGENTS.md").write_text("root rules", encoding="utf-8")
        (tmp_path / "sub" / "AGENTS.md").write_text("sub rules", encoding="utf-8")
        config = _config(
            cwd=str(tmp_path / "sub"),
            agents_md_enabled=True,
            agents_md_filenames=["AGENTS.md"],
            agents_md_walk_up=True,
        )
        refs = _refs(target_project_root=str(tmp_path))
        ctx = build_prompt_context(refs, config, "session-1", None)

        contents = [content for _, content in ctx.agents_md_content]
        assert contents == ["sub rules", "root rules"]


class TestRenderContextBlocks:
    def test_renders_system_messages_and_tool_schemas(self):
        refs = _refs(
            tools=[
                {
                    "name": "echo",
                    "description": "Echo tool",
                    "schema_json": "{}",
                }
            ],
            templates={},
        )
        ctx = build_prompt_context(refs, _config(cwd="/different"), "session-1", _ms())

        system_messages, tool_schemas, reminder = render_context_blocks(ctx, refs)

        assert isinstance(system_messages, list)
        assert tool_schemas == [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo tool",
                    "parameters": {},
                },
            }
        ]
        assert isinstance(reminder, str)
        assert reminder
        assert "Working directory: /different" in reminder


class TestRenderSnapshotContext:
    @pytest.mark.asyncio
    async def test_happy_path(self, mock_store):
        mock_store.get_tools_by_ids.return_value = [
            {
                "id": 1,
                "name": "echo",
                "description": "Echo tool",
                "schema_json": "{}",
            }
        ]
        mock_store.get_templates_by_ids.return_value = []
        mock_store.get_repo_snapshot_by_hash.return_value = None
        mock_store.get_memories_by_ids.return_value = []

        result = await render_snapshot_context(
            mock_store, _snapshot(), _config(cwd="/different"), "session-1", _ms()
        )

        assert isinstance(result, SnapshotContextResult)
        assert result.resolved is True
        assert result.warnings == []
        assert result.tool_schemas == [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo tool",
                    "parameters": {},
                },
            }
        ]
        assert isinstance(result.system_reminder, str)
        assert result.system_reminder
        assert "Working directory: /different" in result.system_reminder

    @pytest.mark.asyncio
    async def test_tolerant_with_store_error(self, mock_store):
        mock_store.get_tools_by_ids.side_effect = RuntimeError("store down")
        mock_store.get_templates_by_ids.return_value = []
        mock_store.get_repo_snapshot_by_hash.return_value = None
        mock_store.get_memories_by_ids.return_value = []

        result = await render_snapshot_context(
            mock_store, _snapshot(), _config(cwd="/different"), "session-1", _ms()
        )

        assert result.resolved is False
        assert result.warnings == ["tools: store down"]
        assert result.tool_schemas == []
        assert isinstance(result.system_reminder, str)
        assert result.system_reminder
