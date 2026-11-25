"""Integration tests for ExecutionStore.update_context_artifacts.

Covers tool/template/repo-snapshot registration with content-hash
deduplication, version computation, memory snapshot versioning,
single-transaction atomicity, and rollback on error.
"""

from __future__ import annotations

import uuid

import pytest

from local_coding_assistant.repository.repo_map import RepoMapData
from local_coding_assistant.storage.execution_store import ExecutionStore


def _new_id() -> str:
    return f"t_{uuid.uuid4().hex[:12]}"


def _tool_spec(name: str = "read_file", description: str = "Read a file") -> dict:
    return {
        "name": name,
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        "description": description,
    }


def _template_spec(path: str = "tmpl.jinja2", content: str = "Hi {{ name }}") -> dict:
    return {path: content}


def _repo_map_data() -> RepoMapData:
    return RepoMapData(
        project_name="test",
        language_distribution={"python": 1.0},
        total_symbols=5,
        grouped_symbols={},
    )


async def _call_artifacts(
    store: ExecutionStore,
    *,
    tool_specs: list | None = None,
    template_specs: list | None = None,
    repo_map_data: object | None = None,
    project_info: object | None = None,
    target_project_root: str = "/root",
    memories: list | None = None,
):
    return await store.update_context_artifacts(
        tool_specs=tool_specs or [],
        template_specs=template_specs or [],
        repo_map_data=repo_map_data,
        project_info=project_info,
        target_project_root=target_project_root,
        memories=memories or [],
    )


class TestUpdateContextArtifactsTools:
    async def test_registers_multiple_tools_returns_ids_and_version(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        result = await _call_artifacts(
            store,
            tool_specs=[
                _tool_spec("read_file", "Read a file"),
                _tool_spec("write_file", "Write a file"),
                _tool_spec("list_dir", "List a directory"),
            ],
        )
        assert len(result["tools_with_ids"]) == 3
        assert (
            len(set(tool_with_id["id"] for tool_with_id in result["tools_with_ids"]))
            == 3
        )
        assert len(result["tool_set_version"]) == 16
        for tool_with_id in result["tools_with_ids"]:
            assert await store.get_tool_by_id(tool_with_id["id"]) is not None

    async def test_deduplicates_tools_by_content_hash(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        spec = _tool_spec("read_file", "Read a file")
        r1 = await _call_artifacts(store, tool_specs=[spec])
        r2 = await _call_artifacts(store, tool_specs=[spec])
        assert r1["tools_with_ids"] == r2["tools_with_ids"]
        assert len(r1["tools_with_ids"]) == 1
        assert len(await store.tool_repo.list_all()) == 1

    async def test_tool_set_version_deterministic_for_same_inputs(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        specs = [
            _tool_spec("read_file", "Read a file"),
            _tool_spec("write_file", "Write a file"),
        ]
        r1 = await _call_artifacts(store, tool_specs=specs)
        r2 = await _call_artifacts(store, tool_specs=specs)
        assert r1["tool_set_version"] == r2["tool_set_version"]

    async def test_tool_set_version_changes_when_inputs_change(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        r_a = await _call_artifacts(
            store, tool_specs=[_tool_spec("read_file", "Read a file")]
        )
        r_b = await _call_artifacts(
            store, tool_specs=[_tool_spec("write_file", "Write a file")]
        )
        assert r_a["tool_set_version"] != r_b["tool_set_version"]


class TestUpdateContextArtifactsTemplates:
    async def test_registers_multiple_templates_returns_ids_and_version(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        result = await _call_artifacts(
            store,
            template_specs=[
                _template_spec("a.jinja2", "Hi {{ name }}"),
                _template_spec("b.jinja2", "Bye {{ name }}"),
            ],
        )
        assert len(result["template_ids"]) == 2
        assert len(set(result["template_ids"])) == 2
        assert len(result["template_set_version"]) == 16
        for template_id in result["template_ids"]:
            assert await store.get_template_by_id(template_id) is not None

    async def test_deduplicates_templates_by_content_hash(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        result = await _call_artifacts(
            store,
            template_specs=[
                _template_spec("a.jinja2", "Hi {{ name }}"),
                _template_spec("b.jinja2", "Hi {{ name }}"),
            ],
        )
        assert len(set(result["template_ids"])) == 1
        assert len(await store.template_repo.list_all()) == 1

    async def test_template_set_version_deterministic_for_same_inputs(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        specs = [_template_spec("a.jinja2", "Hi {{ name }}")]
        r1 = await _call_artifacts(store, template_specs=specs)
        r2 = await _call_artifacts(store, template_specs=specs)
        assert r1["template_set_version"] == r2["template_set_version"]

    async def test_template_set_version_changes_when_inputs_change(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        r_a = await _call_artifacts(
            store, template_specs=[_template_spec("a.jinja2", "Hi")]
        )
        r_b = await _call_artifacts(
            store, template_specs=[_template_spec("a.jinja2", "Bye")]
        )
        assert r_a["template_set_version"] != r_b["template_set_version"]


class TestUpdateContextArtifactsRepoSnapshot:
    async def test_registers_repo_snapshot_returns_hash(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        result = await _call_artifacts(
            store, repo_map_data=_repo_map_data(), project_info=None
        )
        assert result["repo_snapshot_hash"] != "no_repo"
        assert len(result["repo_snapshot_hash"]) == 16
        assert (
            await store.get_repo_snapshot_by_hash(result["repo_snapshot_hash"])
        ) is not None

    async def test_deduplicates_repo_snapshot_by_hash(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        repo_map = _repo_map_data()
        r1 = await _call_artifacts(store, repo_map_data=repo_map, project_info=None)
        r2 = await _call_artifacts(store, repo_map_data=repo_map, project_info=None)
        assert r1["repo_snapshot_hash"] == r2["repo_snapshot_hash"]

    async def test_returns_no_repo_when_no_repo_map_data(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        result = await _call_artifacts(store, repo_map_data=None, project_info=None)
        assert result["repo_snapshot_hash"] == "no_repo"


class TestUpdateContextArtifactsMemory:
    async def test_memory_snapshot_version_is_hash_of_sorted_memories(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        r1 = await _call_artifacts(store, memories=["b", "a"])
        r2 = await _call_artifacts(store, memories=["a", "b"])
        assert r1["memory_snapshot_version"] == r2["memory_snapshot_version"]
        assert r1["memory_snapshot_version"] != "empty"
        assert len(r1["memory_snapshot_version"]) == 16

    async def test_memory_snapshot_version_empty_when_no_memories(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        result = await _call_artifacts(store, memories=[])
        assert result["memory_snapshot_version"] == "empty"


class TestUpdateContextArtifactsTransaction:
    async def test_registers_all_artifact_kinds_in_one_transaction(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        result = await _call_artifacts(
            store,
            tool_specs=[_tool_spec("read_file", "Read a file")],
            template_specs=[_template_spec("a.jinja2", "Hi {{ name }}")],
            repo_map_data=_repo_map_data(),
            project_info=None,
            target_project_root="/root",
            memories=["mem_a", "mem_b"],
        )
        assert len(result["tools_with_ids"]) == 1
        assert len(result["template_ids"]) == 1
        assert result["repo_snapshot_hash"] != "no_repo"
        assert result["memory_snapshot_version"] != "empty"
        assert await store.get_tool_by_id(result["tools_with_ids"][0]["id"]) is not None
        assert await store.get_template_by_id(result["template_ids"][0]) is not None
        assert (
            await store.get_repo_snapshot_by_hash(result["repo_snapshot_hash"])
        ) is not None

    async def test_rollback_on_error_leaves_no_partial_state(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        with pytest.raises(KeyError):
            await _call_artifacts(
                store,
                tool_specs=[
                    _tool_spec("read_file", "Read a file"),
                    {"parameters": {"type": "object"}},
                ],
                template_specs=[_template_spec("a.jinja2", "Hi {{ name }}")],
                repo_map_data=_repo_map_data(),
                project_info=None,
                memories=["mem_a"],
            )
        cursor = await store._conn.execute("SELECT COUNT(*) FROM tools")
        assert (await cursor.fetchone())[0] == 0
        cursor = await store._conn.execute("SELECT COUNT(*) FROM templates")
        assert (await cursor.fetchone())[0] == 0
        cursor = await store._conn.execute("SELECT COUNT(*) FROM repo_snapshots")
        assert (await cursor.fetchone())[0] == 0
