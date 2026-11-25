"""Unit tests for the delete_file tool.

Covers the test cases defined in plan_docs/2026_09/tools_platform_tests.md
(section test_delete_file_tool.py):

- removes existing file (success=true, path, error=None)
- fails if file does not exist (ToolPathViolationError)
- transaction commit persists deletion
- transaction rollback restores file from snapshot
- fails with ToolPathViolationError for path outside workspace
- supports recursive=true for directories
- supports dry_run=true returning would_delete list without deleting
- LLM output exposes only success, path, error, would_delete
- ToolDefinition output_schema uses model_llm_schema
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_coding_assistant.tools_platform.exceptions import ToolPathViolationError
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.tools.delete_file import (
    DeleteFileInput,
    DeleteFileOutput,
    delete_file_impl,
    register_delete_file,
)


def _write(workspace: Path, name: str, content: str) -> Path:
    file_path = workspace / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, newline="")
    return file_path


class TestDeleteFileContent:
    @pytest.mark.asyncio
    async def test_delete_existing_file_succeeds(
        self, backend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "delete_me.py", "content to delete\n")
        result = await delete_file_impl(
            backend,
            DeleteFileInput(path="delete_me.py"),
        )
        assert isinstance(result, DeleteFileOutput)
        assert result.success is True
        assert result.path == "delete_me.py"
        assert result.error is None
        assert result.would_delete is None
        assert not (temp_workspace / "delete_me.py").exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_file_raises(
        self, backend, temp_workspace: Path
    ) -> None:
        with pytest.raises(ToolPathViolationError):
            await delete_file_impl(
                backend,
                DeleteFileInput(path="does_not_exist.py"),
            )


class TestDeleteFileTransaction:
    @pytest.mark.asyncio
    async def test_commit_persists_deletion(
        self, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "commit_delete.py", "commit delete content\n")
        workspace_manager.begin_transaction()
        workspace_manager.observe_file("commit_delete.py")
        workspace_manager.atomic_delete("commit_delete.py")
        workspace_manager.commit_transaction()
        assert not (temp_workspace / "commit_delete.py").exists()

    @pytest.mark.asyncio
    async def test_rollback_restores_file_from_snapshot(
        self, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "rollback_delete.py", "rollback delete content\n")
        workspace_manager.begin_transaction()
        workspace_manager.observe_file("rollback_delete.py")
        workspace_manager.atomic_delete("rollback_delete.py")
        workspace_manager.rollback_transaction()
        assert (temp_workspace / "rollback_delete.py").exists()
        assert (
            temp_workspace / "rollback_delete.py"
        ).read_text() == "rollback delete content\n"


class TestDeleteFileValidation:
    @pytest.mark.asyncio
    async def test_outside_workspace_raises_tool_path_violation_error(
        self, backend, temp_workspace: Path
    ) -> None:
        with pytest.raises(ToolPathViolationError):
            await delete_file_impl(
                backend,
                DeleteFileInput(path="/outside/foo.py"),
            )


class TestDeleteFileDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_returns_would_delete_without_deleting(
        self, backend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "dry_run.py", "dry run content\n")
        result = await delete_file_impl(
            backend,
            DeleteFileInput(path="dry_run.py", dry_run=True),
        )
        assert isinstance(result, DeleteFileOutput)
        assert result.success is True
        assert result.error is None
        assert result.would_delete == ["dry_run.py"]
        # File must still exist after a dry run.
        assert (temp_workspace / "dry_run.py").exists()

    @pytest.mark.asyncio
    async def test_dry_run_directory_lists_contents(
        self, backend, temp_workspace: Path
    ) -> None:
        (temp_workspace / "sub").mkdir(parents=True, exist_ok=True)
        _write(temp_workspace, "sub/nested.py", "nested\n")
        result = await delete_file_impl(
            backend,
            DeleteFileInput(path="sub", recursive=True, dry_run=True),
        )
        assert isinstance(result, DeleteFileOutput)
        assert result.success is True
        assert result.would_delete == ["sub/nested.py"]
        # Directory must still exist after a dry run.
        assert (temp_workspace / "sub").exists()


class TestDeleteFileRecursive:
    @pytest.mark.asyncio
    async def test_recursive_deletes_directory(
        self, backend, temp_workspace: Path
    ) -> None:
        (temp_workspace / "sub").mkdir(parents=True, exist_ok=True)
        _write(temp_workspace, "sub/nested.py", "nested\n")
        result = await delete_file_impl(
            backend,
            DeleteFileInput(path="sub", recursive=True),
        )
        assert isinstance(result, DeleteFileOutput)
        assert result.success is True
        assert result.error is None
        assert not (temp_workspace / "sub").exists()


class TestDeleteFileLLMOutput:
    @pytest.mark.asyncio
    async def test_llm_output_exposes_only_public_fields(
        self, backend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "llm_delete.py", "content\n")
        result = await delete_file_impl(
            backend,
            DeleteFileInput(path="llm_delete.py"),
        )
        llm = result.to_llm_dict()
        assert set(llm.keys()) == {"success", "path", "error", "would_delete"}
        assert llm["success"] is True
        assert llm["path"] == "llm_delete.py"
        assert llm["error"] is None
        assert llm["would_delete"] is None


class TestDeleteFileSchema:
    def test_tool_definition_output_schema_uses_model_llm_schema(self) -> None:
        registry = ToolRegistry()
        register_delete_file(registry)
        definition = registry.get_tool("delete_file")
        assert definition is not None
        schema = definition.output_schema
        assert schema == DeleteFileOutput.model_llm_schema()
        schema_str = str(schema)
        assert "success" in schema_str
        assert "path" in schema_str
        assert "error" in schema_str
        assert "would_delete" in schema_str
        assert "before_hash" not in schema_str
        assert "before_size" not in schema_str
        assert "before_lines" not in schema_str
