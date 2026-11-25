"""Unit tests for the create_file tool.

Covers the test cases defined in plan_docs/2026_09/tools_platform_tests.md
(section test_create_file_tool.py):

- new path writes content (success=true, path, error=None)
- fails if file already exists unless overwrite=true
- creates parent directories if needed
- transaction commit persists file
- transaction rollback removes created file
- fails with ToolPathViolationError for path outside workspace
- LLM output excludes after_hash, after_size, after_lines
- ToolDefinition output_schema uses model_llm_schema
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from local_coding_assistant.tools_platform.exceptions import ToolPathViolationError
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.tools.create_file import (
    CreateFileInput,
    CreateFileOutput,
    create_file_impl,
    register_create_file,
)


def _write(workspace: Path, name: str, content: str) -> Path:
    file_path = workspace / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, newline="")
    return file_path


def _hash_of(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class TestCreateFileContent:
    @pytest.mark.asyncio
    async def test_new_file_writes_content(self, backend, temp_workspace: Path) -> None:
        result = await create_file_impl(
            backend,
            CreateFileInput(path="new.py", content="hello world\n"),
        )
        assert isinstance(result, CreateFileOutput)
        assert result.success is True
        assert result.path == "new.py"
        assert result.error is None
        assert result.after_hash == _hash_of("hello world\n")
        assert result.after_size == len(b"hello world\n")
        assert result.after_lines == 1
        assert (temp_workspace / "new.py").read_text() == "hello world\n"

    @pytest.mark.asyncio
    async def test_file_already_exists_fails_without_overwrite(
        self, backend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "existing.py", "original\n")
        with pytest.raises(ToolPathViolationError):
            await create_file_impl(
                backend,
                CreateFileInput(path="existing.py", content="new\n", overwrite=False),
            )
        # File unchanged.
        assert (temp_workspace / "existing.py").read_text() == "original\n"

    @pytest.mark.asyncio
    async def test_overwrite_true_replaces_existing_file(
        self, backend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "existing.py", "original\n")
        result = await create_file_impl(
            backend,
            CreateFileInput(path="existing.py", content="replaced\n", overwrite=True),
        )
        assert isinstance(result, CreateFileOutput)
        assert result.success is True
        assert result.error is None
        assert (temp_workspace / "existing.py").read_text() == "replaced\n"

    @pytest.mark.asyncio
    async def test_creates_parent_directories(
        self, backend, temp_workspace: Path
    ) -> None:
        result = await create_file_impl(
            backend,
            CreateFileInput(path="a/b/c/nested.py", content="nested\n"),
        )
        assert isinstance(result, CreateFileOutput)
        assert result.success is True
        assert result.error is None
        assert (temp_workspace / "a/b/c/nested.py").read_text() == "nested\n"


class TestCreateFileTransaction:
    @pytest.mark.asyncio
    async def test_commit_persists_file(
        self, workspace_manager, temp_workspace: Path
    ) -> None:
        workspace_manager.begin_transaction()
        workspace_manager.atomic_create("commit.py", content="commit content\n")
        workspace_manager.commit_transaction()
        assert (temp_workspace / "commit.py").read_text() == "commit content\n"

    @pytest.mark.asyncio
    async def test_rollback_removes_created_file(
        self, workspace_manager, temp_workspace: Path
    ) -> None:
        workspace_manager.begin_transaction()
        workspace_manager.atomic_create("rollback.py", content="rollback content\n")
        workspace_manager.rollback_transaction()
        assert not (temp_workspace / "rollback.py").exists()


class TestCreateFileValidation:
    @pytest.mark.asyncio
    async def test_outside_workspace_raises_tool_path_violation_error(
        self, backend, temp_workspace: Path
    ) -> None:
        with pytest.raises(ToolPathViolationError):
            await create_file_impl(
                backend,
                CreateFileInput(path="/outside/foo.py", content="x\n"),
            )


class TestCreateFileLLMOutput:
    @pytest.mark.asyncio
    async def test_llm_output_excludes_internal_fields(
        self, backend, temp_workspace: Path
    ) -> None:
        result = await create_file_impl(
            backend,
            CreateFileInput(path="llm.py", content="content\n"),
        )
        llm = result.to_llm_dict()
        assert set(llm.keys()) == {"success", "path", "error"}
        assert llm["success"] is True
        assert llm["path"] == "llm.py"
        assert llm["error"] is None
        assert "after_hash" not in llm
        assert "after_size" not in llm
        assert "after_lines" not in llm


class TestCreateFileSchema:
    def test_tool_definition_output_schema_uses_model_llm_schema(self) -> None:
        registry = ToolRegistry()
        register_create_file(registry)
        definition = registry.get_tool("create_file")
        assert definition is not None
        schema = definition.output_schema
        assert schema == CreateFileOutput.model_llm_schema()
        schema_str = str(schema)
        assert "success" in schema_str
        assert "path" in schema_str
        assert "error" in schema_str
        assert "after_hash" not in schema_str
        assert "after_size" not in schema_str
        assert "after_lines" not in schema_str
