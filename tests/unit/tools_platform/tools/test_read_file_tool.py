"""Unit tests for the read_file tool.

Covers the test cases defined in plan_docs/2026_09/tools_platform_tests.md
(section test_read_file_tool.py):

- full read with line numbers
- offset/limit line range
- limit-based pagination
- next_offset resume
- internal 256KB byte limit and truncation
- total_lines on full read
- file observation for conflict detection
- validation errors (missing file, directory, outside workspace)
- LLM output excludes file_observation
- ToolDefinition output_schema uses model_llm_schema
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_coding_assistant.tools_platform.exceptions import (
    ToolPathViolationError,
    ToolValidationError,
)
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.tools.read_file import (
    ReadFileInput,
    ReadFileOutput,
    read_file_impl,
    register_read_file,
)


def _write_lines(workspace: Path, name: str, count: int) -> Path:
    file_path = workspace / name
    file_path.write_text(
        "".join(f"line {i}\n" for i in range(1, count + 1)), newline=""
    )
    return file_path


class TestReadFileContent:
    @pytest.mark.asyncio
    async def test_read_full_file_with_line_numbers(
        self, backend, temp_workspace: Path
    ) -> None:
        _write_lines(temp_workspace, "test.py", 3)
        result = await read_file_impl(backend, ReadFileInput(path="test.py"))
        assert isinstance(result, ReadFileOutput)
        assert result.content == "1\tline 1\n2\tline 2\n3\tline 3\n"
        assert result.total_lines == 3
        assert result.truncated is False
        assert result.next_offset is None

    @pytest.mark.asyncio
    async def test_read_offset_limit_returns_lines_10_to_19(
        self, backend, temp_workspace: Path
    ) -> None:
        _write_lines(temp_workspace, "test.py", 25)
        result = await read_file_impl(
            backend, ReadFileInput(path="test.py", offset=10, limit=10)
        )
        assert isinstance(result, ReadFileOutput)
        lines = result.content.splitlines()
        assert lines[0] == "10\tline 10"
        assert lines[-1] == "19\tline 19"
        assert not any(line.startswith("9\t") for line in lines)
        assert not any(line.startswith("20\t") for line in lines)
        assert result.total_lines is None

    @pytest.mark.asyncio
    async def test_read_limit_50_returns_first_50_lines(
        self, backend, temp_workspace: Path
    ) -> None:
        _write_lines(temp_workspace, "test.py", 60)
        result = await read_file_impl(backend, ReadFileInput(path="test.py", limit=50))
        assert isinstance(result, ReadFileOutput)
        lines = result.content.splitlines()
        assert len(lines) == 50
        assert lines[0] == "1\tline 1"
        assert lines[-1] == "50\tline 50"
        assert result.next_offset == 51
        assert result.truncated is False

    @pytest.mark.asyncio
    async def test_read_next_offset_resumes_from_position(
        self, backend, temp_workspace: Path
    ) -> None:
        # Large enough to exceed the internal 256KB byte limit so next_offset is set.
        content = "".join(
            f"long line number {i} with some padding\n" for i in range(40000)
        )
        (temp_workspace / "big.py").write_text(content)

        first = await read_file_impl(backend, ReadFileInput(path="big.py"))
        assert first.truncated is True
        assert first.next_offset is not None

        second = await read_file_impl(
            backend, ReadFileInput(path="big.py", offset=first.next_offset)
        )
        assert second.content.splitlines()[0].startswith(f"{first.next_offset}\t")

    @pytest.mark.asyncio
    async def test_read_respects_256kb_byte_limit_truncates(
        self, backend, temp_workspace: Path
    ) -> None:
        content = "".join(
            f"long line number {i} with some padding\n" for i in range(40000)
        )
        (temp_workspace / "big.py").write_text(content)

        result = await read_file_impl(backend, ReadFileInput(path="big.py"))
        assert isinstance(result, ReadFileOutput)
        assert result.truncated is True
        assert result.next_offset is not None
        assert result.next_offset >= 1

    @pytest.mark.asyncio
    async def test_read_full_file_returns_total_lines(
        self, backend, temp_workspace: Path
    ) -> None:
        _write_lines(temp_workspace, "test.py", 17)
        result = await read_file_impl(
            backend, ReadFileInput(path="test.py", offset=0, limit=None)
        )
        assert isinstance(result, ReadFileOutput)
        assert result.total_lines == 17

    @pytest.mark.asyncio
    async def test_read_returns_file_observation_internal_not_llm(
        self, backend, temp_workspace: Path
    ) -> None:
        _write_lines(temp_workspace, "test.py", 3)
        result = await read_file_impl(backend, ReadFileInput(path="test.py"))
        assert isinstance(result, ReadFileOutput)
        assert result.file_observation is not None
        llm_dict = result.to_llm_dict()
        assert "file_observation" not in llm_dict
        assert set(llm_dict.keys()) == {
            "content",
            "total_lines",
            "next_offset",
            "truncated",
        }

    @pytest.mark.asyncio
    async def test_read_nonexistent_file_raises_tool_validation_error(
        self, backend, temp_workspace: Path
    ) -> None:
        with pytest.raises(ToolValidationError):
            await read_file_impl(backend, ReadFileInput(path="does_not_exist.py"))

    @pytest.mark.asyncio
    async def test_read_directory_path_raises_tool_validation_error(
        self, backend, temp_workspace: Path
    ) -> None:
        (temp_workspace / "subdir").mkdir()
        with pytest.raises(ToolValidationError):
            await read_file_impl(backend, ReadFileInput(path="subdir"))

    @pytest.mark.asyncio
    async def test_read_outside_workspace_raises_tool_path_violation_error(
        self, backend, temp_workspace: Path
    ) -> None:
        with pytest.raises(ToolPathViolationError):
            await read_file_impl(backend, ReadFileInput(path="/outside/foo.py"))


class TestReadFileSchema:
    def test_tool_definition_output_schema_uses_model_llm_schema(self) -> None:
        registry = ToolRegistry()
        register_read_file(registry)
        definition = registry.get_tool("read_file")
        assert definition is not None
        schema = definition.output_schema
        assert schema == ReadFileOutput.model_llm_schema()
        assert "file_observation" not in str(schema)
        assert "content" in str(schema)
        assert "total_lines" in str(schema)
        assert "next_offset" in str(schema)
        assert "truncated" in str(schema)
