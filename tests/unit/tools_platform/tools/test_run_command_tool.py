"""Unit tests for the run_command tool.

Covers the test cases defined in plan_docs/2026_09/tools_platform_tests.md
(section test_run_command_tool.py) that are testable at the tool-implementation
level:

- argv execution returns completed status with collected stdout/stderr
- cwd is forwarded to the backend (resolved within workspace)
- env parameter is not accepted / never forwarded to the backend
- handle_id, return_code, state and truncated are captured from the status chunk
- real command execution via HostBackend (cross-platform)
- LLM output excludes internal tracking fields (stdout_bytes, stderr_bytes,
  elapsed_seconds) via to_llm_dict
- ToolDefinition.output_schema uses model_llm_schema

Middleware-level behaviours (approval ASK/DENY for request_shell /
request_network, command pattern matching from tool_policies.yaml, timeouts,
cancellation) are exercised by the gateway / authorization / scheduler test
suites rather than here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from local_coding_assistant.tools_platform.contracts import (
    ProcessState,
    ToolOutputChunk,
)
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.tools.run_command import (
    RunCommandInput,
    RunCommandOutput,
    register_run_command,
    run_command_impl,
)

_WINDOWS = sys.platform == "win32"


def _status_chunk(
    handle_id: str = "handle-1", state: str = ProcessState.COMPLETED.value
) -> ToolOutputChunk:
    return ToolOutputChunk(
        stream_id=handle_id,
        sequence=-1,
        channel="status",
        payload={
            "handle_id": handle_id,
            "return_code": 0,
            "state": state,
            "truncated": False,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "stdout_lines": 0,
            "stderr_lines": 0,
        },
    )


def _stdout_chunk(
    text: str, stream_id: str = "handle-1", sequence: int = 0
) -> ToolOutputChunk:
    encoded = text.encode("utf-8")
    return ToolOutputChunk(
        stream_id=stream_id,
        sequence=sequence,
        channel="stdout",
        payload=text,
        cumulative_bytes=len(encoded),
    )


class TestRunCommandExecution:
    @pytest.mark.asyncio
    async def test_echo_returns_completed_with_stdout(
        self, command_backend, temp_workspace: Path
    ) -> None:
        backend = command_backend(
            [
                _stdout_chunk("hello\n"),
                _status_chunk("handle-1", ProcessState.COMPLETED.value),
            ]
        )

        result = await run_command_impl(
            backend, RunCommandInput(argv=["echo", "hello"])
        )

        assert isinstance(result, RunCommandOutput)
        assert result.state == ProcessState.COMPLETED.value
        assert result.return_code == 0
        assert result.handle_id == "handle-1"
        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.truncated is False

        # env must never be forwarded to the backend
        last_call = backend._calls[-1]
        assert last_call["env"] is None
        assert last_call["argv"] == ["echo", "hello"]
        assert last_call["cwd"] is None

    @pytest.mark.asyncio
    async def test_stderr_collected_separately(
        self, command_backend, temp_workspace: Path
    ) -> None:
        backend = command_backend(
            [
                _stdout_chunk("out\n"),
                ToolOutputChunk(
                    stream_id="h",
                    sequence=1,
                    channel="stderr",
                    payload="err\n",
                    cumulative_bytes=4,
                ),
                _status_chunk("h", ProcessState.COMPLETED.value),
            ]
        )

        result = await run_command_impl(backend, RunCommandInput(argv=["echo", "x"]))

        assert result.stdout == "out\n"
        assert result.stderr == "err\n"

    @pytest.mark.asyncio
    async def test_cwd_forwarded_to_backend(
        self, command_backend, temp_workspace: Path
    ) -> None:
        backend = command_backend([_status_chunk("h", ProcessState.COMPLETED.value)])

        await run_command_impl(
            backend, RunCommandInput(argv=["cat", "file.txt"], cwd="subdir")
        )

        last_call = backend._calls[-1]
        assert last_call["cwd"] == Path("subdir")

    @pytest.mark.asyncio
    async def test_truncated_flag_from_status_chunk(
        self, command_backend, temp_workspace: Path
    ) -> None:
        status = _status_chunk("h", ProcessState.COMPLETED.value)
        status.payload["truncated"] = True  # type: ignore[index]
        backend = command_backend([status])

        result = await run_command_impl(backend, RunCommandInput(argv=["echo", "x"]))

        assert result.truncated is True

    @pytest.mark.asyncio
    async def test_failed_state_propagated(
        self, command_backend, temp_workspace: Path
    ) -> None:
        backend = command_backend([_status_chunk("h", ProcessState.FAILED.value)])

        result = await run_command_impl(backend, RunCommandInput(argv=["false"]))

        assert result.state == ProcessState.FAILED.value
        assert result.return_code == 0  # status chunk did not override return_code

    @pytest.mark.asyncio
    async def test_env_parameter_not_accepted(
        self, command_backend, temp_workspace: Path
    ) -> None:
        """run_command has no env input field; backend is always called with env=None."""

        # ``env`` is not a field of the input model: it is not accepted.
        assert "env" not in RunCommandInput.model_fields

        backend = command_backend([_status_chunk("h", ProcessState.COMPLETED.value)])
        await run_command_impl(backend, RunCommandInput(argv=["echo", "x"]))
        last_call = backend._calls[-1]
        assert last_call["env"] is None

    @pytest.mark.asyncio
    async def test_real_command_execution(self, backend, temp_workspace: Path) -> None:
        """Real subprocess execution returns completed status with stdout."""
        executable = sys.executable
        result = await run_command_impl(
            backend, RunCommandInput(argv=[executable, "-c", "print('real-output')"])
        )

        assert isinstance(result, RunCommandOutput)
        assert result.state == ProcessState.COMPLETED.value
        assert result.return_code == 0
        assert result.handle_id is not None
        assert "real-output" in result.stdout

    @pytest.mark.asyncio
    async def test_real_command_with_cwd(self, backend, temp_workspace: Path) -> None:
        """Command runs in the supplied working directory (resolved within workspace)."""
        (temp_workspace / "subdir").mkdir()
        (temp_workspace / "subdir" / "marker.txt").write_text("marker-content")

        executable = sys.executable
        result = await run_command_impl(
            backend,
            RunCommandInput(
                argv=[
                    executable,
                    "-c",
                    "import pathlib; print(pathlib.Path('marker.txt').read_text())",
                ],
                cwd="subdir",
            ),
        )

        assert result.state == ProcessState.COMPLETED.value
        assert "marker-content" in result.stdout

    @pytest.mark.asyncio
    async def test_nonexistent_command_fails(
        self, backend, temp_workspace: Path
    ) -> None:
        """A command that cannot be resolved via PATH returns a FAILED result."""
        nonexistent = (
            "no-such-command-xyz-12345.exe" if _WINDOWS else "no-such-command-xyz-12345"
        )

        result = await run_command_impl(backend, RunCommandInput(argv=[nonexistent]))

        assert result.state == ProcessState.FAILED.value
        assert result.return_code is None


class TestRunCommandLLMOutput:
    def test_to_llm_dict_excludes_internal_fields(self) -> None:
        output = RunCommandOutput(
            handle_id="h",
            return_code=0,
            state=ProcessState.COMPLETED.value,
            stdout="out",
            stderr="err",
            truncated=False,
            stdout_bytes=123,
            stderr_bytes=456,
            elapsed_seconds=7.89,
        )

        llm_dict = output.to_llm_dict()
        assert set(llm_dict.keys()) == {
            "handle_id",
            "return_code",
            "state",
            "stdout",
            "stderr",
            "truncated",
        }
        assert "stdout_bytes" not in llm_dict
        assert "stderr_bytes" not in llm_dict
        assert "elapsed_seconds" not in llm_dict

    def test_tool_definition_output_schema_uses_model_llm_schema(self) -> None:
        registry = ToolRegistry()
        register_run_command(registry)
        definition = registry.get_tool("run_command")
        assert definition is not None
        assert definition.output_schema == RunCommandOutput.model_llm_schema()
        schema_str = str(definition.output_schema)
        assert "handle_id" in schema_str
        assert "return_code" in schema_str
        assert "state" in schema_str
        assert "stdout" in schema_str
        assert "stderr" in schema_str
        assert "truncated" in schema_str
        assert "stdout_bytes" not in schema_str
        assert "stderr_bytes" not in schema_str
        assert "elapsed_seconds" not in schema_str
