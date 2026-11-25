"""Unit tests for the kill_process tool.

Covers the test cases defined in plan_docs/2026_09/tools_platform_tests.md
(section test_kill_process_tool.py):

- kill_process with valid handle_id sends graceful termination, returns terminated=True
- kill_process returns final process state and termination details
- kill_process with invalid handle_id raises ToolValidationError
- kill_process acquires EXCLUSIVE lock on process:handle_id
- kill_process LLM output includes all fields (via to_llm_dict)
- kill_process ToolDefinition.output_schema uses model_llm_schema
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.tools_platform.backends.host import HostBackend
from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    LockMode,
    ProcessHandle,
    ProcessState,
    ResourceLock,
)
from local_coding_assistant.tools_platform.exceptions import ToolValidationError
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.tools.check_process_status import (
    CheckProcessStatusInput,
    check_process_status_impl,
)
from local_coding_assistant.tools_platform.tools.kill_process import (
    KillProcessInput,
    KillProcessOutput,
    kill_process_impl,
    register_kill_process,
)


def _status_data(
    handle_id: str = "handle-1",
    state: str = ProcessState.RUNNING.value,
    return_code: int | None = None,
    exit_signal: str | None = None,
) -> dict[str, Any]:
    """Build a backend status dict shaped like HostBackend.check_process_status output."""
    return {
        "handle_id": handle_id,
        "state": state,
        "return_code": return_code,
        "exit_signal": exit_signal,
        "os_pid": 4242,
        "backend": "host",
        "start_time": "2026-10-22T20:00:00+00:00",
        "last_activity": None,
        "elapsed_seconds": 2.5,
        "idle_seconds": None,
        "resource_usage": {},
        "metadata": {},
    }


class TestKillProcessImpl:
    """Tests for kill_process_impl cancellation and status mapping."""

    @pytest.mark.asyncio
    async def test_valid_handle_id_terminates_and_returns_details(
        self, kill_process_backend: MagicMock
    ) -> None:
        backend = kill_process_backend(
            status_data=[
                _status_data(handle_id="h-1", state=ProcessState.RUNNING.value),
                _status_data(
                    handle_id="h-1",
                    state=ProcessState.CANCELLED.value,
                    return_code=-9,
                    exit_signal="SIGKILL",
                ),
            ]
        )

        result = await kill_process_impl(
            backend, KillProcessInput(handle_id="h-1"), session_id="sess-1"
        )

        assert isinstance(result, KillProcessOutput)
        assert result.handle_id == "h-1"
        assert result.state == ProcessState.CANCELLED.value
        assert result.return_code == -9
        assert result.exit_signal == "SIGKILL"
        assert result.terminated is True
        assert result.already_completed is False
        assert backend._status_calls == ["h-1", "h-1"]
        assert backend._cancel_calls == ["h-1"]

    @pytest.mark.asyncio
    async def test_invalid_handle_id_raises_tool_validation_error(
        self, kill_process_backend: MagicMock
    ) -> None:
        backend = kill_process_backend(
            status_error=ToolValidationError(
                message="Process handle not found: nope-nope",
                details={"handle_id": "nope-nope"},
            )
        )

        with pytest.raises(ToolValidationError):
            await kill_process_impl(backend, KillProcessInput(handle_id="nope-nope"))

        # Cancellation must not be attempted when the handle is invalid.
        assert backend._status_calls == ["nope-nope"]
        assert backend._cancel_calls == []

    @pytest.mark.asyncio
    async def test_already_completed_process_not_terminated(
        self, kill_process_backend: MagicMock
    ) -> None:
        backend = kill_process_backend(
            status_data=_status_data(
                handle_id="h-done",
                state=ProcessState.COMPLETED.value,
                return_code=0,
            )
        )

        result = await kill_process_impl(backend, KillProcessInput(handle_id="h-done"))

        assert result.already_completed is True
        assert result.state == ProcessState.COMPLETED.value
        assert result.return_code == 0
        assert result.terminated is True

    @pytest.mark.asyncio
    async def test_cancel_failure_returns_terminated_false(
        self, kill_process_backend: MagicMock
    ) -> None:
        backend = kill_process_backend(
            status_data=_status_data(handle_id="h-2"),
            cancel_error=RuntimeError("cancel failed"),
        )

        result = await kill_process_impl(backend, KillProcessInput(handle_id="h-2"))

        assert result.terminated is False
        assert result.state == ProcessState.RUNNING.value


class TestKillProcessDefinition:
    def test_acquires_exclusive_lock_on_process_handle_id(self) -> None:
        registry = ToolRegistry()
        register_kill_process(registry)
        definition = registry.get_tool("kill_process")
        assert definition is not None
        assert definition.concurrency_scope is not None

        locks = definition.concurrency_scope.resource_locks({"handle_id": "h-1"})  # type: ignore[call-arg]
        assert locks == [ResourceLock(key="process:h-1", mode=LockMode.EXCLUSIVE)]

    def test_tool_definition_output_schema_uses_model_llm_schema(self) -> None:
        registry = ToolRegistry()
        register_kill_process(registry)
        definition = registry.get_tool("kill_process")
        assert definition is not None
        assert definition.output_schema == KillProcessOutput.model_llm_schema()

        properties = definition.output_schema.get("properties", {})
        assert set(properties) == {
            "handle_id",
            "state",
            "return_code",
            "exit_signal",
            "terminated",
            "already_completed",
        }


class TestKillProcessLLMOutput:
    def test_to_llm_dict_includes_all_fields(self) -> None:
        output = KillProcessOutput(
            handle_id="h-1",
            state=ProcessState.CANCELLED.value,
            return_code=-9,
            exit_signal="SIGKILL",
            terminated=True,
            already_completed=False,
        )

        llm_dict = output.to_llm_dict()

        assert set(llm_dict) == {
            "handle_id",
            "state",
            "return_code",
            "exit_signal",
            "terminated",
            "already_completed",
        }
        assert llm_dict["handle_id"] == "h-1"
        assert llm_dict["state"] == ProcessState.CANCELLED.value
        assert llm_dict["return_code"] == -9
        assert llm_dict["exit_signal"] == "SIGKILL"
        assert llm_dict["terminated"] is True
        assert llm_dict["already_completed"] is False


class TestKillProcessRoundtrip:
    """End-to-end process lifecycle against the real HostBackend:

    check_process_status (running) → kill_process (terminated) →
    check_process_status (cancelled).

    ``run_command_impl`` blocks until a command completes and the supervisor
    removes completed handles from ``_active_processes`` (supervisor.py:194),
    so the live check→kill→check roundtrip is exercised by registering a
    running handle directly (mirroring test_check_process_status_tool.py).
    os_pid is None so no real OS process is signalled.
    """

    @pytest.mark.asyncio
    async def test_check_kill_check_lifecycle(self, backend: HostBackend) -> None:
        handle_id = "h-rt-1"
        handle = ProcessHandle(
            handle_id=handle_id,
            os_pid=None,
            state=ProcessState.RUNNING,
            backend=BackendType.HOST,
        )
        backend._supervisor._active_processes[handle_id] = handle

        # 1. check_process_status: running
        status = await check_process_status_impl(
            backend, CheckProcessStatusInput(handle_id=handle_id)
        )
        assert status.state == ProcessState.RUNNING.value
        assert status.return_code is None

        # 2. kill_process: terminates the running process
        result = await kill_process_impl(backend, KillProcessInput(handle_id=handle_id))
        assert result.terminated is True
        assert result.already_completed is False
        assert result.state == ProcessState.CANCELLED.value

        # 3. check_process_status: cancelled after kill
        final = await check_process_status_impl(
            backend, CheckProcessStatusInput(handle_id=handle_id)
        )
        assert final.state == ProcessState.CANCELLED.value
        assert (
            backend._supervisor._active_processes[handle_id].state
            == ProcessState.CANCELLED
        )
