"""Unit tests for the check_process_status tool.

Covers the test cases defined in plan_docs/2026_09/tools_platform_tests.md
(section test_check_process_status_tool.py):

- check_process_status with valid handle_id returns state, return_code, elapsed_seconds
- check_process_status with completed process returns final state and return_code
- check_process_status with running process returns state=running, elapsed
- check_process_status with invalid handle_id raises ToolValidationError
- check_process_status acquires SHARED lock on process:handle_id
- check_process_status LLM output excludes exit_signal, os_pid, backend, start_time, last_activity, idle_seconds, resource_usage, metadata (via to_llm_dict)
- check_process_status ToolDefinition.output_schema uses model_llm_schema (only handle_id, state, return_code, elapsed_seconds)
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
    ProcessMetadata,
    ProcessStatusOutput,
    ResourceUsage,
    check_process_status_impl,
    register_check_process_status,
)


def _status_data(
    handle_id: str = "handle-1",
    state: str = ProcessState.RUNNING.value,
    return_code: int | None = None,
    elapsed_seconds: float | None = 2.5,
) -> dict[str, Any]:
    """Build a backend status dict shaped like HostBackend.check_process_status output."""
    return {
        "handle_id": handle_id,
        "state": state,
        "return_code": return_code,
        "exit_signal": None,
        "os_pid": 4242,
        "backend": "host",
        "start_time": "2026-10-22T20:00:00+00:00",
        "last_activity": None,
        "elapsed_seconds": elapsed_seconds,
        "idle_seconds": None,
        "resource_usage": {},
        "metadata": {},
    }


class TestCheckProcessStatusImpl:
    """Tests for check_process_status_impl status mapping and handle forwarding."""

    @pytest.mark.asyncio
    async def test_valid_handle_id_returns_state_return_code_elapsed(
        self, process_status_backend: MagicMock
    ) -> None:
        backend = process_status_backend(
            status_data=_status_data(
                handle_id="h-1", state=ProcessState.RUNNING.value, elapsed_seconds=2.5
            )
        )

        result = await check_process_status_impl(
            backend, CheckProcessStatusInput(handle_id="h-1"), session_id="sess-1"
        )

        assert isinstance(result, ProcessStatusOutput)
        assert result.handle_id == "h-1"
        assert result.state == ProcessState.RUNNING.value
        assert result.return_code is None
        assert result.elapsed_seconds == pytest.approx(2.5)
        assert backend._calls == ["h-1"]

    @pytest.mark.asyncio
    async def test_completed_process_returns_final_state_and_return_code(
        self, process_status_backend: MagicMock
    ) -> None:
        backend = process_status_backend(
            status_data=_status_data(
                handle_id="h-2",
                state=ProcessState.COMPLETED.value,
                return_code=0,
                elapsed_seconds=3.25,
            )
        )

        result = await check_process_status_impl(
            backend, CheckProcessStatusInput(handle_id="h-2")
        )

        assert isinstance(result, ProcessStatusOutput)
        assert result.state == ProcessState.COMPLETED.value
        assert result.return_code == 0
        assert result.elapsed_seconds == pytest.approx(3.25)

    @pytest.mark.asyncio
    async def test_running_process_returns_state_running_and_elapsed(
        self, process_status_backend: MagicMock
    ) -> None:
        backend = process_status_backend(
            status_data=_status_data(
                handle_id="h-3",
                state=ProcessState.RUNNING.value,
                elapsed_seconds=0.75,
            )
        )

        result = await check_process_status_impl(
            backend, CheckProcessStatusInput(handle_id="h-3")
        )

        assert isinstance(result, ProcessStatusOutput)
        assert result.state == ProcessState.RUNNING.value
        assert result.elapsed_seconds == pytest.approx(0.75)
        assert result.elapsed_seconds is not None


class TestCheckProcessStatusBackend:
    """Tests for check_process_status against a real HostBackend process registry."""

    @pytest.mark.asyncio
    async def test_completed_process_returns_final_state_and_return_code(
        self, backend: HostBackend
    ) -> None:
        handle = ProcessHandle(
            handle_id="h-real-completed",
            backend=BackendType.HOST,
            state=ProcessState.COMPLETED,
            return_code=0,
        )
        backend._supervisor._active_processes[handle.handle_id] = handle

        result = await check_process_status_impl(
            backend, CheckProcessStatusInput(handle_id="h-real-completed")
        )

        assert isinstance(result, ProcessStatusOutput)
        assert result.state == ProcessState.COMPLETED.value
        assert result.return_code == 0
        assert result.elapsed_seconds is not None

    @pytest.mark.asyncio
    async def test_running_process_returns_state_running_and_elapsed(
        self, backend: HostBackend
    ) -> None:
        handle = ProcessHandle(
            handle_id="h-real-running",
            backend=BackendType.HOST,
            state=ProcessState.RUNNING,
        )
        backend._supervisor._active_processes[handle.handle_id] = handle

        result = await check_process_status_impl(
            backend, CheckProcessStatusInput(handle_id="h-real-running")
        )

        assert isinstance(result, ProcessStatusOutput)
        assert result.state == ProcessState.RUNNING.value
        assert result.elapsed_seconds is not None


class TestCheckProcessStatusInvalidHandle:
    @pytest.mark.asyncio
    async def test_invalid_handle_id_raises_tool_validation_error(
        self, backend: HostBackend
    ) -> None:
        with pytest.raises(ToolValidationError):
            await check_process_status_impl(
                backend, CheckProcessStatusInput(handle_id="nope-nope")
            )


class TestCheckProcessStatusDefinition:
    def test_acquires_shared_lock_on_process_handle_id(self) -> None:
        registry = ToolRegistry()
        register_check_process_status(registry)
        definition = registry.get_tool("check_process_status")
        assert definition is not None
        assert definition.concurrency_scope is not None

        locks = definition.concurrency_scope.resource_locks({"handle_id": "h-1"})  # type: ignore[call-arg]
        assert locks == [ResourceLock(key="process:h-1", mode=LockMode.SHARED)]


class TestCheckProcessStatusLLMOutput:
    def test_to_llm_dict_excludes_internal_fields(self) -> None:
        output = ProcessStatusOutput(
            handle_id="h-1",
            state=ProcessState.COMPLETED.value,
            return_code=0,
            exit_signal=None,
            os_pid=4242,
            backend="host",
            start_time="2026-10-22T20:00:00+00:00",
            last_activity="2026-10-22T20:00:05+00:00",
            elapsed_seconds=2.5,
            idle_seconds=None,
            resource_usage=ResourceUsage(),
            metadata=ProcessMetadata(),
        )

        llm_dict = output.to_llm_dict()

        assert set(llm_dict) == {"handle_id", "state", "return_code", "elapsed_seconds"}
        assert llm_dict["handle_id"] == "h-1"
        assert llm_dict["state"] == ProcessState.COMPLETED.value
        assert llm_dict["return_code"] == 0
        assert llm_dict["elapsed_seconds"] == pytest.approx(2.5)

        for internal_field in (
            "exit_signal",
            "os_pid",
            "backend",
            "start_time",
            "last_activity",
            "idle_seconds",
            "resource_usage",
            "metadata",
        ):
            assert internal_field not in llm_dict

    def test_tool_definition_output_schema_uses_model_llm_schema(self) -> None:
        registry = ToolRegistry()
        register_check_process_status(registry)
        definition = registry.get_tool("check_process_status")
        assert definition is not None
        assert definition.output_schema == ProcessStatusOutput.model_llm_schema()

        properties = definition.output_schema.get("properties", {})
        assert set(properties) == {
            "handle_id",
            "state",
            "return_code",
            "elapsed_seconds",
        }
        assert "exit_signal" not in properties
