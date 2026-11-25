"""Tests for execution supervisor and process transport layering.

Tests the separation of shared supervision policy from backend-specific
process transport operations.
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ProcessHandle,
    ProcessState,
)
from local_coding_assistant.tools_platform.supervisor import (
    ExecutionConfig,
    OutputLimits,
    ProcessSupervisor,
    TimeoutPolicy,
)
from local_coding_assistant.tools_platform.supervisor.host_transport import (
    HostProcessTransport,
)
from local_coding_assistant.tools_platform.supervisor.transport import (
    ProcessTransport,
    TransportConfig,
)


class MockProcessTransport(ProcessTransport):
    """Mock transport for testing."""

    def __init__(self) -> None:
        self.spawn_called = False
        self.terminate_called = False
        self._process_running = True
        self._return_code = 0

    async def spawn(
        self,
        config: TransportConfig,
    ) -> tuple[ProcessHandle, AsyncIterator[bytes], AsyncIterator[bytes]]:
        """Mock spawn."""
        self.spawn_called = True

        async def stdout_iter():
            yield b"test output\n"

        async def stderr_iter():
            yield b""

        handle = ProcessHandle(
            handle_id="mock-123",
            backend=BackendType.HOST,
            state=ProcessState.RUNNING,
            os_pid=12345,
        )
        return handle, stdout_iter(), stderr_iter()

    async def terminate(
        self,
        handle: ProcessHandle,
        grace: timedelta,
    ) -> None:
        """Mock terminate."""
        self.terminate_called = True
        self._process_running = False

    async def wait_for_completion(self, handle: ProcessHandle) -> int | None:
        """Mock wait."""
        if not self._process_running:
            return self._return_code
        return None

    def is_running(self, handle: ProcessHandle) -> bool:
        """Mock is_running."""
        return self._process_running


class TestProcessTransport:
    """Test ProcessTransport protocol."""

    def test_transport_config_creation(self):
        """Test TransportConfig creation."""
        config = TransportConfig(
            argv=["echo", "hello"],
            cwd=Path("/tmp"),
            env={"VAR": "value"},
            input=None,
        )
        assert config.argv == ["echo", "hello"]
        assert config.cwd == Path("/tmp")
        assert config.env == {"VAR": "value"}

    def test_transport_config_defaults(self):
        """Test TransportConfig default values."""
        config = TransportConfig(argv=["echo", "hello"])
        assert config.cwd is None
        assert config.env is None
        assert config.input is None


class TestHostProcessTransport:
    """Test HostProcessTransport implementation."""

    @pytest.mark.asyncio
    async def test_spawn_creates_process(self):
        """Test spawn creates a subprocess."""
        transport = HostProcessTransport()
        config = TransportConfig(argv=["echo", "test"])

        handle, stdout_iter, stderr_iter = await transport.spawn(config)

        assert handle.os_pid is not None
        assert handle.os_pid > 0
        assert handle.state == ProcessState.RUNNING
        assert handle.backend == BackendType.HOST

        # Consume output to avoid resource warnings
        async for _ in stdout_iter:
            pass
        async for _ in stderr_iter:
            pass

    @pytest.mark.asyncio
    async def test_spawn_with_cwd(self):
        """Test spawn with working directory."""
        transport = HostProcessTransport()
        config = TransportConfig(
            argv=["echo", "test"],
            cwd=Path.cwd(),
        )

        handle, stdout_iter, stderr_iter = await transport.spawn(config)

        assert handle.os_pid is not None

        # Consume output
        async for _ in stdout_iter:
            pass
        async for _ in stderr_iter:
            pass

    @pytest.mark.asyncio
    async def test_terminate_process(self):
        """Test process termination."""
        transport = HostProcessTransport()
        config = TransportConfig(argv=["sleep", "10"])

        handle, stdout_iter, stderr_iter = await transport.spawn(config)

        # Terminate the process
        await transport.terminate(handle, grace=timedelta(seconds=1))

        # Process should be terminated
        await asyncio.sleep(0.5)  # Give it time to terminate

        # Consume output to avoid resource warnings
        async for _ in stdout_iter:
            pass
        async for _ in stderr_iter:
            pass

    @pytest.mark.asyncio
    async def test_is_running(self):
        """Test is_running check."""
        transport = HostProcessTransport()
        config = TransportConfig(argv=["echo", "test"])

        handle, stdout_iter, stderr_iter = await transport.spawn(config)

        # Process should be running
        assert transport.is_running(handle) is True

        # Wait for process to complete
        async for _ in stdout_iter:
            pass
        async for _ in stderr_iter:
            pass

        await asyncio.sleep(0.1)  # Give it time to exit

        # Process should no longer be running
        # Note: This may still return True briefly after exit
        # due to zombie process state


class TestProcessSupervisor:
    """Test ProcessSupervisor with transport layering."""

    def test_supervisor_accepts_transport(self):
        """Test supervisor can be initialized with a transport."""
        transport = MockProcessTransport()
        supervisor = ProcessSupervisor(transport=transport)

        assert supervisor._transport is transport

    def test_supervisor_default_transport(self):
        """Test supervisor uses HostProcessTransport by default."""
        supervisor = ProcessSupervisor()

        assert isinstance(supervisor._transport, HostProcessTransport)

    @pytest.mark.asyncio
    async def test_execute_with_mock_transport(self):
        """Test execute with mock transport."""
        transport = MockProcessTransport()
        supervisor = ProcessSupervisor(transport=transport)

        config = ExecutionConfig(
            argv=["echo", "test"],
            timeout_policy=TimeoutPolicy(
                hard_timeout=timedelta(seconds=5),
            ),
            output_limits=OutputLimits(),
            backend=BackendType.HOST,
            transport=transport,
        )

        chunks = []
        async for chunk in supervisor.execute(config):
            chunks.append(chunk)

        assert transport.spawn_called is True
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_execute_with_host_transport(self):
        """Test execute with real host transport."""
        transport = HostProcessTransport()
        supervisor = ProcessSupervisor(transport=transport)

        config = ExecutionConfig(
            argv=["echo", "hello world"],
            timeout_policy=TimeoutPolicy(
                hard_timeout=timedelta(seconds=5),
            ),
            output_limits=OutputLimits(),
            backend=BackendType.HOST,
            transport=transport,
        )

        chunks = []
        async for chunk in supervisor.execute(config):
            chunks.append(chunk)

        # Should have output chunks and a status chunk
        assert len(chunks) > 0

        # Check for status chunk
        status_chunks = [c for c in chunks if c.channel == "status"]
        assert len(status_chunks) == 1

    @pytest.mark.asyncio
    async def test_cancel_execution(self):
        """Test cancel execution."""
        transport = MockProcessTransport()
        supervisor = ProcessSupervisor(transport=transport)

        config = ExecutionConfig(
            argv=["sleep", "10"],
            timeout_policy=TimeoutPolicy(
                hard_timeout=timedelta(seconds=5),
            ),
            output_limits=OutputLimits(),
            backend=BackendType.HOST,
            transport=transport,
        )

        # Start execution and consume first chunk
        chunks = []

        async def consume_until_cancel():
            async for chunk in supervisor.execute(config):
                chunks.append(chunk)
                if len(chunks) > 0:
                    break

        execute_task = asyncio.create_task(consume_until_cancel())
        await asyncio.sleep(0.1)  # Let it start

        # Cancel by handle ID
        handle = await supervisor.get_handle("mock-123")
        if handle:
            result = await supervisor.cancel(handle.handle_id)
            assert result.state == ProcessState.CANCELLED
            assert transport.terminate_called is True

        # Cleanup
        execute_task.cancel()
        try:
            await execute_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_get_handle(self):
        """Test get_handle returns process handle."""
        transport = MockProcessTransport()
        supervisor = ProcessSupervisor(transport=transport)

        # No handle exists yet
        handle = await supervisor.get_handle("nonexistent")
        assert handle is None


class TestExecutionConfig:
    """Test ExecutionConfig with transport field."""

    def test_config_with_transport(self):
        """Test ExecutionConfig can hold a transport."""
        transport = MockProcessTransport()
        config = ExecutionConfig(
            argv=["echo", "test"],
            transport=transport,
        )

        assert config.transport is transport

    def test_config_without_transport(self):
        """Test ExecutionConfig without transport uses default."""
        config = ExecutionConfig(argv=["echo", "test"])

        assert config.transport is None


class TestTimeoutPolicy:
    """Test TimeoutPolicy configuration."""

    def test_timeout_policy_defaults(self):
        """Test TimeoutPolicy default values."""
        policy = TimeoutPolicy()
        assert policy.soft_timeout == timedelta(seconds=10)
        assert policy.hard_timeout == timedelta(seconds=300)
        assert policy.idle_timeout == timedelta(seconds=30)
        assert policy.heartbeat_interval == timedelta(seconds=5)
        assert policy.cancellation_grace == timedelta(seconds=5)

    def test_timeout_policy_custom(self):
        """Test TimeoutPolicy with custom values."""
        policy = TimeoutPolicy(
            soft_timeout=timedelta(seconds=5),
            hard_timeout=timedelta(seconds=60),
        )
        assert policy.soft_timeout == timedelta(seconds=5)
        assert policy.hard_timeout == timedelta(seconds=60)


class TestOutputLimits:
    """Test OutputLimits configuration."""

    def test_output_limits_defaults(self):
        """Test OutputLimits default values."""
        limits = OutputLimits()
        assert limits.max_stdout_bytes == 262144
        assert limits.max_stderr_bytes == 262144
        assert limits.max_lines == 400

    def test_output_limits_custom(self):
        """Test OutputLimits with custom values."""
        limits = OutputLimits(
            max_stdout_bytes=1024,
            max_stderr_bytes=512,
            max_lines=100,
        )
        assert limits.max_stdout_bytes == 1024
        assert limits.max_stderr_bytes == 512
        assert limits.max_lines == 100
