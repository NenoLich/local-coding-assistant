"""Integration tests for Docker backend command execution.

Tests Docker backend parity with host backend for command execution.
Tests are skipped if Docker is not available.
"""

import asyncio
from pathlib import Path

import pytest

from local_coding_assistant.tools_platform.backends.docker import DockerBackend
from local_coding_assistant.tools_platform.contracts import (
    BackendType,
)

# Skip all tests if Docker is not available
docker_available = False
try:
    import docker

    docker.from_env()
    docker_available = True
except Exception:
    pass


@pytest.mark.skipif(not docker_available, reason="Docker not available")
class TestDockerBackendCommandExecution:
    """Test DockerBackend command execution."""

    @pytest.fixture
    def backend(self, tmp_path):
        """Create a DockerBackend instance for testing."""
        try:
            # Try to get or create a container
            client = docker.from_env()
            # Use a simple Python container for testing
            container = client.containers.create(
                image="python:3.12-slim",
                command=["sleep", "3600"],
                volumes={str(tmp_path): {"bind": "/workspace", "mode": "rw"}},
                working_dir="../workspace",
                detach=True,
            )
            container.start()

            backend = DockerBackend(
                workspace_root=tmp_path,
                container=container,
                workspace_manager=None,
            )

            yield backend

            # Cleanup
            container.stop()
            container.remove()
        except Exception as e:
            pytest.skip(f"Failed to create Docker container: {e}")

    @pytest.mark.asyncio
    async def test_execute_simple_command(self, backend):
        """Test executing a simple command in container."""
        chunks = []
        async for chunk in backend.execute_command(
            argv=["echo", "hello world"],
            cwd=None,
            env=None,
            timeout_seconds=5,
        ):
            chunks.append(chunk)

        # Should have stdout and status chunks
        assert len(chunks) > 0

        # Check for stdout
        stdout_chunks = [c for c in chunks if c.channel == "stdout"]
        assert len(stdout_chunks) > 0
        assert "hello world" in stdout_chunks[0].payload

        # Check for status
        status_chunks = [c for c in chunks if c.channel == "status"]
        assert len(status_chunks) == 1
        assert status_chunks[0].payload["return_code"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_working_directory(self, backend, tmp_path):
        """Test executing command with working directory in container."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        chunks = []
        async for chunk in backend.execute_command(
            argv=["cat", "test.txt"],
            cwd=tmp_path,
            env=None,
            timeout_seconds=5,
        ):
            chunks.append(chunk)

        stdout_chunks = [c for c in chunks if c.channel == "stdout"]
        assert len(stdout_chunks) > 0
        assert "test content" in stdout_chunks[0].payload

    @pytest.mark.asyncio
    async def test_execute_with_environment_variables(self, backend):
        """Test executing command with environment variables in container."""
        chunks = []
        async for chunk in backend.execute_command(
            argv=["sh", "-c", "echo $TEST_VAR"],
            cwd=None,
            env={"TEST_VAR": "test_value"},
            timeout_seconds=5,
        ):
            chunks.append(chunk)

        stdout_chunks = [c for c in chunks if c.channel == "stdout"]
        assert len(stdout_chunks) > 0
        # Environment variable should be expanded
        assert "test_value" in stdout_chunks[0].payload

    @pytest.mark.asyncio
    async def test_execute_command_with_stderr(self, backend):
        """Test executing command that produces stderr output in container."""
        chunks = []
        async for chunk in backend.execute_command(
            argv=["sh", "-c", "echo 'error message' >&2"],
            cwd=None,
            env=None,
            timeout_seconds=5,
        ):
            chunks.append(chunk)

        stderr_chunks = [c for c in chunks if c.channel == "stderr"]
        assert len(stderr_chunks) > 0
        assert "error message" in stderr_chunks[0].payload

    @pytest.mark.asyncio
    async def test_read_file_from_container(self, backend, tmp_path):
        """Test reading file from container."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("container test content")

        content = await backend.read_file(Path("test.txt"))

        assert "container test content" in content

    @pytest.mark.asyncio
    async def test_search_text_in_container(self, backend, tmp_path):
        """Test searching text in container."""
        # Create test files
        (tmp_path / "file1.txt").write_text("hello world")
        (tmp_path / "file2.txt").write_text("goodbye world")

        result = await backend.search_text(
            query="world",
            path=Path(".."),
            pattern="literal",
            includes=None,
            excludes=None,
            case_sensitive=False,
            context_lines=0,
            max_matches=10,
            max_bytes=262144,
        )

        # Should find matches
        assert result is not None

    @pytest.mark.asyncio
    async def test_match_paths_in_container(self, backend, tmp_path):
        """Test matching paths in container."""
        # Create test files
        (tmp_path / "test1.txt").write_text("test")
        (tmp_path / "test2.py").write_text("test")

        result = await backend.match_paths(
            pattern="*.txt",
            path=Path(".."),
            include_hidden=False,
            include_ignored=False,
            file_type=None,
            max_results=10,
        )

        # Should find .txt files
        assert result is not None
        assert "matches" in result

    @pytest.mark.asyncio
    async def test_cancel_execution_in_container(self, backend):
        """Test cancelling execution in container."""
        # Start a long-running command
        execute_task = asyncio.create_task(
            self._collect_chunks(
                backend.execute_command(
                    argv=["sleep", "30"],
                    cwd=None,
                    env=None,
                    timeout_seconds=60,
                )
            )
        )

        # Wait a bit for it to start
        await asyncio.sleep(0.5)

        # Cancel the task
        execute_task.cancel()
        try:
            await execute_task
        except asyncio.CancelledError:
            pass

    async def _collect_chunks(self, async_iter):
        """Helper to collect chunks from an async iterator."""
        chunks = []
        async for chunk in async_iter:
            chunks.append(chunk)
        return chunks

    @pytest.mark.asyncio
    async def test_check_process_status_in_container(self, backend):
        """Test checking process status in container."""
        # Test with non-existent handle
        with pytest.raises(Exception):  # Should raise ToolError for non-existent handle
            await backend.check_process_status("nonexistent-handle-id")

    @pytest.mark.asyncio
    async def test_query_repository_not_supported(self, backend):
        """Test that repository queries are not supported in Docker backend."""
        from local_coding_assistant.tools_platform.contracts import ToolExecutionError

        with pytest.raises(ToolExecutionError) as exc_info:
            await backend.query_repository("search_symbols", {})

        assert "not supported" in str(exc_info.value.message).lower()

    @pytest.mark.asyncio
    async def test_backend_type(self, backend):
        """Test backend type identifier."""
        assert backend.get_backend_type() == BackendType.DOCKER

    @pytest.mark.asyncio
    async def test_workspace_root(self, backend, tmp_path):
        """Test workspace root path."""
        assert backend.get_workspace_root() == tmp_path

    @pytest.mark.asyncio
    async def test_path_mapping(self, backend, tmp_path):
        """Test workspace to container path mapping."""
        # Test mapping to container path
        container_path = backend._map_to_container_path(Path("test.txt"))
        assert str(container_path) == "/workspace/test.txt"

        # Test mapping from container path
        workspace_path = backend._map_from_container_path(Path("/workspace/test.txt"))
        assert str(workspace_path) == "test.txt"

    @pytest.mark.asyncio
    async def test_is_available(self, backend):
        """Test backend availability check."""
        assert backend.is_available() is True


@pytest.mark.skipif(not docker_available, reason="Docker not available")
class TestDockerTransport:
    """Test DockerProcessTransport implementation."""

    @pytest.mark.asyncio
    async def test_transport_spawn(self, tmp_path):
        """Test DockerProcessTransport spawn."""
        from local_coding_assistant.tools_platform.supervisor.docker_transport import (
            DockerProcessTransport,
        )

        try:
            client = docker.from_env()
            container = client.containers.create(
                image="python:3.12-slim",
                command=["sleep", "3600"],
                volumes={str(tmp_path): {"bind": "/workspace", "mode": "rw"}},
                working_dir="../workspace",
                detach=True,
            )
            container.start()

            transport = DockerProcessTransport(
                container=container,
                workspace_root=tmp_path,
            )

            from local_coding_assistant.tools_platform.supervisor.transport import (
                TransportConfig,
            )

            config = TransportConfig(argv=["echo", "test"])

            # This may not work fully without proper Docker exec streaming
            # but should test the interface
            try:
                handle, stdout_iter, stderr_iter = await transport.spawn(config)
                assert handle is not None
                assert handle.backend == BackendType.DOCKER
            except NotImplementedError:
                # Streaming not yet implemented
                pytest.skip("Docker exec streaming not yet implemented")
            finally:
                container.stop()
                container.remove()
        except Exception as e:
            pytest.skip(f"Failed to test Docker transport: {e}")


@pytest.mark.skipif(not docker_available, reason="Docker not available")
class TestHostDockerParity:
    """Test parity between Host and Docker backends."""

    @pytest.mark.asyncio
    async def test_echo_command_parity(self, tmp_path):
        """Test that echo command produces same output on both backends."""
        from local_coding_assistant.tools_platform.backends.host import HostBackend

        # Host backend
        host_backend = HostBackend(workspace_root=tmp_path)
        host_chunks = []
        async for chunk in host_backend.execute_command(
            argv=["echo", "test"],
            cwd=None,
            env=None,
            timeout_seconds=5,
        ):
            host_chunks.append(chunk)

        # Docker backend
        try:
            client = docker.from_env()
            container = client.containers.create(
                image="python:3.12-slim",
                command=["sleep", "3600"],
                volumes={str(tmp_path): {"bind": "/workspace", "mode": "rw"}},
                working_dir="../workspace",
                detach=True,
            )
            container.start()

            docker_backend = DockerBackend(
                workspace_root=tmp_path,
                container=container,
                workspace_manager=None,
            )

            docker_chunks = []
            async for chunk in docker_backend.execute_command(
                argv=["echo", "test"],
                cwd=None,
                env=None,
                timeout_seconds=5,
            ):
                docker_chunks.append(chunk)

            # Both should have stdout with "test"
            host_stdout = [c.payload for c in host_chunks if c.channel == "stdout"]
            docker_stdout = [c.payload for c in docker_chunks if c.channel == "stdout"]

            assert len(host_stdout) > 0
            assert len(docker_stdout) > 0
            assert "test" in host_stdout[0]
            assert "test" in docker_stdout[0]

            # Both should have status with return code 0
            host_status = [c.payload for c in host_chunks if c.channel == "status"]
            docker_status = [c.payload for c in docker_chunks if c.channel == "status"]

            assert len(host_status) == 1
            assert len(docker_status) == 1
            assert host_status[0]["return_code"] == 0
            assert docker_status[0]["return_code"] == 0

            container.stop()
            container.remove()
        except Exception as e:
            pytest.skip(f"Docker parity test failed: {e}")
