"""Unit tests for DockerSandbox container management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from local_coding_assistant.sandbox.docker_sandbox import DockerSandbox
from local_coding_assistant.sandbox.exceptions import SandboxRuntimeError


class FakePathManager:
    """Minimal PathManager stub for isolating filesystem behavior."""

    def __init__(self, project_root: Path):
        self._project_root = project_root
        self.project_root_calls = 0
        self.resolve_calls: list[str] = []

    def get_project_root(self) -> Path:
        self.project_root_calls += 1
        return self._project_root

    def resolve_path(self, path: str) -> Path:
        self.resolve_calls.append(path)
        if path == "@tools/sandbox_tools":
            resolved = self._project_root / "tools" / "sandbox_tools"
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved
        msg = f"Unexpected resolve_path call: {path}"
        raise ValueError(msg)


class DummyContainer:
    """Minimal container stub for reuse tests."""

    def __init__(self, *, status: str = "running", ident: str = "dummy"):
        self.status = status
        self.id = ident
        self.reload_calls = 0

    def reload(self) -> None:
        self.reload_calls += 1


@pytest.fixture()
def sandbox_project_root(tmp_path: Path) -> Path:
    """Provide a temporary project root with expected sub-structure."""
    (tmp_path / "src" / "local_coding_assistant" / "sandbox" / "guest").mkdir(
        parents=True, exist_ok=True
    )
    return tmp_path


@pytest.mark.asyncio
async def test_reuse_tracked_container_rejects_default_session(
    sandbox_project_root: Path,
) -> None:
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    sandbox._containers["default"] = DummyContainer(ident="cont-1")
    sandbox._cleanup_container = AsyncMock()  # type: ignore[attr-defined]

    reused = await sandbox._reuse_tracked_container("default", persistence=True)

    assert reused is None
    sandbox._cleanup_container.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_reuse_tracked_container_drops_non_running_container(
    sandbox_project_root: Path,
) -> None:
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    sandbox._cleanup_container = AsyncMock()  # type: ignore[attr-defined]
    sandbox._containers["session"] = DummyContainer(status="exited")

    reused = await sandbox._reuse_tracked_container("session", persistence=True)

    assert reused is None
    sandbox._cleanup_container.assert_not_awaited()  # type: ignore[attr-defined]
    assert "session" not in sandbox._containers


@pytest.mark.asyncio
async def test_cleanup_container_success(sandbox_project_root: Path) -> None:
    """Test successful container cleanup with IPC directory removal."""
    import shutil
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    paths = await sandbox.ensure_directories()
    
    mock_container = Mock()
    mock_container.status = "running"
    mock_container.id = "container-id"
    mock_container.exec_run.return_value = (0, b"", b"")
    
    # Create the IPC directory so it exists for cleanup
    ipc_dir = paths.host_workspace / "ipc" / "locca_ipc_test-session"
    ipc_dir.mkdir(parents=True, exist_ok=True)
    
    with patch.object(sandbox, 'stop_container', new_callable=AsyncMock) as mock_stop, \
         patch('shutil.rmtree') as mock_rmtree:
        
        await sandbox._cleanup_container("test-session", mock_container, remove=True)
        
        mock_stop.assert_called_once_with("container-id", remove=True, session_id="test-session")
        # Verify IPC directory cleanup was attempted
        mock_rmtree.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_container_no_container(sandbox_project_root: Path) -> None:
    """Test container cleanup when container is None."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    with patch.object(sandbox, 'stop_container', new_callable=AsyncMock) as mock_stop:
        await sandbox._cleanup_container("test-session", None, remove=True)
        
        mock_stop.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_container_stop_failure(sandbox_project_root: Path) -> None:
    """Test container cleanup when stopping container fails."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    paths = await sandbox.ensure_directories()
    
    mock_container = Mock()
    mock_container.id = "container-id"
    mock_container.status = "running"
    
    # Create the IPC directory so it exists for cleanup
    ipc_dir = paths.host_workspace / "ipc" / "locca_ipc_test-session"
    ipc_dir.mkdir(parents=True, exist_ok=True)
    
    with patch.object(sandbox, 'stop_container', new_callable=AsyncMock) as mock_stop, \
         patch('shutil.rmtree') as mock_rmtree:
        mock_stop.side_effect = Exception("Stop failed")
        
        await sandbox._cleanup_container("test-session", mock_container, remove=True)
        
        mock_stop.assert_called_once()
        # IPC cleanup should still happen despite stop failure
        mock_rmtree.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_container_default_session(sandbox_project_root: Path) -> None:
    """Test container cleanup for default session (no IPC cleanup)."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    
    with patch.object(sandbox, 'stop_container', new_callable=AsyncMock) as mock_stop, \
         patch('shutil.rmtree') as mock_rmtree:
        
        await sandbox._cleanup_container("default", mock_container, remove=True)
        
        mock_stop.assert_called_once()
        mock_rmtree.assert_not_called()


@pytest.mark.asyncio
async def test_get_existing_container_success(sandbox_project_root: Path) -> None:
    """Test successful retrieval of existing running container."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    mock_container.status = "running"
    mock_container.reload.return_value = None
    
    mock_client = Mock()
    mock_client.containers.get.return_value = mock_container
    sandbox._client = mock_client
    
    with patch.object(sandbox, '_should_reuse_running_container', return_value=True):
        result = await sandbox._reuse_existing_named_container("session1", "test-container", True)
        
    assert result == mock_container
    assert "session1" in sandbox._containers
    assert sandbox._containers["session1"] == mock_container


@pytest.mark.asyncio
async def test_get_existing_container_not_found(sandbox_project_root: Path) -> None:
    """Test retrieval when container doesn't exist."""
    import docker.errors
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_client = Mock()
    mock_client.containers.get.side_effect = docker.errors.NotFound("Not found")
    sandbox._client = mock_client
    
    result = await sandbox._reuse_existing_named_container("test-container", "session1", True)
    
    assert result is None


@pytest.mark.asyncio
async def test_get_existing_container_reload_failure(sandbox_project_root: Path) -> None:
    """Test retrieval when container reload fails."""
    import docker.errors
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    mock_container.reload.side_effect = Exception("Reload failed")
    
    mock_client = Mock()
    mock_client.containers.get.return_value = mock_container
    sandbox._client = mock_client
    
    result = await sandbox._reuse_existing_named_container("test-container", "session1", True)
    
    assert result is None


@pytest.mark.asyncio
async def test_get_existing_container_not_running(sandbox_project_root: Path) -> None:
    """Test retrieval when container is not running."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    mock_container.status = "exited"
    mock_container.reload.return_value = None
    mock_container.remove.return_value = None
    
    mock_client = Mock()
    mock_client.containers.get.return_value = mock_container
    sandbox._client = mock_client
    
    result = await sandbox._reuse_existing_named_container("test-container", "session1", True)
    
    assert result is None
    mock_container.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_get_existing_container_should_not_reuse(sandbox_project_root: Path) -> None:
    """Test retrieval when container should not be reused."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    mock_container.status = "running"
    mock_container.reload.return_value = None
    
    mock_client = Mock()
    mock_client.containers.get.return_value = mock_container
    sandbox._client = mock_client
    
    with patch.object(sandbox, '_should_reuse_running_container', return_value=False), \
         patch.object(sandbox, '_cleanup_container', new_callable=AsyncMock) as mock_cleanup:
        
        result = await sandbox._reuse_existing_named_container("session1", "test-container", True)
        
    assert result is None
    mock_cleanup.assert_called_once_with("session1", mock_container, remove=True)


@pytest.mark.asyncio
async def test_start_session_success_with_new_container(sandbox_project_root: Path) -> None:
    """Test successful session start with new container creation."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    await sandbox.ensure_directories()
    
    mock_container = Mock()
    mock_container.short_id = "abc123"
    
    with patch.object(sandbox, '_ensure_initialized', new_callable=AsyncMock) as mock_init, \
         patch.object(sandbox, '_reuse_tracked_container', new_callable=AsyncMock, return_value=None) as mock_reuse, \
         patch.object(sandbox, '_ensure_image_ready', new_callable=AsyncMock) as mock_image, \
         patch.object(sandbox, '_build_volume_mounts', return_value={}) as mock_volumes, \
         patch.object(sandbox, '_get_network_mode', return_value="none") as mock_network, \
         patch.object(sandbox, '_determine_container_parameters', return_value=("test-name", None)) as mock_params, \
         patch.object(sandbox, '_build_logging_config', return_value={}) as mock_logging, \
         patch.object(sandbox, '_build_environment', return_value={}) as mock_env, \
         patch.object(sandbox, '_reuse_existing_named_container', new_callable=AsyncMock, return_value=None) as mock_named_reuse, \
         patch.object(sandbox, '_create_container', return_value=mock_container) as mock_create:
        
        result = await sandbox.start_container("test-session", persistence=False)
        
        assert result == mock_container
        assert "test-session" in sandbox._containers
        assert sandbox._containers["test-session"] == mock_container
        mock_init.assert_called_once()
        mock_reuse.assert_called_once_with("test-session", False)
        mock_image.assert_called_once()
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_start_session_success_with_reused_container(sandbox_project_root: Path) -> None:
    """Test successful session start with reused container."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    await sandbox.ensure_directories()
    
    mock_container = Mock()
    
    with patch.object(sandbox, '_ensure_initialized', new_callable=AsyncMock) as mock_init, \
         patch.object(sandbox, '_reuse_tracked_container', new_callable=AsyncMock, return_value=mock_container) as mock_reuse, \
         patch.object(sandbox, '_ensure_image_ready', new_callable=AsyncMock) as mock_image, \
         patch.object(sandbox, '_build_volume_mounts', return_value={}) as mock_volumes, \
         patch.object(sandbox, '_get_network_mode', return_value="none") as mock_network, \
         patch.object(sandbox, '_determine_container_parameters', return_value=("test-name", None)) as mock_params, \
         patch.object(sandbox, '_build_logging_config', return_value={}) as mock_logging, \
         patch.object(sandbox, '_build_environment', return_value={}) as mock_env, \
         patch.object(sandbox, '_reuse_existing_named_container', new_callable=AsyncMock, return_value=None) as mock_named_reuse:
        
        result = await sandbox.start_container("test-session", persistence=False)
        
        assert result == mock_container
        mock_init.assert_called_once()
        mock_reuse.assert_called_once_with("test-session", False)


@pytest.mark.asyncio
async def test_start_session_persistent_validates_capacity(sandbox_project_root: Path) -> None:
    """Test that persistent sessions validate capacity."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root), max_sessions=1)
    await sandbox.ensure_directories()
    sandbox._containers["existing"] = Mock()
    
    with patch.object(sandbox, '_ensure_initialized', new_callable=AsyncMock), \
         patch.object(sandbox, '_reuse_tracked_container', new_callable=AsyncMock, return_value=None), \
         patch.object(sandbox, '_ensure_image_ready', new_callable=AsyncMock), \
         patch.object(sandbox, '_build_volume_mounts', return_value={}) as mock_volumes, \
         patch.object(sandbox, '_get_network_mode', return_value="none") as mock_network, \
         patch.object(sandbox, '_determine_container_parameters', return_value=("test-name", None)) as mock_params, \
         patch.object(sandbox, '_build_logging_config', return_value={}) as mock_logging, \
         patch.object(sandbox, '_build_environment', return_value={}) as mock_env, \
         patch.object(sandbox, '_reuse_existing_named_container', new_callable=AsyncMock, return_value=None) as mock_named_reuse, \
         pytest.raises(SandboxRuntimeError, match="Max sessions"):
        
        await sandbox.start_container("new-session", persistence=True)


@pytest.mark.asyncio
async def test_start_session_with_named_container_reuse(sandbox_project_root: Path) -> None:
    """Test session start with named container reuse."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    await sandbox.ensure_directories()
    
    mock_container = Mock()
    
    with patch.object(sandbox, '_ensure_initialized', new_callable=AsyncMock), \
         patch.object(sandbox, '_reuse_tracked_container', new_callable=AsyncMock, return_value=None), \
         patch.object(sandbox, '_ensure_image_ready', new_callable=AsyncMock), \
         patch.object(sandbox, '_build_volume_mounts', return_value={}) as mock_volumes, \
         patch.object(sandbox, '_get_network_mode', return_value="none") as mock_network, \
         patch.object(sandbox, '_determine_container_parameters', return_value=("test-name", None)) as mock_params, \
         patch.object(sandbox, '_build_logging_config', return_value={}) as mock_logging, \
         patch.object(sandbox, '_build_environment', return_value={}) as mock_env, \
         patch.object(sandbox, '_reuse_existing_named_container', new_callable=AsyncMock, return_value=mock_container) as mock_named_reuse:
        
        result = await sandbox.start_container("test-session", persistence=True)
        
        assert result == mock_container
        mock_named_reuse.assert_called_once()


@pytest.mark.asyncio
async def test_start_session_container_creation_failure(sandbox_project_root: Path) -> None:
    """Test session start when container creation fails."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    await sandbox.ensure_directories()
    
    with patch.object(sandbox, '_ensure_initialized', new_callable=AsyncMock), \
         patch.object(sandbox, '_reuse_tracked_container', new_callable=AsyncMock, return_value=None), \
         patch.object(sandbox, '_ensure_image_ready', new_callable=AsyncMock), \
         patch.object(sandbox, '_build_volume_mounts', return_value={}) as mock_volumes, \
         patch.object(sandbox, '_get_network_mode', return_value="none") as mock_network, \
         patch.object(sandbox, '_determine_container_parameters', return_value=("test-name", None)) as mock_params, \
         patch.object(sandbox, '_build_logging_config', return_value={}) as mock_logging, \
         patch.object(sandbox, '_build_environment', return_value={}) as mock_env, \
         patch.object(sandbox, '_reuse_existing_named_container', new_callable=AsyncMock, return_value=None) as mock_named_reuse, \
         patch.object(sandbox, '_create_container', side_effect=Exception("Creation failed")) as mock_create:
        
        with pytest.raises(Exception, match="Creation failed"):
            await sandbox.start_container("test-session", persistence=False)
        
        assert "test-session" not in sandbox._containers


@pytest.mark.asyncio
async def test_pop_tracked_container_with_session_id(sandbox_project_root: Path) -> None:
    """Test popping tracked container with session ID."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    sandbox._containers["test-session"] = mock_container
    
    session_id, container = sandbox._pop_tracked_container("test-session", None)
    
    assert session_id == "test-session"
    assert container == mock_container
    assert "test-session" not in sandbox._containers


@pytest.mark.asyncio
async def test_pop_tracked_container_with_container_id(sandbox_project_root: Path) -> None:
    """Test popping tracked container with container ID lookup."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    mock_container.id = "container-123"
    sandbox._containers["test-session"] = mock_container
    
    with patch.object(sandbox, '_find_session_id_by_container', return_value="test-session") as mock_find:
        session_id, container = sandbox._pop_tracked_container(None, "container-123")
        
        assert session_id == "test-session"
        assert container == mock_container
        mock_find.assert_called_once_with("container-123")


@pytest.mark.asyncio
async def test_pop_tracked_container_not_found(sandbox_project_root: Path) -> None:
    """Test popping tracked container when not found."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    with patch.object(sandbox, '_find_session_id_by_container', return_value=None) as mock_find:
        session_id, container = sandbox._pop_tracked_container(None, "unknown-container")
        
        assert session_id is None
        assert container is None
        mock_find.assert_called_once_with("unknown-container")


@pytest.mark.asyncio
async def test_stop_container_once_not_found(sandbox_project_root: Path) -> None:
    """Test stopping container when container is not found."""
    import docker.errors
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    mock_container.reload.side_effect = docker.errors.NotFound("Not found")
    
    result = sandbox._stop_container_once(mock_container, "container-123", "test-session", False)
    
    assert result is True


@pytest.mark.asyncio
async def test_stop_container_once_already_exited(sandbox_project_root: Path) -> None:
    """Test stopping container when already exited."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    mock_container.status = "exited"
    mock_container.reload.return_value = None
    
    result = sandbox._stop_container_once(mock_container, "container-123", "test-session", True)
    
    assert result is True
    mock_container.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_stop_container_once_running_stop_only(sandbox_project_root: Path) -> None:
    """Test stopping running container without removal."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    mock_container.status = "running"
    mock_container.reload.return_value = None
    
    result = sandbox._stop_container_once(mock_container, "container-123", "test-session", False)
    
    assert result is True
    mock_container.stop.assert_called_once_with(timeout=5)
    mock_container.remove.assert_not_called()


@pytest.mark.asyncio
async def test_stop_container_once_running_stop_and_remove(sandbox_project_root: Path) -> None:
    """Test stopping and removing running container."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    mock_container.status = "running"
    mock_container.reload.return_value = None
    
    result = sandbox._stop_container_once(mock_container, "container-123", "test-session", True)
    
    assert result is True
    mock_container.stop.assert_called_once_with(timeout=5)
    mock_container.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_stop_container_once_reload_failure(sandbox_project_root: Path) -> None:
    """Test stopping container when reload fails."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    mock_container.reload.side_effect = Exception("Reload failed")
    
    result = sandbox._stop_container_once(mock_container, "container-123", "test-session", False)
    
    assert result is False


@pytest.mark.asyncio
async def test_stop_container_success(sandbox_project_root: Path) -> None:
    """Test successful container stopping."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    sandbox._containers["test-session"] = mock_container
    
    with patch.object(sandbox, '_ensure_initialized', new_callable=AsyncMock) as mock_init, \
         patch.object(sandbox, '_pop_tracked_container', return_value=("test-session", mock_container)) as mock_pop, \
         patch.object(sandbox, '_stop_container_once', return_value=True) as mock_stop_once:
        
        await sandbox.stop_container("container-123", remove=False, session_id="test-session", max_retries=3)
        
        mock_init.assert_called_once()
        mock_pop.assert_called_once_with("test-session", "container-123")
        mock_stop_once.assert_called_once_with(mock_container, "container-123", "test-session", False)


@pytest.mark.asyncio
async def test_stop_container_not_found(sandbox_project_root: Path) -> None:
    """Test stopping container when not found."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    with patch.object(sandbox, '_ensure_initialized', new_callable=AsyncMock) as mock_init, \
         patch.object(sandbox, '_pop_tracked_container', return_value=(None, None)) as mock_pop:
        
        await sandbox.stop_container("container-123", remove=False)
        
        mock_init.assert_called_once()
        mock_pop.assert_called_once_with("", "container-123")


@pytest.mark.asyncio
async def test_stop_container_with_retries(sandbox_project_root: Path) -> None:
    """Test stopping container with retries on conflict."""
    import docker.errors
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    sandbox._containers["test-session"] = mock_container
    
    # Create a proper APIError mock that will trigger retry
    api_error = Mock(spec=docker.errors.APIError)
    api_error.status_code = 409
    api_error.__str__ = lambda: "409 Client Error: Conflict (container removal already in progress)"
    
    with patch.object(sandbox, '_ensure_initialized', new_callable=AsyncMock), \
         patch.object(sandbox, '_pop_tracked_container', return_value=("test-session", mock_container)), \
         patch.object(sandbox, '_stop_container_once', side_effect=[api_error, True]) as mock_stop_once, \
         patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        
        await sandbox.stop_container("container-123", remove=False, session_id="test-session", max_retries=2)
        
        assert mock_stop_once.call_count == 2
        mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_stop_container_max_retries_exceeded(sandbox_project_root: Path) -> None:
    """Test stopping container when max retries exceeded."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    sandbox._containers["test-session"] = mock_container
    
    with patch.object(sandbox, '_ensure_initialized', new_callable=AsyncMock), \
         patch.object(sandbox, '_pop_tracked_container', return_value=("test-session", mock_container)), \
         patch.object(sandbox, '_stop_container_once', side_effect=Exception("Always fails")) as mock_stop_once, \
         patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        
        # The method should not raise an exception when max retries is exceeded
        await sandbox.stop_container("container-123", remove=False, session_id="test-session", max_retries=2)
        
        assert mock_stop_once.call_count == 2
        assert mock_sleep.call_count == 1
