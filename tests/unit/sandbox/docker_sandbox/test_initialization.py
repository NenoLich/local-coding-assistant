"""Unit tests for DockerSandbox initialization and Docker availability."""

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


@pytest.fixture()
def sandbox_project_root(tmp_path: Path) -> Path:
    """Provide a temporary project root with expected sub-structure."""
    (tmp_path / "src" / "local_coding_assistant" / "sandbox" / "guest").mkdir(
        parents=True, exist_ok=True
    )
    return tmp_path


@pytest.mark.asyncio
async def test_check_docker_available_success(sandbox_project_root: Path) -> None:
    """Test successful Docker availability check."""
    import docker
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_client = Mock()
    mock_client.ping.return_value = True
    mock_client.version.return_value = {"Version": "20.10.0", "ApiVersion": "1.41"}
    mock_client.close.return_value = None
    
    with patch('docker.from_env', return_value=mock_client):
        result = sandbox.check_availability()
        
    assert result is True
    mock_client.ping.assert_called_once()
    mock_client.version.assert_called_once()
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_check_docker_available_no_docker_sdk(sandbox_project_root: Path) -> None:
    """Test Docker availability check when SDK is not installed."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    with patch('local_coding_assistant.sandbox.docker_sandbox.docker', None):
        result = sandbox.check_availability()
        
    assert result is False


@pytest.mark.asyncio
async def test_check_docker_available_connection_failure(sandbox_project_root: Path) -> None:
    """Test Docker availability check when connection fails."""
    import docker
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_client = Mock()
    mock_client.ping.side_effect = Exception("Connection failed")
    mock_client.close.return_value = None
    
    with patch('docker.from_env', return_value=mock_client):
        result = sandbox.check_availability()
        
    assert result is False
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_check_docker_available_client_close_error(sandbox_project_root: Path) -> None:
    """Test Docker availability check when client close fails."""
    import docker
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_client = Mock()
    mock_client.ping.return_value = True
    mock_client.version.return_value = {"Version": "20.10.0"}
    mock_client.close.side_effect = Exception("Close failed")
    
    with patch('docker.from_env', return_value=mock_client):
        result = sandbox.check_availability()
        
    assert result is True


@pytest.mark.asyncio
async def test_discover_existing_containers_success(sandbox_project_root: Path) -> None:
    """Test successful discovery of existing containers."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root), image="test-image")
    
    mock_container1 = Mock()
    mock_container1.name = "locca-sandbox-session1"
    mock_container1.id = "container1-id"
    
    mock_container2 = Mock()
    mock_container2.name = "locca-sandbox-session2"
    mock_container2.id = "container2-id"
    
    mock_container3 = Mock()
    mock_container3.name = "other-container"
    mock_container3.id = "container3-id"
    
    mock_client = Mock()
    mock_client.containers.list.return_value = [mock_container1, mock_container2, mock_container3]
    sandbox._client = mock_client
    
    await sandbox._discover_existing_containers()
    
    assert "session1" in sandbox._containers
    assert "session2" in sandbox._containers
    assert sandbox._containers["session1"] == mock_container1
    assert sandbox._containers["session2"] == mock_container2
    assert "other-container" not in sandbox._containers


@pytest.mark.asyncio
async def test_discover_existing_containers_no_client(sandbox_project_root: Path) -> None:
    """Test container discovery when client is not initialized."""
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    sandbox._client = None
    
    await sandbox._discover_existing_containers()
    
    assert len(sandbox._containers) == 0


@pytest.mark.asyncio
async def test_discover_existing_containers_exception(sandbox_project_root: Path) -> None:
    """Test container discovery when listing fails."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_client = Mock()
    mock_client.containers.list.side_effect = Exception("List failed")
    sandbox._client = mock_client
    
    await sandbox._discover_existing_containers()
    
    assert len(sandbox._containers) == 0


@pytest.mark.asyncio
async def test_initialize_sandbox_success(sandbox_project_root: Path) -> None:
    """Test successful sandbox initialization."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    with patch.object(sandbox, '_ensure_client', new_callable=AsyncMock) as mock_ensure_client, \
         patch.object(sandbox, '_discover_existing_containers', new_callable=AsyncMock) as mock_discover, \
         patch.object(sandbox, 'ensure_directories', new_callable=AsyncMock) as mock_dirs:
        
        mock_ensure_client.return_value = None
        mock_discover.return_value = None
        mock_dirs.return_value = None
        
        await sandbox.initialize()
        
        mock_ensure_client.assert_called_once()
        mock_discover.assert_called_once()
        mock_dirs.assert_called_once()
        assert sandbox._initialized is True


@pytest.mark.asyncio
async def test_initialize_sandbox_already_initialized(sandbox_project_root: Path) -> None:
    """Test sandbox initialization when already initialized."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    sandbox._initialized = True
    
    with patch.object(sandbox, '_ensure_client', new_callable=AsyncMock) as mock_ensure_client, \
         patch.object(sandbox, '_discover_existing_containers', new_callable=AsyncMock) as mock_discover, \
         patch.object(sandbox, 'ensure_directories', new_callable=AsyncMock) as mock_dirs:
        
        await sandbox.initialize()
        
        mock_ensure_client.assert_not_called()
        mock_discover.assert_not_called()
        mock_dirs.assert_not_called()


@pytest.mark.asyncio
async def test_initialize_sandbox_failure(sandbox_project_root: Path) -> None:
    """Test sandbox initialization when an error occurs."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    with patch.object(sandbox, '_ensure_client', new_callable=AsyncMock) as mock_ensure_client:
        mock_ensure_client.side_effect = Exception("Init failed")
        
        with pytest.raises(Exception, match="Init failed"):
            await sandbox.initialize()
        
        assert sandbox._initialized is False


@pytest.mark.asyncio
async def test_ensure_client_success(sandbox_project_root: Path) -> None:
    """Test successful Docker client initialization."""
    import docker
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_client = Mock()
    mock_client.ping.return_value = True
    
    with patch('docker.from_env', return_value=mock_client):
        await sandbox._ensure_client()
        
    assert sandbox._client == mock_client
    mock_client.ping.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_client_already_initialized(sandbox_project_root: Path) -> None:
    """Test client initialization when already initialized."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    existing_client = Mock()
    sandbox._client = existing_client
    
    await sandbox._ensure_client()
    
    assert sandbox._client == existing_client


@pytest.mark.asyncio
async def test_ensure_client_failure(sandbox_project_root: Path) -> None:
    """Test client initialization when Docker is not available."""
    import docker
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_client = Mock()
    mock_client.ping.side_effect = Exception("Docker not running")
    
    with patch('docker.from_env', return_value=mock_client):
        with pytest.raises(SandboxRuntimeError, match="Docker is not running or not installed"):
            await sandbox._ensure_client()
