"""Race condition tests for user edits during agent transactions."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.tools_platform.backends.host import HostBackend
from local_coding_assistant.tools_platform.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_root = Path(tmpdir)
        yield workspace_root


@pytest.fixture
def config_manager(temp_workspace):
    """Create a config manager for testing."""
    # Create a temporary config file that sets the workspace root
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("tools_platform:\n")
        f.write('  tool_policies_path: "@project/config/tool_policies.yaml"\n')
        temp_config_path = f.name

    try:
        cm = ConfigManager(config_paths=[temp_config_path])
        cm.load_global_config()

        # Set session override for the temp workspace
        cm.set_session_overrides({"tools_platform.workspace_root": str(temp_workspace)})

        yield cm
    finally:
        # Clean up temp config file
        try:
            os.unlink(temp_config_path)
        except OSError:
            pass


@pytest.fixture
def workspace_manager(config_manager):
    """Create a workspace manager for testing using config manager."""
    return WorkspaceManager.from_config_sync(config_manager=config_manager)


@pytest.fixture
def host_backend(temp_workspace, workspace_manager, config_manager):
    """Create a host backend for testing."""
    return HostBackend(
        workspace_root=temp_workspace,
        workspace_manager=workspace_manager,
        config_manager=config_manager,
    )


class TestRaceConditions:
    """Test race conditions between user and agent edits."""

    @pytest.mark.asyncio
    async def test_user_edit_during_agent_write(self, host_backend, temp_workspace):
        """Test that user edit during agent write is detected as conflict."""
        # Create a test file
        test_file = temp_workspace / "test.txt"
        with test_file.open(mode="w", encoding="utf-8", newline="") as f:
            f.write("original content")

        # Agent reads the file to populate baseline
        base_observation = host_backend._workspace_manager.observe_file(
            "test.txt", session_id="test_session"
        )

        # Simulate user edit after agent read
        with test_file.open(mode="w", encoding="utf-8", newline="") as f:
            f.write("user modified content")

        # Try to perform agent write (should detect conflict against baseline)
        result = await host_backend.replace_exact(
            path="test.txt",
            old_text="original",
            new_text="agent modified",
            expected_match_count=1,
            session_id="test_session",
        )

        # The agent's read version is stale, so the write must be rejected.
        assert result["success"] is False
        assert result["error"] is not None
        assert "File modified externally" in result["error"]
        assert result["conflict"]["base_hash"] == base_observation.hash
        assert result["conflict"]["current_hash"] != base_observation.hash

        # File should have user's content (agent write was rejected)
        assert test_file.read_text() == "user modified content"

    @pytest.mark.asyncio
    async def test_concurrent_agent_writes_serialize(
        self, host_backend, temp_workspace
    ):
        """Test that concurrent agent writes to the same file serialize."""
        # Create a test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        results = []

        async def write_attempt(text: str):
            """Attempt a write."""
            try:
                result = await host_backend.replace_exact(
                    path="test.txt",
                    old_text="original",
                    new_text=text,
                    expected_match_count=1,
                    session_id="test_session",
                )
                results.append(result)
            except Exception as e:
                results.append({"success": False, "error": str(e)})

        # Attempt concurrent writes (they should serialize)
        import asyncio

        await asyncio.gather(
            write_attempt("first"),
            write_attempt("second"),
        )

        # One should succeed, one should fail
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        assert len(successful) == 1
        assert len(failed) == 1

    @pytest.mark.asyncio
    async def test_user_edit_between_read_and_write(self, host_backend, temp_workspace):
        """Test detection of user edit between read and write phases."""
        # Create a test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")

        # Agent reads the file to populate baseline
        host_backend._workspace_manager.observe_file(
            "test.txt", session_id="test_session"
        )

        # Simulate user edit after read but before write
        test_file.write_text("user modified")

        # Agent tries to write based on stale read
        result = await host_backend.replace_exact(
            path="test.txt",
            old_text="original",
            new_text="agent modified",
            expected_match_count=1,
            session_id="test_session",
        )

        # Should detect conflict
        assert result["success"] is False
        assert "File modified externally" in result["error"]

    @pytest.mark.asyncio
    async def test_agent_write_with_conflict_detection_enabled(
        self, host_backend, temp_workspace
    ):
        """Test that conflict detection prevents destructive writes."""
        # Create a test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        # Agent reads the file to populate baseline
        host_backend._workspace_manager.observe_file(
            "test.txt", session_id="test_session"
        )

        # User modifies file
        test_file.write_text("user version")

        # Agent tries to write with conflict detection
        result = await host_backend.replace_exact(
            path="test.txt",
            old_text="original",
            new_text="agent version",
            expected_match_count=1,
            session_id="test_session",
        )

        # Should fail
        assert result["success"] is False
        assert "File modified externally" in result["error"]

        # Verify user's version is preserved
        assert test_file.read_text() == "user version"

    @pytest.mark.asyncio
    async def test_multiple_edits_in_same_session(self, host_backend, temp_workspace):
        """Test multiple edits in a single session."""
        # Create a test file
        test_file = temp_workspace / "test.txt"
        with test_file.open(mode="w", encoding="utf-8", newline="") as f:
            f.write("line1\nline2\nline3\n")

        # Agent reads the file to populate baseline
        host_backend._workspace_manager.observe_file(
            "test.txt", session_id="test_session"
        )

        # First edit
        result1 = await host_backend.replace_exact(
            path="test.txt",
            old_text="line1",
            new_text="modified1",
            expected_match_count=1,
            session_id="test_session",
        )
        assert result1["success"] is True

        # Second edit (on the modified content - baseline is updated after first write)
        result2 = await host_backend.replace_exact(
            path="test.txt",
            old_text="modified1",
            new_text="modified2",
            expected_match_count=1,
            session_id="test_session",
        )
        assert result2["success"] is True

        # Verify final state
        assert test_file.read_text() == "modified2\nline2\nline3\n"

    @pytest.mark.asyncio
    async def test_user_delete_during_agent_session(self, host_backend, temp_workspace):
        """Test handling of user file deletion during agent session."""
        from local_coding_assistant.tools_platform.exceptions import (
            ToolPathViolationError,
        )

        # Create a test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        # Agent reads the file to populate baseline
        host_backend._workspace_manager.observe_file(
            "test.txt", session_id="test_session"
        )

        # User deletes the file
        test_file.unlink()

        # Agent tries to write - should raise exception
        with pytest.raises(ToolPathViolationError, match="File not found"):
            await host_backend.replace_exact(
                path="test.txt",
                old_text="content",
                new_text="modified",
                expected_match_count=1,
                session_id="test_session",
            )

    @pytest.mark.asyncio
    async def test_baseline_update_after_conflict(self, host_backend, temp_workspace):
        """Test that baseline is updated after conflict detection."""
        # Create a test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")

        # Agent reads the file to populate baseline
        host_backend._workspace_manager.observe_file(
            "test.txt", session_id="test_session"
        )

        # Perform a write
        result1 = await host_backend.replace_exact(
            path="test.txt",
            old_text="original",
            new_text="first modification",
            expected_match_count=1,
            session_id="test_session",
        )
        assert result1["success"] is True

        # User modifies file externally
        test_file.write_text("user external edit")

        # Agent tries another write (should detect conflict)
        result2 = await host_backend.replace_exact(
            path="test.txt",
            old_text="first modification",
            new_text="second modification",
            expected_match_count=1,
            session_id="test_session",
        )
        assert result2["success"] is False
        assert "File modified externally" in result2["error"]

        # After conflict detection, the baseline should be updated to current disk state
        # so a retry with the correct content would work
        result3 = await host_backend.replace_exact(
            path="test.txt",
            old_text="user external",
            new_text="agent retry",
            expected_match_count=1,
            session_id="test_session",
        )
        assert result3["success"] is True
        assert test_file.read_text() == "agent retry edit"

    @pytest.mark.asyncio
    async def test_conflict_with_expected_hash(self, host_backend, temp_workspace):
        """Test conflict detection against the read baseline."""

        # Create a test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        # Populate the read baseline
        base_obs = host_backend._workspace_manager.observe_file(
            "test.txt", session_id="test_session"
        )

        # User modifies file
        test_file.write_text("modified")

        # Agent tries to write → conflict detected against read baseline
        result = await host_backend.replace_exact(
            path="test.txt",
            old_text="original",
            new_text="agent version",
            expected_match_count=1,
            session_id="test_session",
        )

        # Should fail with a structured stale-version conflict
        assert result["success"] is False
        assert "File modified externally" in result["error"]
        assert result["conflict"]["base_hash"] == base_obs.hash

        # Verify user's version is preserved
        assert test_file.read_text() == "modified"
