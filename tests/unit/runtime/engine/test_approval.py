"""Unit tests for approval request."""

from __future__ import annotations

import time

from local_coding_assistant.runtime.engine.approval import (
    ApprovalRequest,
    ApprovalStatus,
)
from local_coding_assistant.tools_platform.contracts import ApprovalResponse


class TestApprovalRequest:
    """Tests for ApprovalRequest."""

    def test_initial_state(self) -> None:
        request = ApprovalRequest(
            token="test_token",
            tool_call_id="call_123",
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            risk_class="low",
            session_id="session_123",
            frame_id="frame_123",
            turn_id="turn_123",
        )
        assert request.token == "test_token"
        assert request.status == ApprovalStatus.PENDING
        assert request.is_pending is True
        assert request.is_expired is False

    def test_expiration(self) -> None:
        request = ApprovalRequest(
            token="test_token",
            tool_call_id="call_123",
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            risk_class="low",
            session_id="session_123",
            frame_id="frame_123",
            turn_id="turn_123",
            timeout_seconds=0,  # Immediate expiration
        )
        time.sleep(0.001)  # Ensure expiration
        assert request.is_expired is True
        assert request.is_pending is False

    def test_approval_response_granted(self) -> None:
        request = ApprovalRequest(
            token="test_token",
            tool_call_id="call_123",
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            risk_class="low",
            session_id="session_123",
            frame_id="frame_123",
            turn_id="turn_123",
        )
        request.response = ApprovalResponse.YES_ONCE
        request.status = ApprovalStatus.GRANTED
        assert request.response == ApprovalResponse.YES_ONCE
        assert request.status == ApprovalStatus.GRANTED

    def test_approval_response_denied(self) -> None:
        request = ApprovalRequest(
            token="test_token",
            tool_call_id="call_123",
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            risk_class="low",
            session_id="session_123",
            frame_id="frame_123",
            turn_id="turn_123",
        )
        request.response = ApprovalResponse.NO
        request.status = ApprovalStatus.DENIED
        assert request.response == ApprovalResponse.NO
        assert request.status == ApprovalStatus.DENIED

    def test_is_pending_after_grant(self) -> None:
        request = ApprovalRequest(
            token="test_token",
            tool_call_id="call_123",
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            risk_class="low",
            session_id="session_123",
            frame_id="frame_123",
            turn_id="turn_123",
        )
        request.status = ApprovalStatus.GRANTED
        assert request.is_pending is False

    def test_is_pending_after_deny(self) -> None:
        request = ApprovalRequest(
            token="test_token",
            tool_call_id="call_123",
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            risk_class="low",
            session_id="session_123",
            frame_id="frame_123",
            turn_id="turn_123",
        )
        request.status = ApprovalStatus.DENIED
        assert request.is_pending is False

    def test_is_pending_after_expire(self) -> None:
        request = ApprovalRequest(
            token="test_token",
            tool_call_id="call_123",
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            risk_class="low",
            session_id="session_123",
            frame_id="frame_123",
            turn_id="turn_123",
            timeout_seconds=0,
        )
        time.sleep(0.001)
        assert request.is_expired is True
        assert request.is_pending is False
