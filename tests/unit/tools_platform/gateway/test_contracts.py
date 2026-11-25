"""Contract tests for tools_platform contracts.

Tests serialization, validation, and immutability of contract models.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ChangeSet,
    ConcurrencyScope,
    ConflictReport,
    ConflictRule,
    FileChange,
    FileObservation,
    LockLifetime,
    LockMode,
    OperationEffect,
    ResourceLock,
    ToolCallRequest,
    ToolDecision,
    ToolDecisionOutcome,
    ToolDefinition,
    ToolErrorCategory,
    ToolErrorCode,
    ToolExecutionError,
    ToolExecutionResult,
    ToolOutputChunk,
    ToolRiskClass,
)


class TestToolDefinition:
    """Test ToolDefinition contract."""

    def test_tool_definition_serialization(self):
        """Test ToolDefinition can be serialized and deserialized."""
        definition = ToolDefinition(
            id="test_tool",
            version="1.0.0",
            description="A test tool",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        # Serialize
        data = definition.model_dump()
        assert data["id"] == "test_tool"
        assert data["version"] == "1.0.0"

        # Deserialize
        restored = ToolDefinition.model_validate(data)
        assert restored.id == "test_tool"
        assert restored.version == "1.0.0"

    def test_tool_definition_immutability(self):
        """Test ToolDefinition is frozen."""
        definition = ToolDefinition(
            id="test_tool",
            version="1.0.0",
            description="A test tool",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        with pytest.raises(ValidationError):
            definition.id = "new_id"

    def test_tool_definition_with_all_fields(self):
        """Test ToolDefinition with all optional fields."""
        definition = ToolDefinition(
            id="test_tool",
            version="2.0.0",
            description="A test tool",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            permissions=["read", "write"],
            risk_class=ToolRiskClass.MEDIUM,
            effects=[OperationEffect.READ, OperationEffect.WRITE],
            supported_backends=[BackendType.HOST, BackendType.DOCKER],
            supports_streaming=True,
            concurrency_scope=ConcurrencyScope(
                effect=OperationEffect.WRITE,
                resource_locks=[ResourceLock(key="test", mode=LockMode.EXCLUSIVE)],
            ),
        )

        assert definition.id == "test_tool"
        assert definition.risk_class == ToolRiskClass.MEDIUM
        assert len(definition.effects) == 2
        assert len(definition.supported_backends) == 2


class TestToolCallRequest:
    """Test ToolCallRequest contract."""

    def test_tool_call_request_validation(self):
        """Test ToolCallRequest validates required fields."""
        # Valid request should succeed
        request = ToolCallRequest(
            call_id="test",
            session_id="session",
            frame_id="frame",
            tool_id="tool",
            raw_arguments={},
            workspace_id="workspace",
        )
        assert request.call_id == "test"

        # Missing required field should raise ValidationError
        with pytest.raises(ValidationError):
            ToolCallRequest(
                call_id="test",
                session_id="session",
                frame_id="frame",
                tool_id="tool",
                raw_arguments={},
                # workspace_id missing
            )

    def test_tool_call_request_serialization(self):
        """Test ToolCallRequest can be serialized."""
        request = ToolCallRequest(
            call_id="test_call",
            session_id="session_1",
            frame_id="frame_1",
            tool_id="test_tool",
            raw_arguments={"arg1": "value1"},
            workspace_id="workspace_1",
            requested_backend=BackendType.HOST,
        )

        data = request.model_dump()
        assert data["call_id"] == "test_call"
        assert data["tool_id"] == "test_tool"
        assert data["requested_backend"] == "host"

        restored = ToolCallRequest.model_validate(data)
        assert restored.call_id == "test_call"


class TestToolExecutionError:
    """Test ToolExecutionError contract."""

    def test_tool_execution_error_serialization(self):
        """Test ToolExecutionError can be serialized."""
        error = ToolExecutionError(
            code=ToolErrorCode.INVALID_ARGUMENTS,
            category=ToolErrorCategory.VALIDATION,
            message="Invalid arguments provided",
            retryable=False,
            remediation="Check your arguments",
            details={"field": "value"},
        )

        data = error.model_dump()
        assert data["code"] == "invalid_arguments"
        assert data["category"] == "validation"

        restored = ToolExecutionError.model_validate(data)
        assert restored.code == ToolErrorCode.INVALID_ARGUMENTS

    def test_tool_execution_error_codes(self):
        """Test all error codes are valid."""
        for code in ToolErrorCode:
            error = ToolExecutionError(
                code=code,
                category=ToolErrorCategory.UNKNOWN,
                message="Test",
            )
            assert error.code == code

    def test_tool_execution_error_categories(self):
        """Test all error categories are valid."""
        for category in ToolErrorCategory:
            error = ToolExecutionError(
                code=ToolErrorCode.INTERNAL_FAILURE,
                category=category,
                message="Test",
            )
            assert error.category == category


class TestToolExecutionResult:
    """Test ToolExecutionResult contract."""

    def test_tool_execution_result_success(self):
        """Test successful execution result."""
        result = ToolExecutionResult(
            status="completed",
            output={"result": "success"},
            backend=BackendType.HOST,
            timing={"total": 1.5},
        )

        assert result.status == "completed"
        assert result.output == {"result": "success"}
        assert result.error is None

    def test_tool_execution_result_error(self):
        """Test failed execution result."""
        error = ToolExecutionError(
            code=ToolErrorCode.TIMEOUT,
            category=ToolErrorCategory.RESOURCE,
            message="Timeout",
        )
        result = ToolExecutionResult(
            status="failed",
            error=error,
            backend=BackendType.HOST,
        )

        assert result.status == "failed"
        assert result.error is not None
        assert result.error.code == ToolErrorCode.TIMEOUT

    def test_tool_execution_result_serialization(self):
        """Test ToolExecutionResult can be serialized."""
        result = ToolExecutionResult(
            status="completed",
            output={"data": "value"},
            backend=BackendType.DOCKER,
        )

        data = result.model_dump()
        assert data["status"] == "completed"
        assert data["backend"] == "docker"

        restored = ToolExecutionResult.model_validate(data)
        assert restored.status == "completed"


class TestToolOutputChunk:
    """Test ToolOutputChunk contract."""

    def test_tool_output_chunk_creation(self):
        """Test ToolOutputChunk creation."""
        chunk = ToolOutputChunk(
            stream_id="stream_1",
            sequence=0,
            channel="stdout",
            payload="output text",
        )

        assert chunk.stream_id == "stream_1"
        assert chunk.sequence == 0
        assert chunk.channel == "stdout"
        assert chunk.payload == "output text"

    def test_tool_output_chunk_serialization(self):
        """Test ToolOutputChunk can be serialized."""
        chunk = ToolOutputChunk(
            stream_id="stream_1",
            sequence=1,
            channel="stderr",
            payload="error text",
            cumulative_bytes=100,
        )

        data = chunk.model_dump()
        assert data["sequence"] == 1
        assert data["cumulative_bytes"] == 100

        restored = ToolOutputChunk.model_validate(data)
        assert restored.sequence == 1


class TestWorkspaceModels:
    """Test workspace-related models."""

    def test_file_observation(self):
        """Test FileObservation model."""
        observation = FileObservation(
            path="test.py",
            hash="abc123",
            size=1024,
            mtime=datetime.utcnow(),
        )

        assert observation.path == "test.py"
        assert observation.hash == "abc123"
        assert observation.size == 1024

    def test_file_change(self):
        """Test FileChange model."""
        change = FileChange(
            path="test.py",
            before_hash="old123",
            after_hash="new456",
            before_size=100,
            after_size=200,
            change_type="modify",
        )

        assert change.path == "test.py"
        assert change.change_type == "modify"

    def test_change_set(self):
        """Test ChangeSet model."""
        changeset = ChangeSet(
            changes=[
                FileChange(path="file1.py", change_type="modify"),
                FileChange(path="file2.py", change_type="create"),
            ],
            total_bytes_changed=300,
            total_files_changed=2,
        )

        assert len(changeset.changes) == 2
        assert changeset.total_files_changed == 2

    def test_conflict_report(self):
        """Test ConflictReport model."""
        report = ConflictReport(
            path="test.py",
            base_hash="base123",
            current_hash="current456",
            proposed_hash="proposed789",
            conflict_type="content_modified",
            resolution_options=["overwrite", "rebase"],
        )

        assert report.path == "test.py"
        assert len(report.resolution_options) == 2


class TestConcurrencyScope:
    """Test ConcurrencyScope model."""

    def test_concurrency_scope_creation(self):
        """Test ConcurrencyScope creation."""
        scope = ConcurrencyScope(
            effect=OperationEffect.WRITE,
            resource_locks=[
                ResourceLock(key="file1.py", mode=LockMode.EXCLUSIVE),
                ResourceLock(key="file2.py", mode=LockMode.EXCLUSIVE),
            ],
            capacity_group="writes",
            conflict_rule=ConflictRule.REJECT,
            lifetime=LockLifetime.COMMIT,
        )

        assert scope.effect == OperationEffect.WRITE
        assert len(scope.resource_locks) == 2
        assert scope.resource_locks[0].mode == LockMode.EXCLUSIVE

    def test_concurrency_scope_defaults(self):
        """Test ConcurrencyScope default values."""
        scope = ConcurrencyScope(effect=OperationEffect.READ)

        assert scope.resource_locks == []
        assert scope.lifetime == LockLifetime.EXECUTION


class TestToolDecision:
    """Test ToolDecision model."""

    def test_tool_decision_allow(self):
        """Test allow decision."""
        decision = ToolDecision(
            outcome=ToolDecisionOutcome.ALLOW,
            reason="Tool is safe",
        )

        assert decision.outcome == ToolDecisionOutcome.ALLOW
        assert decision.confirmation_token is None

    def test_tool_decision_ask(self):
        """Test ask decision with confirmation token."""
        decision = ToolDecision(
            outcome=ToolDecisionOutcome.ASK,
            reason="User confirmation required",
            confirmation_token="token123",
        )

        assert decision.outcome == ToolDecisionOutcome.ASK
        assert decision.confirmation_token == "token123"

    def test_tool_decision_serialization(self):
        """Test ToolDecision can be serialized."""
        decision = ToolDecision(
            outcome=ToolDecisionOutcome.DENY,
            reason="Tool is denied",
            policy_rule="no_write",
        )

        data = decision.model_dump()
        assert data["outcome"] == "deny"

        restored = ToolDecision.model_validate(data)
        assert restored.outcome == ToolDecisionOutcome.DENY
