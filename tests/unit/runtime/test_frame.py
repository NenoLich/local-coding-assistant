"""Tests for the new execution data structures."""

import pytest

from local_coding_assistant.repository.file_monitoring import FileChangeType
from local_coding_assistant.runtime.engine.frame import (
    ActionKind,
    ActionRecord,
    ExecutionResult,
    ExecutionStatus,
    LLMMetrics,
    RuntimeFrame,
)
from local_coding_assistant.runtime.types import ExecutionMode
from local_coding_assistant.tools_platform.contracts import (
    FileChange,
    ToolExecutionResult,
)


class TestActionRecord:
    """Test ActionRecord with new data structure."""

    def test_action_record_creation(self):
        """Test basic ActionRecord creation."""
        action = ActionRecord(kind=ActionKind.LLM_MESSAGE, name="test_action")

        assert action.kind == ActionKind.LLM_MESSAGE
        assert action.name == "test_action"
        assert action.llm_metrics is None
        assert action.tool_execution_data == {}
        assert action.started_at is not None
        assert action.finished_at is None

    def test_action_record_with_llm_metrics(self):
        """Test ActionRecord with LLM metrics."""
        llm_metrics = LLMMetrics(
            prompt_tokens=10,
            completion_tokens=20,
            cached_tokens=30,
            cache_write_tokens=5,
            reasoning_tokens=5,
            latency_ms=100.0,
            model="test-model",
        )

        action = ActionRecord(
            kind=ActionKind.LLM_MESSAGE, name="llm_test", llm_metrics=llm_metrics
        )

        assert action.llm_metrics == llm_metrics
        assert action.llm_metrics.model == "test-model"
        assert action.llm_metrics.prompt_tokens == 10
        assert action.llm_metrics.completion_tokens == 20
        assert action.llm_metrics.cached_tokens == 30
        assert action.llm_metrics.cache_write_tokens == 5
        assert action.llm_metrics.reasoning_tokens == 5

    def test_action_record_with_tool_trace(self):
        """Test ActionRecord with tool execution data."""
        tool_execution_data = {
            "call_id": "test-call-123",
            "tool_id": "test_tool",
            "status": "completed",
            "input": {"param1": "value1"},
            "output": "result",
            "duration_ms": 50.0,
        }

        action = ActionRecord(
            kind=ActionKind.TOOL_CALL,
            name="test_tool",
            tool_execution_data=tool_execution_data,
        )

        assert action.tool_execution_data == tool_execution_data
        assert action.tool_execution_data["tool_id"] == "test_tool"
        assert action.tool_execution_data["status"] == "completed"

    def test_tool_execution_data_properties(self):
        """Test tool_execution_data properties."""
        tool_execution_data = {
            "call_id": "test-call-123",
            "tool_id": "test_tool",
            "status": "completed",
            "input": {"param1": "value1"},
            "output": "result",
            "duration_ms": 50.0,
        }

        action = ActionRecord(
            kind=ActionKind.TOOL_CALL,
            name="test_tool",
            tool_execution_data=tool_execution_data,
        )

        # Test tool_execution_data properties
        assert action.tool_execution_data["input"] == {"param1": "value1"}
        assert action.tool_execution_data["output"] == "result"

        # Test non-tool action has empty tool_execution_data
        llm_action = ActionRecord(kind=ActionKind.LLM_MESSAGE)
        assert llm_action.tool_execution_data == {}


class TestExecutionResult:
    """Test ExecutionResult with simplified structure."""

    def test_execution_result_creation(self):
        """Test basic ExecutionResult creation."""
        result = ExecutionResult(final_answer="Test answer")

        assert result.final_answer == "Test answer"
        assert result.total_latency_ms is None
        assert result.agent_file_changes == []

    def test_execution_result_with_file_operations(self):
        """Test ExecutionResult with file operations."""
        file_change_created = FileChange(
            path="/path/to/new_file.txt", change_type=FileChangeType.CREATED
        )
        file_change_modified = FileChange(
            path="/path/to/modified_file.txt", change_type=FileChangeType.MODIFIED
        )
        result = ExecutionResult(
            agent_file_changes=[file_change_created, file_change_modified],
        )

        assert len(result.agent_file_changes) == 2
        assert "/path/to/new_file.txt" in result.agent_file_changes[0].path
        assert result.agent_file_changes[0].change_type == FileChangeType.CREATED
        assert "/path/to/modified_file.txt" in result.agent_file_changes[1].path
        assert result.agent_file_changes[1].change_type == FileChangeType.MODIFIED


class TestExecutionFrame:
    """Test ExecutionFrame with on-demand data access."""

    @pytest.fixture
    def mock_prompt_context(self):
        """Create a mock prompt context."""
        from local_coding_assistant.runtime.types import PromptContext

        return PromptContext(
            session_id="test-session",
            execution_mode=ExecutionMode.SANDBOX_PYTHON,
            tool_call_mode="ptc",
            user_input="Test input",
            tools=[],
            history=[],
        )

    @pytest.fixture
    def mock_rendered_prompt(self):
        """Create a mock rendered prompt."""
        from local_coding_assistant.runtime.types import RenderedPrompt

        rendered = RenderedPrompt(
            system_messages=["System message"],
            user_messages=["User message"],
            tool_schemas=[],
        )
        return rendered

    def test_execution_frame_creation(self, mock_prompt_context, mock_rendered_prompt):
        """Test basic ExecutionFrame creation with new API."""
        frame = RuntimeFrame(
            session_id="test-session",
            turn_id="test-session_turn_1",
            trace_id="test-trace",
            status=ExecutionStatus.RUNNING_LLM,
        )

        assert frame.session_id == "test-session"
        assert frame.turn_id == "test-session_turn_1"
        assert frame.trace_id == "test-trace"
        assert frame.iteration == 1
        assert frame.actions == []
        assert frame.result is None
        assert frame.started_at is not None
        assert frame.finished_at is None

    def test_add_action(self, mock_prompt_context, mock_rendered_prompt):
        """Test adding actions to frame."""
        frame = RuntimeFrame(
            session_id="test-session",
            turn_id="test-session_turn_1",
            trace_id="test-trace",
            status=ExecutionStatus.RUNNING_LLM,
        )

        # Add LLM action
        llm_action = frame.add_action(ActionKind.LLM_MESSAGE, name="llm_test")
        assert llm_action.kind == ActionKind.LLM_MESSAGE
        assert llm_action.name == "llm_test"
        assert len(frame.actions) == 1

        # Add tool action with input
        tool_action = frame.add_action(
            ActionKind.TOOL_CALL, name="test_tool", _input={"param": "value"}
        )
        assert tool_action.kind == ActionKind.TOOL_CALL
        assert tool_action.tool_execution_data["input"] == {"param": "value"}
        assert len(frame.actions) == 2

    def test_complete_llm_action(self, mock_prompt_context, mock_rendered_prompt):
        """Test completing an LLM action."""
        frame = RuntimeFrame(
            session_id="test-session",
            turn_id="test-session_turn_1",
            trace_id="test-trace",
            status=ExecutionStatus.RUNNING_LLM,
        )

        action = frame.add_action(ActionKind.LLM_MESSAGE, name="llm_test")

        metadata = {
            "model": "test-model",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "reasoning_tokens": 5,
            "latency_ms": 100.0,
            "reasoning": "Test reasoning",
        }

        frame.complete_action(action.id, "Test response", metadata)

        assert action.finished_at is not None
        assert action.llm_metrics is not None
        assert action.llm_metrics.model == "test-model"
        assert action.llm_metrics.reasoning_tokens == 5
        assert action.metadata["reasoning"] == "Test reasoning"

    def test_complete_tool_action_with_result(self):
        """Test completing a tool action with ToolExecutionResult."""
        frame = RuntimeFrame(
            session_id="test-session",
            turn_id="test-session_turn_1",
            trace_id="test-trace",
            status=ExecutionStatus.RUNNING_LLM,
        )

        action = frame.add_action(
            ActionKind.TOOL_CALL, name="test_tool", _input={"param": "value"}
        )

        tool_result = ToolExecutionResult(
            status="completed",
            output="result",
            timing={"elapsed_seconds": 0.05},
        )

        frame.complete_action(
            action.id, tool_result, {"call_id": "test-call-123", "tool_id": "test_tool"}
        )

        assert action.finished_at is not None
        data = action.tool_execution_data
        assert data["tool_id"] == "test_tool"
        assert data["status"] == "completed"
        assert data["output"] == "result"
        assert data["call_id"] == "test-call-123"
        assert data["input"] == {"param": "value"}

    def test_complete_tool_action_legacy(
        self, mock_prompt_context, mock_rendered_prompt
    ):
        """Test completing a tool action with legacy metadata."""
        frame = RuntimeFrame(
            session_id="test-session",
            turn_id="test-session_turn_1",
            trace_id="test-trace",
            status=ExecutionStatus.RUNNING_LLM,
        )

        action = frame.add_action(
            ActionKind.TOOL_CALL, name="test_tool", _input={"param": "value"}
        )

        metadata = {
            "tool_id": "test_tool",
            "call_id": "test-call-123",
            "error": None,
            "parent_call_id": "parent-123",
        }

        frame.complete_action(action.id, "result", metadata)

        assert action.finished_at is not None
        data = action.tool_execution_data
        assert data["tool_id"] == "test_tool"
        assert data["status"] == "completed"
        assert data["output"] == "result"
        assert data["error"] is None
        assert data["input"] == {"param": "value"}
        assert data["timing"] == {}

    def test_get_tool_results_on_demand(
        self, mock_prompt_context, mock_rendered_prompt
    ):
        """Test on-demand tool results generation."""
        frame = RuntimeFrame(
            session_id="test-session",
            turn_id="test-session_turn_1",
            trace_id="test-trace",
            status=ExecutionStatus.RUNNING_LLM,
        )

        # Add tool actions with tool_execution_data
        action1 = frame.add_action(
            ActionKind.TOOL_CALL, name="tool1", _input={"param1": "value1"}
        )
        action1.tool_execution_data = {
            "tool_id": "tool1",
            "call_id": "call-1",
            "status": "completed",
            "output": "result1",
            "input": {"param1": "value1"},
            "error": None,
            "timing": {"elapsed_seconds": 0.05},
            "decision": {},
            "metadata": {},
        }

        action2 = frame.add_action(
            ActionKind.TOOL_CALL, name="tool2", _input={"param2": "value2"}
        )
        action2.tool_execution_data = {
            "tool_id": "tool2",
            "call_id": "call-2",
            "status": "completed",
            "output": "result2",
            "input": {"param2": "value2"},
            "error": None,
            "timing": {"elapsed_seconds": 0.03},
            "decision": {"parent_call_id": "call-1"},
            "metadata": {},
        }

        # Test on-demand access
        tool_results = frame.get_tool_results()

        assert len(tool_results) == 2

        # Check first tool result
        result1 = next(r for r in tool_results if r["tool_name"] == "tool1")
        assert result1["tool_args"] == {"param1": "value1"}
        assert result1["result"] == "result1"
        assert result1["success"] is True
        assert result1["child_call_ids"] == []
        assert result1["call_id"] == "call-1"
        assert result1["parent_call_id"] is None
        assert result1["execution_mode"] is None
        assert result1["source"] == "tools_platform"
        assert result1["resource_metrics"] == []

        # Check second tool result
        result2 = next(r for r in tool_results if r["tool_name"] == "tool2")
        assert result2["tool_args"] == {"param2": "value2"}
        assert result2["result"] == "result2"
        assert result2["success"] is True
        assert result2["parent_call_id"] == "call-1"
        assert result2["call_id"] == "call-2"

    def test_get_tool_metrics_on_demand(
        self, mock_prompt_context, mock_rendered_prompt
    ):
        """Test on-demand tool metrics generation."""
        frame = RuntimeFrame(
            session_id="test-session",
            turn_id="test-session_turn_1",
            trace_id="test-trace",
            status=ExecutionStatus.RUNNING_LLM,
        )

        # Add tool action with tool_execution_data
        action = frame.add_action(ActionKind.TOOL_CALL, name="test_tool")
        action.tool_execution_data = {
            "tool_id": "test_tool",
            "call_id": "call-1",
            "status": "completed",
            "output": "result",
            "input": {"param": "value"},
            "error": None,
            "timing": {"elapsed_seconds": 0.05},
            "decision": {},
            "metadata": {},
        }

        # Test on-demand access
        tool_metrics = frame.get_tool_metrics()

        assert len(tool_metrics) == 1
        metric = tool_metrics[0]

        assert metric["tool_name"] == "test_tool"
        assert metric["call_id"] == "call-1"
        assert metric["execution_time_ms"] == 50.0
        assert metric["success"] is True
        assert metric["resource_metrics"] == []
        assert metric["metadata"] == {}

    def test_get_reasoning_on_demand(self, mock_prompt_context, mock_rendered_prompt):
        """Test on-demand reasoning access."""
        frame = RuntimeFrame(
            session_id="test-session",
            turn_id="test-session_turn_1",
            trace_id="test-trace",
            status=ExecutionStatus.RUNNING_LLM,
        )

        # Add LLM action with reasoning
        action = frame.add_action(ActionKind.LLM_MESSAGE, name="llm_test")
        action.metadata["reasoning"] = "Test reasoning content"

        # Test on-demand access
        reasoning = frame.get_reasoning()

        assert reasoning == "Test reasoning content"

    def test_get_llm_metrics_on_demand(self, mock_prompt_context, mock_rendered_prompt):
        """Test on-demand LLM metrics access."""
        frame = RuntimeFrame(
            session_id="test-session",
            turn_id="test-session_turn_1",
            trace_id="test-trace",
            status=ExecutionStatus.RUNNING_LLM,
        )

        # Add LLM action with metrics
        llm_metrics = LLMMetrics(
            prompt_tokens=10,
            completion_tokens=20,
            cached_tokens=30,
            cache_write_tokens=5,
            reasoning_tokens=5,
            latency_ms=100.0,
            model="test-model",
        )

        action = frame.add_action(ActionKind.LLM_MESSAGE, name="llm_test")
        action.llm_metrics = llm_metrics

        # Test on-demand access
        metrics = frame.get_llm_metrics()

        assert metrics is not None
        assert metrics.model == "test-model"
        assert action.llm_metrics.prompt_tokens == 10
        assert action.llm_metrics.completion_tokens == 20
        assert action.llm_metrics.cached_tokens == 30
        assert action.llm_metrics.cache_write_tokens == 5
        assert action.llm_metrics.reasoning_tokens == 5

    def test_get_tool_results_empty(self, mock_prompt_context, mock_rendered_prompt):
        """Test on-demand access with no tool actions."""
        frame = RuntimeFrame(
            session_id="test-session",
            turn_id="test-session_turn_1",
            trace_id="test-trace",
            status=ExecutionStatus.RUNNING_LLM,
        )

        # Add only LLM action
        frame.add_action(ActionKind.LLM_MESSAGE, name="llm_test")

        # Test on-demand access
        tool_results = frame.get_tool_results()
        tool_metrics = frame.get_tool_metrics()

        assert tool_results == []
        assert tool_metrics == []

    def test_get_reasoning_empty(self, mock_prompt_context, mock_rendered_prompt):
        """Test on-demand reasoning access with no reasoning."""
        frame = RuntimeFrame(
            session_id="test-session",
            turn_id="test-session_turn_1",
            trace_id="test-trace",
            status=ExecutionStatus.RUNNING_LLM,
        )

        # Add LLM action without reasoning
        frame.add_action(ActionKind.LLM_MESSAGE, name="llm_test")

        # Test on-demand access
        reasoning = frame.get_reasoning()

        assert reasoning is None

    def test_get_llm_metrics_empty(self, mock_prompt_context, mock_rendered_prompt):
        """Test on-demand LLM metrics access with no metrics."""
        frame = RuntimeFrame(
            session_id="test-session",
            turn_id="test-session_turn_1",
            trace_id="test-trace",
            status=ExecutionStatus.RUNNING_LLM,
        )

        # Add LLM action without metrics
        frame.add_action(ActionKind.LLM_MESSAGE, name="llm_test")

        # Test on-demand access
        metrics = frame.get_llm_metrics()

        assert metrics is None
