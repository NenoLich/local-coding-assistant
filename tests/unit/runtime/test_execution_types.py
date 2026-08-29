"""Tests for the new execution data structures."""

import pytest

from local_coding_assistant.core.telemetry_types import (
    ResourceMetric,
    ResourceType,
    ToolCallTrace,
    FileChange,
    FileChangeType,
)
from local_coding_assistant.runtime.execution_types import (
    ActionKind,
    ActionRecord,
    ExecutionFrame,
    ExecutionResult,
    ExecutionStatus,
    LLMMetrics,
)
from local_coding_assistant.runtime.runtime_types import ExecutionMode


class TestActionRecord:
    """Test ActionRecord with new data structure."""

    def test_action_record_creation(self):
        """Test basic ActionRecord creation."""
        action = ActionRecord(kind=ActionKind.LLM_MESSAGE, name="test_action")

        assert action.kind == ActionKind.LLM_MESSAGE
        assert action.name == "test_action"
        assert action.llm_metrics is None
        assert action.tool_trace is None
        assert action.started_at is not None
        assert action.finished_at is None

    def test_action_record_with_llm_metrics(self):
        """Test ActionRecord with LLM metrics."""
        llm_metrics = LLMMetrics(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            reasoning_tokens=5,
            latency_ms=100.0,
            model="test-model",
        )

        action = ActionRecord(
            kind=ActionKind.LLM_MESSAGE, name="llm_test", llm_metrics=llm_metrics
        )

        assert action.llm_metrics == llm_metrics
        assert action.llm_metrics.model == "test-model"
        assert action.llm_metrics.reasoning_tokens == 5

    def test_action_record_with_tool_trace(self):
        """Test ActionRecord with tool trace."""
        tool_trace = ToolCallTrace(
            call_id="test-call-123",
            tool_name="test_tool",
            success=True,
            input={"param1": "value1"},
            output="result",
            duration_ms=50.0,
        )

        action = ActionRecord(
            kind=ActionKind.TOOL_CALL, name="test_tool", tool_trace=tool_trace
        )

        assert action.tool_trace == tool_trace
        assert action.tool_trace.tool_name == "test_tool"
        assert action.tool_trace.success is True

    def test_legacy_input_output_properties(self):
        """Test legacy input/output properties for backward compatibility."""
        tool_trace = ToolCallTrace(
            call_id="test-call-123",
            tool_name="test_tool",
            success=True,
            input={"param1": "value1"},
            output="result",
        )

        action = ActionRecord(
            kind=ActionKind.TOOL_CALL, name="test_tool", tool_trace=tool_trace
        )

        # Test legacy properties
        assert action.input == {"param1": "value1"}
        assert action.output == "result"

        # Test non-tool action returns None
        llm_action = ActionRecord(kind=ActionKind.LLM_MESSAGE)
        assert llm_action.input is None
        assert llm_action.output is None


class TestExecutionResult:
    """Test ExecutionResult with simplified structure."""

    def test_execution_result_creation(self):
        """Test basic ExecutionResult creation."""
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS, final_answer="Test answer"
        )

        assert result.status == ExecutionStatus.SUCCESS
        assert result.final_answer == "Test answer"
        assert result.total_latency_ms is None
        assert result.agent_file_changes == []

    def test_execution_result_with_file_operations(self):
        """Test ExecutionResult with file operations."""
        file_change_created = FileChange(path="/path/to/new_file.txt", change_type=FileChangeType.CREATED)
        file_change_modified = FileChange(path="/path/to/modified_file.txt", change_type=FileChangeType.MODIFIED)
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
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
        from local_coding_assistant.runtime.runtime_types import PromptContext

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
        from local_coding_assistant.runtime.runtime_types import RenderedPrompt

        rendered = RenderedPrompt(
            system_messages=["System message"],
            user_messages=["User message"],
            tool_schemas=[],
        )
        return rendered

    def test_execution_frame_creation(self, mock_prompt_context, mock_rendered_prompt):
        """Test basic ExecutionFrame creation."""
        frame = ExecutionFrame(
            session_id="test-session",
            prompt_context=mock_prompt_context,
            rendered_prompt=mock_rendered_prompt,
        )

        assert frame.session_id == "test-session"
        assert frame.iteration == 1
        assert frame.actions == []
        assert frame.result is None
        assert frame.started_at is not None
        assert frame.finished_at is None

    def test_add_action(self, mock_prompt_context, mock_rendered_prompt):
        """Test adding actions to frame."""
        frame = ExecutionFrame(
            session_id="test-session",
            prompt_context=mock_prompt_context,
            rendered_prompt=mock_rendered_prompt,
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
        assert tool_action.metadata["temp_input"] == {"param": "value"}
        assert len(frame.actions) == 2

    def test_complete_llm_action(self, mock_prompt_context, mock_rendered_prompt):
        """Test completing an LLM action."""
        frame = ExecutionFrame(
            session_id="test-session",
            prompt_context=mock_prompt_context,
            rendered_prompt=mock_rendered_prompt,
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

    def test_complete_tool_action_with_trace(
        self, mock_prompt_context, mock_rendered_prompt
    ):
        """Test completing a tool action with ToolCallTrace."""
        frame = ExecutionFrame(
            session_id="test-session",
            prompt_context=mock_prompt_context,
            rendered_prompt=mock_rendered_prompt,
        )

        action = frame.add_action(
            ActionKind.TOOL_CALL, name="test_tool", _input={"param": "value"}
        )

        tool_trace = ToolCallTrace(
            call_id="test-call-123",
            tool_name="test_tool",
            success=True,
            input={"param": "value"},
            output="result",
            duration_ms=50.0,
        )

        frame.complete_action(action.id, tool_trace, {"tool_trace": tool_trace})

        assert action.finished_at is not None
        assert action.tool_trace == tool_trace
        assert action.tool_trace.input == {"param": "value"}
        assert action.tool_trace.output == "result"

    def test_complete_tool_action_legacy(
        self, mock_prompt_context, mock_rendered_prompt
    ):
        """Test completing a tool action with legacy data."""
        frame = ExecutionFrame(
            session_id="test-session",
            prompt_context=mock_prompt_context,
            rendered_prompt=mock_rendered_prompt,
        )

        action = frame.add_action(
            ActionKind.TOOL_CALL, name="test_tool", _input={"param": "value"}
        )

        metadata = {
            "call_id": "test-call-123",
            "success": True,
            "error": None,
            "parent_call_id": "parent-123",
        }

        frame.complete_action(action.id, "result", metadata)

        assert action.finished_at is not None
        assert action.tool_trace is not None
        assert action.tool_trace.call_id == "test-call-123"
        assert action.tool_trace.tool_name == "test_tool"
        assert action.tool_trace.input == {"param": "value"}
        assert action.tool_trace.output == "result"
        assert action.tool_trace.success is True
        assert action.tool_trace.parent_call_id == "parent-123"

    def test_get_tool_results_on_demand(
        self, mock_prompt_context, mock_rendered_prompt
    ):
        """Test on-demand tool results generation."""
        frame = ExecutionFrame(
            session_id="test-session",
            prompt_context=mock_prompt_context,
            rendered_prompt=mock_rendered_prompt,
        )

        # Add tool actions
        tool_trace1 = ToolCallTrace(
            call_id="call-1",
            tool_name="tool1",
            success=True,
            input={"param1": "value1"},
            output="result1",
            duration_ms=50.0,
            child_call_ids=["call-2"],
        )

        tool_trace2 = ToolCallTrace(
            call_id="call-2",
            tool_name="tool2",
            success=True,
            input={"param2": "value2"},
            output="result2",
            duration_ms=30.0,
            parent_call_id="call-1",
        )

        action1 = frame.add_action(ActionKind.TOOL_CALL, name="tool1")
        action1.tool_trace = tool_trace1

        action2 = frame.add_action(ActionKind.TOOL_CALL, name="tool2")
        action2.tool_trace = tool_trace2

        # Test on-demand access
        tool_results = frame.get_tool_results()

        assert len(tool_results) == 2

        # Check first tool result
        result1 = next(r for r in tool_results if r["tool_name"] == "tool1")
        assert result1["tool_args"] == {"param1": "value1"}
        assert result1["result"] == "result1"
        assert result1["success"] is True
        assert result1["child_call_ids"] == ["call-2"]

        # Check second tool result
        result2 = next(r for r in tool_results if r["tool_name"] == "tool2")
        assert result2["tool_args"] == {"param2": "value2"}
        assert result2["result"] == "result2"
        assert result2["parent_call_id"] == "call-1"

    def test_get_tool_metrics_on_demand(
        self, mock_prompt_context, mock_rendered_prompt
    ):
        """Test on-demand tool metrics generation."""
        frame = ExecutionFrame(
            session_id="test-session",
            prompt_context=mock_prompt_context,
            rendered_prompt=mock_rendered_prompt,
        )

        # Add tool action with resource metrics
        resource_metric = ResourceMetric(
            type=ResourceType.CPU, name="cpu_usage", value=25.5, unit="percent"
        )

        tool_trace = ToolCallTrace(
            call_id="call-1",
            tool_name="test_tool",
            success=True,
            input={"param": "value"},
            output="result",
            duration_ms=50.0,
            resource_metrics=[resource_metric],
        )

        action = frame.add_action(ActionKind.TOOL_CALL, name="test_tool")
        action.tool_trace = tool_trace

        # Test on-demand access
        tool_metrics = frame.get_tool_metrics()

        assert len(tool_metrics) == 1
        metric = tool_metrics[0]

        assert metric["tool_name"] == "test_tool"
        assert metric["call_id"] == "call-1"
        assert metric["execution_time_ms"] == 50.0
        assert metric["success"] is True
        assert len(metric["resource_metrics"]) == 1
        assert metric["resource_metrics"][0]["type"] == "cpu"

    def test_get_reasoning_on_demand(self, mock_prompt_context, mock_rendered_prompt):
        """Test on-demand reasoning access."""
        frame = ExecutionFrame(
            session_id="test-session",
            prompt_context=mock_prompt_context,
            rendered_prompt=mock_rendered_prompt,
        )

        # Add LLM action with reasoning
        action = frame.add_action(ActionKind.LLM_MESSAGE, name="llm_test")
        action.metadata["reasoning"] = "Test reasoning content"

        # Test on-demand access
        reasoning = frame.get_reasoning()

        assert reasoning == "Test reasoning content"

    def test_get_llm_metrics_on_demand(self, mock_prompt_context, mock_rendered_prompt):
        """Test on-demand LLM metrics access."""
        frame = ExecutionFrame(
            session_id="test-session",
            prompt_context=mock_prompt_context,
            rendered_prompt=mock_rendered_prompt,
        )

        # Add LLM action with metrics
        llm_metrics = LLMMetrics(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
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
        assert metrics.reasoning_tokens == 5
        assert metrics.total_tokens == 30

    def test_get_tool_results_empty(self, mock_prompt_context, mock_rendered_prompt):
        """Test on-demand access with no tool actions."""
        frame = ExecutionFrame(
            session_id="test-session",
            prompt_context=mock_prompt_context,
            rendered_prompt=mock_rendered_prompt,
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
        frame = ExecutionFrame(
            session_id="test-session",
            prompt_context=mock_prompt_context,
            rendered_prompt=mock_rendered_prompt,
        )

        # Add LLM action without reasoning
        frame.add_action(ActionKind.LLM_MESSAGE, name="llm_test")

        # Test on-demand access
        reasoning = frame.get_reasoning()

        assert reasoning is None

    def test_get_llm_metrics_empty(self, mock_prompt_context, mock_rendered_prompt):
        """Test on-demand LLM metrics access with no metrics."""
        frame = ExecutionFrame(
            session_id="test-session",
            prompt_context=mock_prompt_context,
            rendered_prompt=mock_rendered_prompt,
        )

        # Add LLM action without metrics
        frame.add_action(ActionKind.LLM_MESSAGE, name="llm_test")

        # Test on-demand access
        metrics = frame.get_llm_metrics()

        assert metrics is None
