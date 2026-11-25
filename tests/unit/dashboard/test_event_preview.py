"""
Unit tests for event preview field extraction (preview_for_event).
"""

from local_coding_assistant.dashboard.services.event_preview import preview_for_event


class TestPreviewForEvent:
    def test_llm_complete_preview(self):
        data = {
            "model": "gpt-4",
            "provider": "openai",
            "finish_reason": "stop",
            "input_tokens": 100,
            "output_tokens": 50,
            "tool_calls": [{"id": "tc-1"}],
            "content": "x" * 300,
            "latency_ms": 1250,
        }
        preview = preview_for_event("llm_complete", data)
        assert preview["model"] == "gpt-4"
        assert preview["provider"] == "openai"
        assert preview["finish_reason"] == "stop"
        assert preview["input_tokens"] == 100
        assert preview["output_tokens"] == 50
        assert preview["tool_calls"] == str([{"id": "tc-1"}])
        assert preview["content"] == "x" * 250
        assert preview["llm_exec_time"] == "1.250 sec"

    def test_tool_result_preview(self):
        data = {
            "tool_call_id": "tc-1",
            "tool_name": "echo",
            "success": True,
            "output": "y" * 300,
            "error": None,
            "stdout": "out",
            "timing": {"elapsed_seconds": 1.5},
        }
        preview = preview_for_event("tool_result", data)
        assert preview["tool_call_id"] == "tc-1"
        assert preview["tool_name"] == "echo"
        assert preview["success"] is True
        assert preview["output"] == "y" * 250
        assert preview["stdout"] == "out"
        assert preview["tool_exec_time"] == "1.500 sec"

    def test_turn_complete_preview(self):
        data = {
            "status": "completed",
            "final_answer": "z" * 300,
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "total_frames": 2,
            "finish_reason": "stop",
        }
        preview = preview_for_event("turn_complete", data)
        assert preview["status"] == "completed"
        assert preview["final_answer"] == "z" * 250
        assert preview["total_input_tokens"] == 100

    def test_frame_complete_preview(self):
        data = {
            "status": "completed",
            "finish_reason": "stop",
            "duration_ms": 2000,
            "error_message": "boom",
        }
        preview = preview_for_event("frame_complete", data)
        assert preview["status"] == "completed"
        assert preview["duration"] == "2.000 sec"
        assert preview["error_message"] == "boom"

    def test_turn_start_preview(self):
        data = {
            "user_input": "hello",
            "turn_index": 0,
            "agent_mode": "frame_agent",
            "model_name": "gpt-4",
        }
        preview = preview_for_event("turn_start", data)
        assert preview == data

    def test_session_start_preview(self):
        data = {
            "user_input": "hello",
            "agent_mode": "frame_agent",
            "model_name": "gpt-4",
        }
        preview = preview_for_event("session_start", data)
        assert preview == data

    def test_error_preview(self):
        data = {
            "error_type": "runtime_error",
            "error_message": "boom",
            "recoverable": False,
        }
        preview = preview_for_event("error", data)
        assert preview == data

    def test_approval_required_preview(self):
        data = {
            "approval_requests": [{"token": "a"}, {"token": "b"}],
            "timeout_seconds": 300,
        }
        preview = preview_for_event("approval_required", data)
        assert preview["approval_requests"] == 2

    def test_approval_granted_preview(self):
        data = {"approval_token": "tok", "tool_call_id": "tc-1"}
        preview = preview_for_event("approval_granted", data)
        assert preview["approval_token"] == "tok"  # noqa: S105 - test fixture value, not a real secret
        assert preview["tool_call_id"] == "tc-1"

    def test_approval_denied_preview(self):
        data = {
            "approval_token": "tok",
            "tool_call_id": "tc-1",
            "reason": "user denied",
        }
        preview = preview_for_event("approval_denied", data)
        assert preview["reason"] == "user denied"

    def test_unknown_event_type_returns_empty(self):
        assert preview_for_event("whatever", {"foo": "bar"}) == {}
        assert preview_for_event("", {}) == {}

    def test_missing_keys_are_omitted(self):
        preview = preview_for_event("llm_complete", {"model": "gpt-4"})
        assert preview == {"model": "gpt-4"}
