"""
Integration tests for the end-to-end reasoning chain.

Tests the complete flow from RuntimeManager through AgentLoop with:
- Mock LLM with streaming outputs
- Multiple tool interactions
- Session persistence
- Structured response consistency
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.agent.agent_loop import AgentLoop
from local_coding_assistant.agent.llm_manager import LLMRequest


class TestEndToEndReasoningChain:
    """Integration tests for the complete reasoning chain."""

    @pytest.mark.asyncio
    async def test_end_to_end_reasoning_with_tool_calls(self, runtime_manager):
        """Test complete reasoning chain with multiple tool calls and iterations."""
        # Run the agent in agent mode
        result = await runtime_manager._run_agent_mode(
            text="Plan what to pack for a trip to New York",
            model="gpt-4",
            temperature=0.7,
        )

        # Verify the result structure
        assert "final_answer" in result
        assert "iterations" in result
        assert "history" in result
        assert "session_id" in result

        # Verify multiple iterations occurred
        assert result["iterations"] >= 2  # Should have multiple iterations

        # Verify history contains multiple iterations
        history = result["history"]
        assert len(history) >= 2

        # Verify tool calls were made in sequence
        tool_calls_found = []
        for iteration in history:
            if iteration.get("plan") and iteration["plan"].get("metadata", {}).get(
                "llm_response"
            ):
                # Check for tool calls in the response metadata
                tool_calls_found.extend(
                    iteration["plan"]["metadata"]["llm_response"].tool_calls
                )

        # Should have found tool calls
        assert len(tool_calls_found) > 0

        # Verify final answer is coherent
        final_answer = result["final_answer"]
        assert final_answer is not None
        assert "New York" in final_answer or "pack" in final_answer.lower()

    @pytest.mark.asyncio
    async def test_runtime_manager_orchestrate_delegation(self, runtime_manager):
        """Test that RuntimeManager.orchestrate() delegates to AgentLoop correctly."""
        # Mock the orchestrate method to verify it calls _run_agent_mode
        original_run_agent_mode = runtime_manager._run_agent_mode

        async def mock_orchestrate(query, **kwargs):
            if kwargs.get("agent_mode"):
                return await original_run_agent_mode(query, **kwargs)
            return {"result": "not_agent_mode"}

        runtime_manager.orchestrate = mock_orchestrate

        # Test agent mode delegation
        result = await runtime_manager.orchestrate(
            "Test query", agent_mode=True, model="gpt-4"
        )

        # Should have called _run_agent_mode and returned structured result
        assert "final_answer" in result
        assert "iterations" in result

    @pytest.mark.asyncio
    async def test_session_persistence_across_queries(self, runtime_manager):
        """Test that the runtime maintains session state across multiple queries."""
        # First query - establish some context
        result1 = await runtime_manager._run_agent_mode(
            text="What is the weather like in New York?", model="gpt-4"
        )

        # Second query - should be able to recall previous context
        result2 = await runtime_manager._run_agent_mode(
            text="Based on the previous weather information, what should I pack?",
            model="gpt-4",
        )

        # Both queries should complete successfully
        assert result1["final_answer"] is not None
        assert result2["final_answer"] is not None

        # Both should have session IDs
        assert result1["session_id"] is not None
        assert result2["session_id"] is not None

        # Session IDs should be the same (same session)
        assert result1["session_id"] == result2["session_id"]

    @pytest.mark.asyncio
    async def test_streaming_output_buffering(
        self, streaming_llm_single_response, tool_manager
    ):
        """Test that AgentLoop correctly handles streaming LLM outputs."""
        agent_loop = AgentLoop(
            llm_manager=streaming_llm_single_response,
            tool_manager=tool_manager,
            name="streaming_test",
            max_iterations=3,
        )

        # Mock handlers to simulate streaming scenario
        streaming_chunks = []

        async def mock_observe():
            return {
                "content": "Test observation for streaming",
                "metadata": {"session_id": "streaming_test"},
                "timestamp": time.time(),
                "source": "test",
            }

        async def mock_plan(observation):
            return {
                "reasoning": "Plan for streaming test",
                "actions": ["test_streaming"],
                "confidence": 0.8,
                "metadata": {},
            }

        async def mock_act(plan):
            # Simulate processing streaming response
            response_content = ""
            async for chunk in streaming_llm_single_response.generate_stream(
                LLMRequest(prompt="Test streaming")
            ):
                response_content += chunk + " "
                streaming_chunks.append(chunk)

            return {
                "success": True,
                "output": f"Processed streaming: {response_content.strip()}",
                "metadata": {"streaming_chunks": len(streaming_chunks)},
            }

        async def mock_reflect(action_result, plan):
            return {
                "analysis": f"Reflection on streaming output: {action_result['output'][:100]}",
                "lessons_learned": ["Streaming works correctly"],
                "improvements": [],
                "success_rating": 0.9,
            }

        agent_loop.observe_handler = mock_observe
        agent_loop.plan_handler = mock_plan
        agent_loop.act_handler = mock_act
        agent_loop.reflect_handler = mock_reflect

        # Run the agent loop
        result = await agent_loop.run()

        # Verify streaming was processed
        assert len(streaming_chunks) > 0
        assert result is not None

        # Check that history contains streaming metadata
        history = agent_loop.get_history()
        assert len(history) >= 1

        # Verify the act result contains streaming information
        act_result = history[0]["action_result"]
        assert "streaming_chunks" in act_result["metadata"]

    @pytest.mark.asyncio
    async def test_structured_response_consistency(self, runtime_manager):
        """Test that structured responses maintain consistency across iterations."""
        # Run multiple queries to test response structure
        queries = [
            "Calculate 15 + 27",
            "What's the weather in London?",
            "Help me plan a vacation to Tokyo",
        ]

        results = []
        for query in queries:
            result = await runtime_manager._run_agent_mode(query, model="gpt-4")
            results.append(result)

        # Verify all results have consistent structure
        for result in results:
            assert "final_answer" in result
            assert "iterations" in result
            assert "history" in result
            assert "session_id" in result

            # Verify history structure
            for iteration in result["history"]:
                assert "iteration" in iteration
                assert "observation" in iteration
                assert "plan" in iteration
                assert "action_result" in iteration
                assert "reflection" in iteration

                # Verify each phase has expected fields
                assert "content" in iteration["observation"]
                assert "reasoning" in iteration["plan"]
                assert "actions" in iteration["plan"]
                assert "success" in iteration["action_result"]
                assert "analysis" in iteration["reflection"]

        # Verify session consistency (should use same session for all queries)
        session_ids = {result["session_id"] for result in results}
        assert len(session_ids) == 1  # All queries should use same session


class TestAgentLoopIntegration:
    """Integration tests specifically for AgentLoop behavior."""

    @pytest.mark.asyncio
    async def test_complex_multi_iteration_reasoning(
        self, complex_scenario_llm, complex_tool_manager
    ):
        """Test complex reasoning that spans multiple iterations with different tools."""
        agent_loop = AgentLoop(
            llm_manager=complex_scenario_llm,
            tool_manager=complex_tool_manager,
            name="complex_reasoning_test",
            max_iterations=5,
        )

        # Simple mock handlers for this test
        async def mock_observe():
            return {
                "content": "Complex problem requiring multi-step reasoning",
                "metadata": {"session_id": "complex_test"},
                "timestamp": time.time(),
                "source": "test",
            }

        async def mock_plan(observation):
            return {
                "reasoning": "Need to gather weather data and perform calculations",
                "actions": ["get_weather", "calculate", "analyze"],
                "confidence": 0.9,
                "metadata": {},
            }

        async def mock_act(plan):
            # Use the LLM to generate response
            response = await complex_scenario_llm.generate(
                LLMRequest(prompt=f"Execute plan: {plan['reasoning']}")
            )

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    func_name = tool_call["function"]["name"]
                    if func_name == "final_answer":
                        args = json.loads(tool_call["function"]["arguments"])
                        agent_loop.final_answer = args.get("answer", "")

            return {
                "success": True,
                "output": response.content,
                "metadata": {"tool_calls": response.tool_calls},
            }

        async def mock_reflect(action_result, plan):
            return {
                "analysis": f"Successfully executed: {action_result['output'][:100]}",
                "lessons_learned": [
                    "Multi-step reasoning works",
                    "Tool integration successful",
                ],
                "improvements": [],
                "success_rating": 0.95,
            }

        agent_loop.observe_handler = mock_observe
        agent_loop.plan_handler = mock_plan
        agent_loop.act_handler = mock_act
        agent_loop.reflect_handler = mock_reflect

        # Run the agent loop
        result = await agent_loop.run()

        # Verify complex reasoning completed
        assert result is not None
        assert agent_loop.current_iteration >= 2  # Should have multiple iterations
        assert len(agent_loop.get_history()) >= 2

        # Verify final answer contains expected content
        assert (
            "New York" in result
            or "weather" in result.lower()
            or "calculation" in result.lower()
        )

    @pytest.mark.asyncio
    async def test_error_recovery_in_reasoning_chain(
        self, mock_llm_with_tools, tool_manager
    ):
        """Test that the reasoning chain can recover from errors."""
        agent_loop = AgentLoop(
            llm_manager=mock_llm_with_tools,
            tool_manager=tool_manager,
            name="error_recovery_test",
            max_iterations=3,
        )

        # Track errors for verification
        errors_encountered = []

        async def mock_observe():
            return {
                "content": "Test observation",
                "metadata": {"session_id": "error_test"},
                "timestamp": time.time(),
                "source": "test",
            }

        async def mock_plan(observation):
            # Simulate occasional planning failure
            if agent_loop.current_iteration == 1:
                raise Exception("Simulated planning error")
            return {
                "reasoning": "Plan created successfully",
                "actions": ["test_action"],
                "confidence": 0.8,
                "metadata": {},
            }

        async def mock_act(plan):
            return {"success": True, "output": "Action completed", "metadata": {}}

        async def mock_reflect(action_result, plan):
            return {
                "analysis": "Reflection completed",
                "lessons_learned": ["Error recovery works"],
                "improvements": [],
                "success_rating": 0.8,
            }

        agent_loop.observe_handler = mock_observe
        agent_loop.plan_handler = mock_plan
        agent_loop.act_handler = mock_act
        agent_loop.reflect_handler = mock_reflect

        # This should handle the error gracefully
        result = await agent_loop.run()

        # Should still complete despite the error
        assert result is None  # No final answer due to early termination
        assert agent_loop.current_iteration >= 1  # Should have at least one iteration
