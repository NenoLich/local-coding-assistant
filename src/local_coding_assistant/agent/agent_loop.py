"""Agent loop implementation for observe-plan-act-reflect lifecycle."""

import time
from typing import Any

from local_coding_assistant.agent.llm_manager import LLMManager, LLMRequest, ToolCall
from local_coding_assistant.core.exceptions import AgentError
from local_coding_assistant.core.protocols import IToolManager
from local_coding_assistant.tools.types import ToolExecutionRequest
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("agent.loop")


class AgentLoop:
    """Observe-Plan-Act-Reflect agent loop implementation.

    This class manages the lifecycle of an autonomous agent that follows
    the observe-plan-act-reflect pattern. It creates default handlers that
    use LLM and tool managers for autonomous operation.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        tool_manager: IToolManager | None = None,
        name: str = "agent_loop",
        max_iterations: int = 10,
        streaming: bool = False,
    ):
        """Initialize the agent loop.

        Args:
            llm_manager: The LLM manager to use for reasoning.
            tool_manager: Optional tool manager providing available tools.
                         If not provided, the agent will operate without tools.
            name: A name for this agent loop instance.
            max_iterations: Maximum number of iterations to run.
            streaming: Whether to use streaming LLM responses.
        """
        if max_iterations < 1:
            from local_coding_assistant.core.exceptions import AgentError

            raise AgentError("max_iterations must be at least 1")

        self.llm_manager = llm_manager
        self.tool_manager = tool_manager
        self.name = name
        self.max_iterations = max_iterations
        self.streaming = streaming
        self.current_iteration = 0
        self.is_running = False
        self.final_answer = None
        self.session_id = f"agent_{name}_{int(time.time())}"
        self.history: list[dict[str, Any]] = []

        # Cache tools for performance - avoid multiple iterations
        self._cached_tools = self._get_available_tools()

        # Set up default handlers
        self._setup_default_handlers()

        logger.info(
            f"Initialized agent loop '{name}' with max_iterations={max_iterations}"
        )

    def _setup_default_handlers(self) -> None:
        """Set up default handlers for each phase of the agent loop."""
        # Store handlers
        self.observe_handler = self._default_observe_handler
        self.plan_handler = self._default_plan_handler
        self.act_handler = self._default_act_handler
        self.reflect_handler = self._default_reflect_handler

        logger.info(f"Set up default handlers for agent '{self.name}'")

    def _default_observe_handler(self) -> dict[str, Any]:
        """Generate observation based on current context.

        For now, create a simple observation based on current state.
        In a real implementation, this might gather information from the environment.
        """
        content = (
            f"Agent iteration {self.current_iteration + 1} in session {self.session_id}"
        )
        return {
            "content": content,
            "metadata": {
                "session_id": self.session_id,
                "iteration": self.current_iteration + 1,
            },
            "timestamp": time.time(),
            "source": "agent_loop",
        }

    async def _default_plan_handler(
        self, observation: dict[str, Any]
    ) -> dict[str, Any]:
        """Use LLM to create a plan based on the observation."""
        prompt = f"""
Based on the following observation, create a plan for what the agent should do next:

Observation: {observation["content"]}

Context:
- Session ID: {observation["metadata"].get("session_id", "unknown")}
- Iteration: {observation["metadata"].get("iteration", 0)}

Please provide a plan with specific actions to take. Respond in JSON format with:
- reasoning: your reasoning for the plan
- actions: list of specific actions to take
- confidence: confidence level (0-1)
"""
        try:
            request = LLMRequest(
                prompt=prompt,
                tools=self._cached_tools,
            )

            response_content = (
                await self._stream_response(request)
                if self.streaming
                else (await self.llm_manager.generate(request)).content
            )

            # For now, create a simple plan - in a real implementation,
            # this would parse structured output from the LLM
            return {
                "reasoning": f"Based on observation: {observation['content'][:100]}...",
                "actions": ["analyze_current_state", "determine_next_action"],
                "confidence": 0.8,
                "metadata": {"llm_response": response_content},
            }
        except Exception as e:
            logger.error(f"Failed to generate plan: {e}")
            # Fallback plan
            return {
                "reasoning": "Failed to generate plan, using fallback",
                "actions": ["retry_planning"],
                "confidence": 0.3,
                "metadata": {"error": str(e)},
            }

    async def _default_act_handler(self, plan: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
        """Execute actions using tools and LLM."""
        try:
            # Create a prompt that includes tool calling instructions
            action_prompt = f"""
Execute the following plan:

Plan: {plan["reasoning"]}
Actions: {", ".join(plan["actions"])}

You have access to the following tools:
{self._get_tools_description()}

If you need to use a tool, respond with a function call in the format:
{{"function_call": {{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}}}

If you need to provide a final answer, use the final_answer tool.

Please describe what actions were taken and their results.
"""
            request = LLMRequest(
                prompt=action_prompt,
                tools=self._cached_tools,
            )

            response_content = (
                await self._stream_response(request)
                if self.streaming
                else (await self.llm_manager.generate(request)).content
            )

            # For now, we'll need the complete response to parse tool calls
            # In a real streaming implementation, tool calls would be detected
            # as they stream in, but for simplicity, we'll use the complete response
            if self.streaming:
                # For streaming, we need to get tool calls from the complete response
                # This is a simplified approach - a real implementation would
                # need more sophisticated parsing of streaming content
                response = await self.llm_manager.generate(request)
                tool_calls = response.tool_calls or []
            else:
                # For non-streaming, we already have the response object
                response = await self.llm_manager.generate(request)
                tool_calls = response.tool_calls or []

            # Check if this is a tool call response
            if tool_calls:
                for tool_call in tool_calls:
                    if isinstance(tool_call, ToolCall):
                        func_name = tool_call.name
                        try:
                            # Parse arguments
                            args = tool_call.arguments

                            # Handle tool execution
                            if self.tool_manager is None:
                                tool_result = f"Error: No tool manager available to execute {func_name}"
                            else:
                                try:
                                    request = ToolExecutionRequest(
                                        tool_name=func_name, payload=args
                                    )
                                    response = await self.tool_manager.execute_async(
                                        request
                                    )
                                    if not response.success:
                                        tool_result = f"Error: {response.error_message}"
                                    else:
                                        tool_result = response.result
                                except Exception as e:
                                    tool_result = (
                                        f"Error executing tool {func_name}: {e!s}"
                                    )

                            # Special handling for final_answer tool
                            if func_name == "final_answer":
                                self.final_answer = args.get("answer", "")
                                logger.info(
                                    f"Final answer received: {self.final_answer}"
                                )
                                return {
                                    "success": True,
                                    "output": f"Final answer: {self.final_answer}",
                                    "metadata": {
                                        "tool_calls": tool_calls,
                                        "stopped": True,
                                    },
                                }

                            logger.debug(f"Tool call: {func_name} => {tool_result}")
                            return {
                                "success": True,
                                "output": f"Tool {func_name} executed successfully",
                                "metadata": {
                                    "tool_calls": tool_calls,
                                    "tool_results": {func_name: tool_result},
                                },
                            }
                        except Exception as e:
                            logger.error(f"Tool call failed: {e}")
                            return {
                                "success": False,
                                "error": str(e),
                                "metadata": {"plan_actions": plan["actions"]},
                            }

            # If no tool calls, return the LLM response as output
            return {
                "success": True,
                "output": response_content,
                "metadata": {"tool_calls": tool_calls},
            }
        except Exception as e:
            logger.error(f"Failed to execute actions: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {"plan_actions": plan["actions"]},
            }

    async def _default_reflect_handler(
        self, action_result: dict[str, Any], plan: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze results and learn from the action execution."""
        try:
            reflection_prompt = f"""
Analyze the following action results and provide reflection:

Plan: {plan["reasoning"]}
Actions: {", ".join(plan["actions"])}
Success: {action_result["success"]}
Output: {action_result.get("output", "")}
Error: {action_result.get("error", "")}

Please provide:
- analysis: analysis of what happened
- lessons_learned: key lessons
- improvements: suggested improvements
- success_rating: success rating (0-1)
"""
            request = LLMRequest(prompt=reflection_prompt)
            response_content = (
                await self._stream_response(request)
                if self.streaming
                else (await self.llm_manager.generate(request)).content
            )

            # For now, create a simple reflection - in a real implementation,
            # this would parse structured output from the LLM
            return {
                "analysis": f"Plan execution {'succeeded' if action_result['success'] else 'failed'}: {response_content[:200]}",
                "lessons_learned": [
                    "Monitor action results carefully",
                    "Adjust plans based on outcomes",
                ],
                "improvements": [
                    "Better error handling",
                    "More detailed action planning",
                ]
                if not action_result["success"]
                else [],
                "success_rating": 0.9 if action_result["success"] else 0.3,
            }
        except Exception as e:
            logger.error(f"Failed to generate reflection: {e}")
            return {
                "analysis": "Failed to generate reflection",
                "success_rating": 0.0,
                "lessons_learned": ["Improve reflection process"],
            }

    def _get_available_tools(self) -> list[dict[str, Any]]:
        """Get available tools for LLM requests."""
        if self.tool_manager is None:
            return []

        available_tools = []
        for tool in self.tool_manager.list_tools(available_only=True):
            if hasattr(tool, "name") and hasattr(tool, "description"):
                available_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": getattr(tool, "parameters", {}),
                    }
                )
        return available_tools

    def _get_tools_description(self) -> str:
        """Get a formatted description of available tools."""
        if self.tool_manager is None:
            return "No tools available"

        descriptions = []
        for tool in self.tool_manager.list_tools(available_only=True):
            if hasattr(tool, "name") and hasattr(tool, "description"):
                descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions) if descriptions else "No tools available"

    async def _stream_response(self, request: LLMRequest) -> str:
        """Generate LLM response using streaming.

        Args:
            request: The LLM request to process

        Returns:
            The complete response content as a string

        Note:
            This method should only be called when streaming is enabled.
            For non-streaming mode, use llm_manager.generate() directly.
        """
        # Use streaming for real-time response processing
        logger.debug("Using streaming LLM response")
        response_content = ""
        async for chunk in self.llm_manager.stream(request):
            response_content += chunk
        return response_content

    def _execute_observe_phase(self, iteration_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the observe phase of the agent loop."""
        try:
            logger.debug("Executing observe phase")
            observation = self.observe_handler()
            iteration_data["observation"] = observation
            logger.info(f"Observation collected: {observation['content'][:100]}...")
            return observation
        except Exception as e:
            logger.error(f"Error in observe phase: {e}")
            raise AgentError(f"Observation phase failed: {e}") from e

    async def _execute_plan_phase(
        self, observation: dict[str, Any], iteration_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the plan phase of the agent loop."""
        try:
            logger.debug("Executing plan phase")
            plan = await self.plan_handler(observation)
            iteration_data["plan"] = plan
            logger.info(f"Plan created with {len(plan['actions'])} actions")
            return plan
        except Exception as e:
            logger.error(f"Error in plan phase: {e}")
            # For error recovery testing, handle planning errors gracefully
            # Create a basic plan and continue
            plan = {
                "reasoning": "Planning phase encountered an error, using fallback",
                "actions": ["retry"],
                "confidence": 0.1,
                "metadata": {"error": str(e)},
            }
            iteration_data["plan"] = plan
            return plan

    async def _execute_act_phase(
        self, plan: dict[str, Any], iteration_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the act phase of the agent loop."""
        try:
            logger.debug("Executing act phase")
            action_result = await self.act_handler(plan)
            iteration_data["action_result"] = action_result
            logger.info(
                f"Action completed: {'success' if action_result['success'] else 'failed'}"
            )
            return action_result
        except Exception as e:
            logger.error(f"Error in act phase: {e}")
            # For error recovery testing, handle action errors gracefully
            action_result = {
                "success": False,
                "error": str(e),
                "output": "Action phase failed",
                "metadata": {"plan_actions": plan["actions"]},
            }
            iteration_data["action_result"] = action_result
            # Stop the loop after action error since act is critical
            logger.info("Stopping due to critical action error")
            # Still need to add reflection and append to history
            iteration_data["reflection"] = {
                "analysis": "Action phase failed - loop terminated",
                "success_rating": 0.0,
                "lessons_learned": ["Action execution is critical"],
                "improvements": ["Better error handling for actions"],
            }
            self.history.append(iteration_data)
            raise e

    async def _execute_reflect_phase(
        self,
        action_result: dict[str, Any],
        plan: dict[str, Any],
        iteration_data: dict[str, Any],
    ) -> None:
        """Execute the 'reflect' phase of the agent loop."""
        try:
            logger.debug("Executing reflect phase")
            reflection = await self.reflect_handler(action_result, plan)
            iteration_data["reflection"] = reflection
            logger.info(
                f"Reflection completed with success rating: {reflection['success_rating']}"
            )
        except Exception as e:
            logger.error(f"Error in reflect phase: {e}")
            # Reflection failure is not critical, create a basic reflection
            reflection = {
                "analysis": "Reflection phase encountered an error",
                "success_rating": 0.0,
                "lessons_learned": [],
                "improvements": [],
            }
            iteration_data["reflection"] = reflection

    def _check_early_termination(
        self, plan: dict[str, Any], action_result: dict[str, Any]
    ) -> bool:
        """Check if the loop should terminate early based on various conditions."""
        # Check if final answer was provided
        if action_result.get("metadata", {}).get("stopped"):
            return True

        # Check if we should stop early due to critical failures
        plan_confidence = plan.get("confidence", 1.0)
        try:
            plan_confidence = float(plan_confidence)
        except (ValueError, TypeError):
            plan_confidence = 1.0

        metadata = plan.get("metadata", {})
        plan_had_error = (
            isinstance(metadata, dict) and metadata.get("error") is not None
        )

        return (plan_confidence < 0.5 and plan_had_error) or not action_result[
            "success"
        ]

    def _handle_final_answer(
        self, action_result: dict[str, Any], iteration_data: dict[str, Any]
    ) -> bool:
        """Handle final answer termination and return whether to stop."""
        if action_result.get("metadata", {}).get("stopped"):
            logger.info("Loop stopped due to final answer")
            # Still append to history even if stopping early, but skip reflection
            # since we're terminating due to final answer
            iteration_data["reflection"] = {
                "analysis": "Loop terminated by final answer - reflection skipped",
                "success_rating": 1.0,
                "lessons_learned": ["Final answer tool works correctly"],
                "improvements": [],
            }
            self.history.append(iteration_data)
            return True
        return False

    async def _execute_single_iteration(self, iteration_data: dict[str, Any]) -> bool:
        """Execute a single iteration of the agent loop.

        Returns:
            True if the loop should continue, False if it should stop
        """
        # Execute observe phase
        try:
            observation = self._execute_observe_phase(iteration_data)
        except AgentError:
            # Observe phase is critical, re-raise if it fails
            raise

        # Execute plan phase
        plan = await self._execute_plan_phase(observation, iteration_data)

        # Execute act phase
        try:
            action_result = await self._execute_act_phase(plan, iteration_data)
        except Exception:
            # Act phase error was handled in _execute_act_phase
            # The method already added reflection and history, so we can break
            return False

        # Check if final answer was provided
        if self._handle_final_answer(action_result, iteration_data):
            return False

        # Execute reflect phase
        await self._execute_reflect_phase(action_result, plan, iteration_data)

        # Add to history
        self.history.append(iteration_data)

        # Check if we should stop early
        return not self._check_early_termination(plan, action_result)

    async def run(self) -> str | None:
        if self.is_running:
            raise AgentError("Agent loop is already running")

        self.is_running = True
        self.current_iteration = 0
        self.final_answer = None
        logger.info(f"Starting agent loop '{self.name}'")

        try:
            while self.current_iteration < self.max_iterations and self.is_running:
                self.current_iteration += 1
                logger.info(
                    f"Starting iteration {self.current_iteration}/{self.max_iterations}"
                )

                iteration_data = {
                    "iteration": self.current_iteration,
                    "timestamp": time.time(),
                    "observation": None,
                    "plan": None,
                    "action_result": None,
                    "reflection": None,
                }

                # Execute single iteration
                try:
                    should_continue = await self._execute_single_iteration(
                        iteration_data
                    )
                    if not should_continue:
                        break
                except AgentError:
                    # Re-raise critical errors
                    raise

            logger.info(
                f"Agent loop '{self.name}' completed after {self.current_iteration} iterations"
            )

            # Return final answer if provided
            return self.final_answer

        except Exception as e:
            logger.error(f"Agent loop '{self.name}' failed: {e}")
            raise
        finally:
            self.is_running = False

    def stop(self) -> None:
        """Stop the agent loop gracefully."""
        if self.is_running:
            logger.info(f"Stopping agent loop '{self.name}'")
            self.is_running = False
        else:
            logger.warning(f"Agent loop '{self.name}' is not running")

    def get_history(self) -> list[dict[str, Any]]:
        """Get the complete execution history.

        Returns:
            List of iteration data dictionaries

        Raises:
            AgentError: If history is not properly initialized
        """
        # Ensure history field exists and is properly initialized
        if not hasattr(self, "history"):
            logger.warning("History field not initialized, creating empty history")
            self.history = []

        # Validate that history is a list
        if not isinstance(self.history, list):
            logger.error(f"History field is not a list, got {type(self.history)}")
            from local_coding_assistant.core.exceptions import AgentError

            raise AgentError("History field is corrupted or not properly initialized")

        # Return a copy to prevent external modification
        return self.history.copy()

    def clear_history(self) -> None:
        """Clear the execution history.

        This method resets the history field to an empty list.
        Useful for starting fresh or freeing memory.
        """
        logger.debug(f"Clearing history for agent loop '{self.name}'")
        self.history = []

    def get_history_length(self) -> int:
        """Get the number of iterations in the history.

        Returns:
            Number of completed iterations in the history
        """
        return len(self.history)

    def is_loop_running(self) -> bool:
        """Check if the loop is currently running.

        Returns:
            True if the loop is running, False otherwise
        """
        return self.is_running
