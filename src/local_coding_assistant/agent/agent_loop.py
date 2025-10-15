"""Agent loop implementation for observe-plan-act-reflect lifecycle."""

import json
import time
from typing import Any

from local_coding_assistant.agent.llm_manager import LLMManager, LLMRequest
from local_coding_assistant.core.exceptions import AgentError
from local_coding_assistant.tools.tool_manager import ToolManager
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
        tool_manager: ToolManager,
        name: str = "agent",
        max_iterations: int = 10,
        observation_timeout: float = 30.0,
        action_timeout: float = 60.0,
    ):
        """Initialize the agent loop.

        Args:
            llm_manager: LLM manager for generating responses
            tool_manager: Tool manager for executing tools
            name: Name of the agent for logging purposes
            max_iterations: Maximum number of loop iterations before stopping
            observation_timeout: Timeout for observation phase in seconds
            action_timeout: Timeout for action execution in seconds

        Raises:
            AgentError: If configuration is invalid
        """
        if max_iterations < 1:
            raise AgentError("max_iterations must be at least 1")

        if observation_timeout <= 0:
            raise AgentError("observation_timeout must be positive")

        if action_timeout <= 0:
            raise AgentError("action_timeout must be positive")

        self.llm_manager = llm_manager
        self.tool_manager = tool_manager
        self.name = name
        self.max_iterations = max_iterations
        self.observation_timeout = observation_timeout
        self.action_timeout = action_timeout
        self.is_running = False
        self.current_iteration = 0
        self.history: list[dict[str, Any]] = []
        self.final_answer: str | None = None
        self.session_id = f"agent_{name}_{int(time.time())}"

        # Cache tools for performance - avoid multiple iterations
        self._cached_tools = self._get_available_tools()

        # Set up default handlers
        self._setup_default_handlers()

        logger.info(
            f"Initialized agent loop '{name}' with max_iterations={max_iterations}"
        )

    def _setup_default_handlers(self) -> None:
        """Set up default handlers for each phase of the agent loop."""

        # Observe handler - generate observation based on current context
        async def observe() -> dict[str, Any]:
            # For now, create a simple observation based on current state
            # In a real implementation, this might gather information from the environment
            content = f"Agent iteration {self.current_iteration + 1} in session {self.session_id}"
            return {
                "content": content,
                "metadata": {
                    "session_id": self.session_id,
                    "iteration": self.current_iteration + 1,
                },
                "timestamp": time.time(),
                "source": "agent_loop",
            }

        # Plan handler - use LLM to create a plan
        async def plan(observation: dict[str, Any]) -> dict[str, Any]:
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

                response = await self.llm_manager.generate(request)

                # For now, create a simple plan - in a real implementation,
                # this would parse structured output from the LLM
                return {
                    "reasoning": f"Based on observation: {observation['content'][:100]}...",
                    "actions": ["analyze_current_state", "determine_next_action"],
                    "confidence": 0.8,
                    "metadata": {"llm_response": response.content},
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

        # Act handler - execute actions using tools and LLM
        async def act(plan: dict[str, Any]) -> dict[str, Any]:
            try:
                # Create a prompt that includes tool calling instructions
                action_prompt = f"""
Execute the following plan:

Plan: {plan["reasoning"]}
Actions: {", ".join(plan["actions"])}

You have access to the following tools:
{self._get_tools_description()}

If you need to use a tool, respond with a function call in the format:
{"function_call": {"name": "tool_name", "arguments": {"arg1": "value1"}}}

If you need to provide a final answer, use the final_answer tool.

Please describe what actions were taken and their results.
"""
                request = LLMRequest(
                    prompt=action_prompt,
                    tools=self._cached_tools,
                )

                response = await self.llm_manager.generate(request)

                # Check if this is a tool call response
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        if "function" in tool_call:
                            func_name = tool_call["function"]["name"]
                            try:
                                # Parse arguments and invoke tool
                                args = json.loads(tool_call["function"]["arguments"])
                                tool_result = self.tool_manager.run_tool(
                                    func_name, args
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
                                            "tool_calls": response.tool_calls,
                                            "stopped": True,
                                        },
                                    }

                                logger.debug(f"Tool call: {func_name} => {tool_result}")
                                return {
                                    "success": True,
                                    "output": f"Tool {func_name} executed successfully",
                                    "metadata": {
                                        "tool_calls": response.tool_calls,
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
                    "output": response.content,
                    "metadata": {"tool_calls": response.tool_calls},
                }
            except Exception as e:
                logger.error(f"Failed to execute actions: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "metadata": {"plan_actions": plan["actions"]},
                }

        # Reflect handler - analyze results and learn
        async def reflect(
            action_result: dict[str, Any], plan: dict[str, Any]
        ) -> dict[str, Any]:
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
                response = await self.llm_manager.generate(request)

                # For now, create a simple reflection - in a real implementation,
                # this would parse structured output from the LLM
                return {
                    "analysis": f"Plan execution {'succeeded' if action_result['success'] else 'failed'}: {response.content[:200]}",
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

        # Store handlers
        self.observe_handler = observe
        self.plan_handler = plan
        self.act_handler = act
        self.reflect_handler = reflect

        logger.info(f"Set up default handlers for agent '{self.name}'")

    def _get_available_tools(self) -> list[dict[str, Any]]:
        """Get available tools for LLM requests - cached for performance."""
        available_tools = []
        for tool in self.tool_manager:
            if hasattr(tool, "name") and hasattr(tool, "description"):
                available_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                        },
                    }
                )
        return available_tools

    def _get_tools_description(self) -> str:
        """Get a formatted description of available tools."""
        descriptions = []
        for tool in self.tool_manager:
            if hasattr(tool, "name") and hasattr(tool, "description"):
                descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions) if descriptions else "No tools available"

    async def run(self) -> str | None:
        """Run the agent loop until completion, final answer, or error.

        Returns:
            Final answer if provided, None otherwise

        Raises:
            AgentError: If the loop is already running or handlers are missing
        """
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

                # Observe phase
                try:
                    logger.debug("Executing observe phase")
                    observation = await self.observe_handler()
                    iteration_data["observation"] = observation
                    logger.info(
                        f"Observation collected: {observation['content'][:100]}..."
                    )
                except Exception as e:
                    logger.error(f"Error in observe phase: {e}")
                    raise AgentError(f"Observation phase failed: {e}") from e

                # Plan phase
                try:
                    logger.debug("Executing plan phase")
                    plan = await self.plan_handler(observation)
                    iteration_data["plan"] = plan
                    logger.info(f"Plan created with {len(plan['actions'])} actions")
                except Exception as e:
                    logger.error(f"Error in plan phase: {e}")
                    raise AgentError(f"Planning phase failed: {e}") from e

                # Act phase
                try:
                    logger.debug("Executing act phase")
                    action_result = await self.act_handler(plan)
                    iteration_data["action_result"] = action_result
                    logger.info(
                        f"Action completed: {'success' if action_result['success'] else 'failed'}"
                    )

                except Exception as e:
                    logger.error(f"Error in act phase: {e}")
                    action_result = {
                        "success": False,
                        "error": str(e),
                        "metadata": {"plan_actions": plan["actions"]},
                    }
                    iteration_data["action_result"] = action_result
                    raise AgentError(f"Action phase failed: {e}") from e

                # Check if final answer was provided
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
                    break

                # Reflect phase
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

                self.history.append(iteration_data)

                # Check if we should continue based on action success
                if not action_result["success"]:
                    logger.info("Stopping due to action failure")
                    break

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
        """
        return self.history.copy()

    def is_loop_running(self) -> bool:
        """Check if the loop is currently running.

        Returns:
            True if the loop is running, False otherwise
        """
        return self.is_running
