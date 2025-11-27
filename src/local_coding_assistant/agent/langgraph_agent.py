"""LangGraph-based agent implementation for observe-plan-act-reflect lifecycle."""

import json
import time
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StreamWriter
from pydantic import BaseModel, Field

from local_coding_assistant.agent.llm_manager import LLMManager, LLMRequest
from local_coding_assistant.core.exceptions import AgentError
from local_coding_assistant.core.protocols import IToolManager
from local_coding_assistant.tools.types import ToolExecutionRequest
from local_coding_assistant.utils.langgraph_utility import (
    handle_graph_error,
    node_logger,
    safe_node,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("agent.langgraph")


# State definition for the LangGraph
class AgentState(BaseModel):
    """State class for the LangGraph agent execution.

    This state is passed between nodes and contains all information
    about the current execution context.
    """

    iteration: int = Field(default=0, description="Current iteration number")
    max_iterations: int = Field(default=10, description="Maximum iterations allowed")
    final_answer: str | None = Field(
        default=None, description="Final answer when determined"
    )
    history: list[dict[str, Any]] = Field(
        default_factory=list, description="Execution history"
    )
    session_id: str = Field(
        default_factory=lambda: f"langgraph_agent_{int(time.time())}",
        description="Session identifier",
    )
    error: dict[str, Any] | None = Field(
        default=None, description="Current error information"
    )
    current_phase: str = Field(default="observe", description="Current execution phase")
    current_observation: dict[str, Any] | None = Field(
        default=None, description="Current observation data"
    )
    current_plan: dict[str, Any] | None = Field(
        default=None, description="Current plan data"
    )
    current_action: dict[str, Any] | None = Field(
        default=None, description="Current action result"
    )
    current_reflection: dict[str, Any] | None = Field(
        default=None, description="Current reflection data"
    )
    user_input: str | None = Field(default=None, description="Original user input")

    def should_continue(self) -> bool:
        """Check if the graph should continue execution."""
        # Stop if we have a final answer
        if self.final_answer is not None:
            return False

        # Stop if we've reached max iterations
        if self.iteration >= self.max_iterations:
            return False

        # Stop if there's a critical error
        if self.error and self.error.get("should_stop", False):
            return False

        return True

    def increment_iteration(self) -> None:
        """Increment the current iteration counter."""
        self.iteration += 1

    def set_final_answer(self, answer: str) -> None:
        """Set the final answer and stop execution."""
        self.final_answer = answer

    def add_to_history(self, phase: str, data: dict[str, Any]) -> None:
        """Add phase data to execution history."""
        self.history.append(
            {
                "phase": phase,
                "iteration": self.iteration,
                "timestamp": time.time(),
                "data": data,
            }
        )

    class Config:
        """Pydantic configuration for AgentState."""

        arbitrary_types_allowed = True


class LangGraphAgent:
    """LangGraph-based agent that replaces AgentLoop functionality.

    This agent implements the observe-plan-act-reflect cycle using LangGraph
    for better state management and execution control.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        tool_manager: IToolManager | None = None,
        name: str = "langgraph_agent",
        max_iterations: int = 10,
        streaming: bool = False,
    ):
        """Initialize the LangGraph agent.

        Args:
            llm_manager: The LLM manager to use for reasoning.
            tool_manager: Optional tool manager providing available tools. If not provided,
                        the agent will operate in a tool-less mode.
            name: A name for this agent instance.
            max_iterations: Maximum number of iterations to run.
            streaming: Whether to use streaming LLM responses.
        """
        if max_iterations < 1:
            raise AgentError("max_iterations must be at least 1")

        self.llm_manager = llm_manager
        self.tool_manager = tool_manager
        self.name = name
        self.max_iterations = max_iterations
        self.streaming = streaming

        # Cache tools for performance
        self._cached_tools = self._get_available_tools()

        # Build the graph
        self.graph = self._build_graph()
        logger.info(
            f"Initialized LangGraph agent '{name}' with max_iterations={max_iterations}"
        )

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
                descriptions.append(f"- {tool.name}: {tool.name}: {tool.description}")
        return "\n".join(descriptions) if descriptions else "No tools available"

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph with all nodes and edges."""
        # Create state graph
        workflow = StateGraph(AgentState)

        # Create wrapper functions for LangGraph nodes
        async def observe_wrapper(state: AgentState) -> AgentState:
            """Wrapper for observe node that LangGraph can call."""
            return await self.observe_node(state)

        async def plan_wrapper(state: AgentState) -> AgentState:
            """Wrapper for plan node that LangGraph can call."""
            return await self.plan_node(state)

        async def act_wrapper(state: AgentState) -> AgentState:
            """Wrapper for act node that LangGraph can call."""
            return await self.act_node(state)

        async def reflect_wrapper(state: AgentState) -> AgentState:
            """Wrapper for reflect node that LangGraph can call."""
            return await self.reflect_node(state)

        # Add nodes using wrapper functions
        workflow.add_node("observe", observe_wrapper)
        workflow.add_node("plan", plan_wrapper)
        workflow.add_node("act", act_wrapper)
        workflow.add_node("reflect", reflect_wrapper)

        # Add conditional edges
        workflow.add_conditional_edges(
            "observe",
            self._should_continue_from_observe,
            {
                "plan": "plan",
                END: END,
            },
        )

        workflow.add_conditional_edges(
            "plan",
            self._should_continue_from_plan,
            {
                "act": "act",
                END: END,
            },
        )

        workflow.add_conditional_edges(
            "act",
            self._should_continue_from_act,
            {
                "reflect": "reflect",
                END: END,
            },
        )

        workflow.add_conditional_edges(
            "reflect",
            self._should_continue_from_reflect,
            {
                "observe": "observe",
                END: END,
            },
        )

        # Set entry point
        workflow.set_entry_point("observe")

        # Compile the graph
        return workflow.compile()

    def _should_continue_from_observe(self, state: AgentState) -> str:
        """Determine next step after observe node."""
        if not state.should_continue():
            return END
        return "plan"

    def _should_continue_from_plan(self, state: AgentState) -> str:
        """Determine next step after plan node."""
        if not state.should_continue():
            return END
        return "act"

    def _should_continue_from_act(self, state: AgentState) -> str:
        """Determine next step after act node."""
        if not state.should_continue():
            return END
        return "reflect"

    def _should_continue_from_reflect(self, state: AgentState) -> str:
        """Determine next step after reflect node."""
        if not state.should_continue():
            return END
        # Increment iteration before going back to observe
        state.increment_iteration()
        return "observe"

    @safe_node("observe")
    async def observe_node(self, state: AgentState, writer: StreamWriter) -> AgentState:
        """Observe node - generate observation based on current context."""
        node_logger_instance = node_logger("observe")

        # Create observation based on current state
        content = f"Agent iteration {state.iteration + 1} in session {state.session_id}"

        observation_data = {
            "content": content,
            "metadata": {
                "session_id": state.session_id,
                "iteration": state.iteration + 1,
            },
            "timestamp": time.time(),
            "source": "langgraph_agent",
        }

        # Add to state and history
        state.current_observation = observation_data
        state.add_to_history("observe", observation_data)

        node_logger_instance.info(f"Generated observation: {content}")

        # Stream observation data if streaming is enabled
        if self.streaming:
            writer({"phase": "observe", "data": observation_data})

        return state

    @safe_node("plan")
    async def plan_node(self, state: AgentState, writer: StreamWriter) -> AgentState:
        """Plan node - use LLM to create a plan."""
        node_logger_instance = node_logger("plan")

        observation = state.current_observation or {}
        prompt = f"""
Based on the following observation, create a plan for what the agent should do next:

Observation: {observation.get("content", "")}

Context:
- Session ID: {observation.get("metadata", {}).get("session_id", "unknown")}
- Iteration: {observation.get("metadata", {}).get("iteration", 0)}

Please provide a plan with specific actions to take. Respond in JSON format with:
- reasoning: your reasoning for the plan
- actions: list of specific actions to take
- confidence: confidence level (0-1)
"""

        try:
            # Get LLM response using unified method
            response_content = await self._get_llm_response(prompt, writer, "plan")

            # Parse response as JSON for structured output
            try:
                plan_data = json.loads(response_content)
                reasoning = plan_data.get("reasoning", "Failed to parse reasoning")
                actions = plan_data.get("actions", ["retry"])
                confidence = plan_data.get("confidence", 0.5)
            except json.JSONDecodeError:
                # Fallback if LLM doesn't return JSON
                reasoning = (
                    f"Based on observation: {observation.get('content', '')[:100]}..."
                )
                actions = ["analyze_current_state", "determine_next_action"]
                confidence = 0.8

            plan_result = {
                "reasoning": reasoning,
                "actions": actions,
                "confidence": confidence,
                "metadata": {"llm_response": response_content},
            }

            # Add to state and history
            state.current_plan = plan_result
            state.add_to_history("plan", plan_result)

            node_logger_instance.info(
                f"Created plan with {len(actions)} actions, confidence: {confidence}"
            )

            # Stream plan result if streaming is enabled
            if self.streaming:
                writer({"phase": "plan", "data": plan_result})

            return state

        except Exception as e:
            node_logger_instance.error(f"Failed to generate plan: {e}")
            # Fallback plan
            fallback_plan = {
                "reasoning": "Failed to generate plan, using fallback",
                "actions": ["retry_planning"],
                "confidence": 0.3,
                "metadata": {"error": str(e)},
            }

            state.current_plan = fallback_plan
            state.add_to_history("plan", fallback_plan)

            if self.streaming:
                writer({"phase": "plan", "data": fallback_plan, "error": str(e)})

            return state

    async def _get_llm_response_with_tools(
        self, prompt: str, writer: StreamWriter, phase: str
    ) -> tuple[str, list[dict[str, Any]]]:
        """Get LLM response and extract tool calls.

        Unified method to interact with llm_manager.stream/ainvoke.

        Args:
            prompt: The prompt to send to LLM
            writer: Stream writer for streaming output
            phase: The current phase (for streaming metadata)

        Returns:
            Tuple of (response_content, tool_calls)
        """
        request = LLMRequest(
            prompt=prompt,
            tools=self._cached_tools,
        )

        if self.streaming:
            # Use astream for streaming mode
            response_content = ""
            async for chunk in self.llm_manager.stream(request):
                response_content += chunk
                # Stream LLM tokens if streaming is enabled
                writer({"phase": phase, "type": "llm_token", "content": chunk})

            # For streaming mode, we need to get tool calls from complete response
            response = await self.llm_manager.ainvoke(request)
            tool_calls = response.tool_calls or []
        else:
            # Use ainvoke for non-streaming mode
            response = await self.llm_manager.ainvoke(request)
            response_content = response.content
            tool_calls = response.tool_calls or []

        return response_content, tool_calls

    async def _get_llm_response(
        self, prompt: str, writer: StreamWriter, phase: str
    ) -> str:
        """Get LLM response without tool calls.

        Unified method for nodes that don't need tool calling (plan, reflect).

        Args:
            prompt: The prompt to send to LLM
            writer: Stream writer for streaming output
            phase: The current phase (for streaming metadata)

        Returns:
            Response content as string
        """
        request = LLMRequest(prompt=prompt)

        if self.streaming:
            # Use stream for streaming mode
            response_content = ""
            async for chunk in self.llm_manager.stream(request):
                response_content += chunk
                # Stream LLM tokens if streaming is enabled
                writer({"phase": phase, "type": "llm_token", "content": chunk})
        else:
            # Use ainvoke for non-streaming mode
            response = await self.llm_manager.ainvoke(request)
            response_content = response.content

        return response_content

    async def _process_tool_calls(
        self, tool_calls: list[dict[str, Any]], plan: dict[str, Any], state: AgentState
    ) -> dict[str, Any] | None:
        """Process tool calls and return action result.

        Args:
            tool_calls: List of tool calls to process
            plan: Current plan data
            state: Current agent state

        Returns:
            Action result dict or None if no tool calls to process
        """
        if not tool_calls:
            return None

        for tool_call in tool_calls:
            if "function" in tool_call:
                await self._handle_tool_call(tool_call, plan, state)

        return None

    async def _handle_tool_call(
        self, tool_call: dict[str, Any], plan: dict[str, Any], state: AgentState
    ) -> dict[str, Any]:
        """Handle execution of a single tool call."""
        func_name = tool_call["function"]["name"]
        try:
            # Parse arguments
            args = json.loads(tool_call["function"]["arguments"])

            # Handle tool execution
            if self.tool_manager is None:
                tool_result = f"Error: No tool manager available to execute {func_name}"
            else:
                try:
                    request = ToolExecutionRequest(tool_name=func_name, payload=args)
                    response = await self.tool_manager.execute_async(request)
                    if not response.success:
                        tool_result = f"Error: {response.error_message}"
                    else:
                        tool_result = response.result
                except Exception as e:
                    tool_result = f"Error executing tool {func_name}: {e!s}"

            # Special handling for final_answer tool
            if func_name == "final_answer":
                # Set final answer on state and return action result
                state.set_final_answer(args.get("answer", ""))

                return {
                    "success": True,
                    "output": f"Final answer: {args.get('answer', '')}",
                    "metadata": {
                        "tool_calls": [tool_call],
                        "stopped": True,
                    },
                }
            else:
                return {
                    "success": True,
                    "output": f"Tool {func_name} executed successfully",
                    "metadata": {
                        "tool_calls": [tool_call],
                        "tool_results": {func_name: tool_result},
                    },
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "metadata": {"plan_actions": plan.get("actions", [])},
            }

    def _create_action_result(
        self, response_content: str, tool_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Create action result from LLM response."""
        return {
            "success": True,
            "output": response_content,
            "metadata": {"tool_calls": tool_calls},
        }

    @safe_node("act")
    async def act_node(self, state: AgentState, writer: StreamWriter) -> AgentState:
        """Act node - execute actions using tools and LLM."""
        node_logger_instance = node_logger("act")

        plan = state.current_plan or {}

        try:
            # Create a prompt that includes tool calling instructions
            action_prompt = f"""
Execute the following plan:

Plan: {plan.get("reasoning", "")}
Actions: {", ".join(plan.get("actions", []))}

You have access to the following tools:
{self._get_tools_description()}

If you need to use a tool, respond with a function call in the format:
{{"function_call": {{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}}}

If you need to provide a final answer, use the final_answer tool.

Please describe what actions were taken and their results.
"""

            # Get LLM response and tool calls
            response_content, tool_calls = await self._get_llm_response_with_tools(
                action_prompt, writer, "act"
            )

            # Process tool calls if any
            action_result = await self._process_tool_calls(tool_calls, plan, state)

            # If no tool calls were processed, create default action result
            if action_result is None:
                action_result = self._create_action_result(response_content, tool_calls)

            # Add to state and history
            state.current_action = action_result
            state.add_to_history("act", action_result)

            node_logger_instance.info(
                f"Action completed: {'success' if action_result['success'] else 'failed'}"
            )

            # Stream action result if streaming is enabled
            if self.streaming:
                writer({"phase": "act", "data": action_result})

            return state

        except Exception as e:
            node_logger_instance.error(f"Failed to execute actions: {e}")
            action_result = {
                "success": False,
                "error": str(e),
                "metadata": {"plan_actions": plan.get("actions", [])},
            }

            state.current_action = action_result
            state.add_to_history("act", action_result)

            if self.streaming:
                writer({"phase": "act", "data": action_result, "error": str(e)})

            return state

    @safe_node("reflect")
    async def reflect_node(self, state: AgentState, writer: StreamWriter) -> AgentState:
        """Reflect node - analyze results and learn."""
        node_logger_instance = node_logger("reflect")

        action_result = state.current_action or {}
        plan = state.current_plan or {}

        try:
            reflection_prompt = f"""
Analyze the following action results and provide reflection:

Plan: {plan.get("reasoning", "")}
Actions: {", ".join(plan.get("actions", []))}
Success: {action_result.get("success", False)}
Output: {action_result.get("output", "")}
Error: {action_result.get("error", "")}

Please provide:
- analysis: analysis of what happened
- lessons_learned: key lessons
- improvements: suggested improvements
- success_rating: success rating (0-1)
"""

            # Get LLM response using unified method
            response_content = await self._get_llm_response(
                reflection_prompt, writer, "reflect"
            )

            # Parse response as JSON for structured output
            try:
                reflection_data = json.loads(response_content)
                analysis = reflection_data.get("analysis", "Failed to parse analysis")
                lessons_learned = reflection_data.get("lessons_learned", [])
                improvements = reflection_data.get("improvements", [])
                success_rating = reflection_data.get("success_rating", 0.5)
            except json.JSONDecodeError:
                # Fallback if LLM doesn't return JSON
                analysis = f"Plan execution {'succeeded' if action_result.get('success') else 'failed'}: {response_content[:200]}"
                lessons_learned = [
                    "Monitor action results carefully",
                    "Adjust plans based on outcomes",
                ]
                improvements = (
                    ["Better error handling", "More detailed action planning"]
                    if not action_result.get("success")
                    else []
                )
                success_rating = 0.9 if action_result.get("success") else 0.3

            reflection_result = {
                "analysis": analysis,
                "lessons_learned": lessons_learned,
                "improvements": improvements,
                "success_rating": success_rating,
            }

            # Add to state and history
            state.current_reflection = reflection_result
            state.add_to_history("reflect", reflection_result)

            node_logger_instance.info(
                f"Reflection completed with success rating: {success_rating}"
            )

            # Stream reflection result if streaming is enabled
            if self.streaming:
                writer({"phase": "reflect", "data": reflection_result})

            return state

        except Exception as e:
            node_logger_instance.error(f"Failed to generate reflection: {e}")
            reflection_result = {
                "analysis": "Failed to generate reflection",
                "success_rating": 0.0,
                "lessons_learned": ["Improve reflection process"],
                "improvements": [],
            }

            state.current_reflection = reflection_result
            state.add_to_history("reflect", reflection_result)

            if self.streaming:
                writer({"phase": "reflect", "data": reflection_result, "error": str(e)})

            return state

    async def run(self, initial_state: AgentState | None = None) -> str | None:
        """Run the LangGraph agent until completion.

        Args:
            initial_state: Optional initial state, if None a new state is created

        Returns:
            Final answer if provided, None otherwise
        """
        # Create initial state if not provided
        if initial_state is None:
            state = AgentState()
            state.max_iterations = self.max_iterations
        else:
            state = initial_state

        logger.info(f"Starting LangGraph agent '{self.name}'")

        try:
            # Run the graph using ainvoke for non-streaming mode
            final_state = await self.graph.ainvoke(state)

            # Type check to ensure final_state has iteration attribute
            iterations = getattr(final_state, "iteration", "unknown")
            logger.info(
                f"LangGraph agent '{self.name}' completed after {iterations} iterations"
            )

            # Safely access final_answer with a default of None
            return getattr(final_state, "final_answer", None)

        except Exception as e:
            logger.error(f"LangGraph agent '{self.name}' failed: {e}")
            # Ensure state is a dict for error handling
            state_dict = (
                state.model_dump() if hasattr(state, "model_dump") else {"state": state}
            )
            handle_graph_error(e, state_dict)
            return None

    async def run_stream(self, initial_state: AgentState | None = None):
        """Run the LangGraph agent with streaming output.

        Args:
            initial_state: Optional initial state, if None a new state is created

        Yields:
            Intermediate state updates during execution
        """
        # Create initial state if not provided
        if initial_state is None:
            state = AgentState()
            state.max_iterations = self.max_iterations
        else:
            state = initial_state

        logger.info(f"Starting LangGraph agent '{self.name}' in streaming mode")

        try:
            # Use astream for streaming mode
            async for state_update, metadata in self.graph.astream(
                state, stream_mode="custom"
            ):
                yield state_update, metadata

        except Exception as e:
            logger.error(f"LangGraph agent '{self.name}' streaming failed: {e}")
            # Ensure state is a dict for error handling
            state_dict = (
                state.model_dump() if hasattr(state, "model_dump") else {"state": state}
            )
            handle_graph_error(e, state_dict)
            yield state

    def get_history(self) -> list[dict[str, Any]]:
        """Get the complete execution history.

        Returns:
            List of phase data from execution history
        """
        # This would need to be implemented based on how state is managed
        # For now, return empty list
        return []

    def get_current_state(self) -> AgentState:
        """Get the current state of the agent.

        Returns:
            Current agent state
        """
        # This would need to be implemented based on how state is managed
        # For now, return a new empty state
        return AgentState()
