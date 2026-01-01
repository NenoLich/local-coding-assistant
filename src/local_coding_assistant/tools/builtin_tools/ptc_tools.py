"""Sandbox wrapper tools used for Programmatic Tool Calling (PTC)."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from local_coding_assistant.sandbox.manager import SandboxManager
from local_coding_assistant.sandbox.sandbox_types import (
    SandboxExecutionRequest,
    SandboxExecutionResponse,
)
from local_coding_assistant.tools.tool_registry import register_tool
from local_coding_assistant.tools.types import (
    ToolCategory,
    ToolPermission,
    ToolTag,
)

# Define type variables for the Input and Output models
Input = TypeVar("Input", bound=BaseModel)
Output = TypeVar("Output", bound=BaseModel)


class SandboxTool(ABC):
    """Base class for tools that need to execute code in a sandbox."""

    sandbox_manager: SandboxManager | None = None

    @abstractmethod
    async def run(self, input_data: Input) -> Output:
        """Execute the tool's main functionality.

        This method must be implemented by all subclasses.

        Args:
            input_data: The input parameters for the tool, already validated against Input model

        Returns:
            The result of the tool execution, must match the Output model

        Raises:
            ToolExecutionError: If the tool execution fails
        """
        pass

    def _require_sandbox(self) -> Any:
        if self.sandbox_manager is None:
            raise RuntimeError("Sandbox manager not available")

        config = getattr(self.sandbox_manager, "config", None)
        if not getattr(config, "enabled", False):
            raise RuntimeError("Sandbox execution is disabled in configuration")

        sandbox = self.sandbox_manager.get_sandbox()
        if sandbox is None:
            raise RuntimeError("Sandbox instance is not available")

        return sandbox

    async def execute_in_sandbox(
        self,
        code: str,
        session_id: str = "default",
        timeout: int = 30,
        env_vars: dict[str, str] | None = None,
        persistence: bool = False,
    ) -> SandboxExecutionResponse:
        """Execute code in the sandbox.

        Args:
            code: The Python code to execute
            session_id: Session ID for state persistence
            timeout: Execution timeout in seconds
            env_vars: Environment variables to set in the sandbox
            persistence: Whether to persist the sandbox state

        Returns:
            The sandbox execution response

        Raises:
            RuntimeError: If sandbox execution fails
        """
        sandbox = self._require_sandbox()

        # Create and execute the request
        request = SandboxExecutionRequest(
            code=code,
            session_id=session_id,
            timeout=timeout,
            env_vars=env_vars or {},
            persistence=persistence,
        )

        return await sandbox.execute(request)


@register_tool(
    name="execute_python_code",
    description=(
        "Execute Python code inside the isolated sandbox. "
        "Use this to run Python that orchestrates other tools or performs computations."
    ),
    category=ToolCategory.PTC,
    tags=[ToolTag.UTILITY, ToolTag.SECURITY],
    permissions=[ToolPermission.SANDBOX, ToolPermission.COMPUTE],
    is_async=True,
)
class ExecutePythonCodeTool(SandboxTool):
    """Execute Python code in the sandbox with defense-in-depth validation."""

    class Input(BaseModel):
        """Input model for Python code execution."""

        code: str = Field(
            ...,
            description="""
Python code to execute. It can import from the Python standard library and any pre-approved tools.
Example:
```python
# Simple calculation
result = 2 + 2
from tools_api import final_answer
final_answer(f"The result is {result}")

# Using tools
from tools_api import read_file, process_data

data = read_file("test.txt")
processed = process_data(data)
final_answer(processed)
```""",
        )
        session_id: str = Field(
            default="default",
            description="""Session ID for maintaining state between executions. 
        Using the same session_id across executions preserves variables and imports.
        Use different session_ids for independent executions.""",
        )

    class Output(BaseModel):
        """Output model for sandbox execution."""

        response: SandboxExecutionResponse

    async def run(self, input_data: Input) -> Output:
        """Execute Python code in the sandbox.

        Args:
            input_data: The input parameters for code execution

        Returns:
            The execution result with output and status
        """
        try:
            # Validate code if security manager is available
            sandbox = self._require_sandbox()
            security_manager = getattr(sandbox, "security_manager", None)
            if security_manager:
                security_manager.validate_code(input_data.code)

            persistence = self.sandbox_manager.config.persistence  # type: ignore[possibly-missing-attribute]

            # Execute the code in the sandbox
            response = await self.execute_in_sandbox(
                code=input_data.code,
                session_id=input_data.session_id,
                timeout=self.sandbox_manager.config.timeout,  # type: ignore[possibly-missing-attribute]
                env_vars=self._get_env_vars(input_data.session_id),
                persistence=persistence,
            )

            return self.Output(response=response)

        except Exception as e:
            return self.Output(
                response=SandboxExecutionResponse(
                    success=False, error=str(e), stderr=str(e)
                )
            )

    def _get_env_vars(self, session_id: str) -> dict[str, str]:
        manager = self.sandbox_manager
        if manager is None:
            return {}

        resolver = getattr(manager, "get_session_env", None)
        if callable(resolver):
            return resolver(session_id) or {}

        return {}


@register_tool(
    name="run_shell_command",
    description=(
        "Execute a shell command inside the sandbox environment. "
        "Use this to run system commands in a controlled environment."
    ),
    category=ToolCategory.PTC,
    tags=[ToolTag.UTILITY, ToolTag.SECURITY],
    permissions=[ToolPermission.SANDBOX],
    is_async=True,
)
class RunShellCommandTool(SandboxTool):
    """Execute shell commands with sandbox-level enforcement."""

    class Input(BaseModel):
        """Input model for shell command execution."""

        command: str = Field(..., description="Shell command to execute")
        session_id: str = Field(
            default="default",
            description="Session ID for maintaining state between command executions",
        )

        cwd: str | None = Field(
            None,
            description="Working directory for the command (default: sandbox root)",
        )

    class Output(BaseModel):
        """Output model for command execution."""

        success: bool
        return_code: int
        stdout: str = ""
        stderr: str = ""
        error: str | None = None

    async def run(self, input_data: Input) -> Output:
        """Execute a shell command in the sandbox.

        This method first tries to use the sandbox's native shell execution if available,
        otherwise falls back to a Python subprocess wrapper.

        Args:
            input_data: The command and execution parameters

        Returns:
            The command execution results
        """
        try:
            sandbox = self._require_sandbox()

            # Validate command if security manager is available
            security_manager = getattr(sandbox, "security_manager", None)
            if security_manager:
                security_manager.validate_command(input_data.command)

            # 1. Try native shell execution if available
            if hasattr(sandbox, "execute_shell") and asyncio.iscoroutinefunction(
                sandbox.execute_shell
            ):
                response = await sandbox.execute_shell(
                    command=input_data.command,
                    session_id=input_data.session_id,
                    timeout=self.sandbox_manager.config.timeout,  # type: ignore[possibly-missing-attribute]
                )
                return self.Output(
                    success=response.success,
                    return_code=getattr(
                        response, "return_code", 0 if response.success else 1
                    ),
                    stdout=getattr(response, "stdout", ""),
                    stderr=getattr(response, "stderr", ""),
                    error=getattr(response, "error", None),
                )

            # 2. Fall back to Python subprocess wrapper
            code = await self._build_subprocess_wrapper(
                command=input_data.command,
                timeout=self.sandbox_manager.config.timeout,  # type: ignore[possibly-missing-attribute]
                cwd=input_data.cwd,
            )

            response = await self.execute_in_sandbox(
                code=code,
                session_id=input_data.session_id,
                timeout=self.sandbox_manager.config.timeout,  # type: ignore[possibly-missing-attribute]
                persistence=bool(
                    input_data.session_id and input_data.session_id != "default"
                ),
            )

            # Parse the return code from the subprocess wrapper
            return_code = 0
            if not response.success:
                return_code = 1
                if "Command timed out" in response.stderr:
                    return_code = 124

            return self.Output(
                success=response.success,
                return_code=return_code,
                stdout=response.stdout,
                stderr=response.stderr,
                error=response.error,
            )

        except Exception as e:
            return self.Output(
                success=False,
                return_code=-1,
                error=str(e),
                stderr=str(e),
            )

    @staticmethod
    async def _build_subprocess_wrapper(
        command: str, timeout: int, cwd: str | None = None
    ) -> str:
        """Build Python code to execute a shell command with timeout and output capture.

        Args:
            command: The shell command to execute
            timeout: Command timeout in seconds
            cwd: Working directory for the command

        Returns:
            Python code as a string that will execute the command
        """

        return f"""
import subprocess
import sys

try:
    result = subprocess.run(
        {command!r},
        shell=True,
        cwd={repr(cwd) if cwd else "None"},
        capture_output=True,
        text=True,
        timeout={timeout}
    )
    print(result.stdout, end='')
    print(result.stderr, file=sys.stderr, end='')
    sys.exit(result.returncode)

# Handle command timeout
except subprocess.TimeoutExpired:
    print('Command timed out', file=sys.stderr)
    sys.exit(124)

# Handle other errors
except Exception as exc:
    print(f'Error: {{exc}}', file=sys.stderr)
    sys.exit(1)
"""
