"""Session manager for the sandbox environment."""

import contextlib
import io
import json
import time
import traceback
from typing import Any

from pydantic import ValidationError

try:
    from resource_tracker import tracker
except ImportError:
    # Fallback for relative import if run as a module
    from .resource_tracker import tracker


class Session:
    """Represents a single execution context."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._final_answer_data = None  # Store final answer here
        self.globals = {
            "__builtins__": __builtins__,
            "tracker": tracker,  # Make tracker available in the execution context
        }
        self.created_at = time.time()
        self.last_accessed = self.created_at

    def touch(self):
        """Update last accessed time."""
        self.last_accessed = time.time()

    def is_expired(self, timeout: int) -> bool:
        """Check if session is expired."""
        return (time.time() - self.last_accessed) > timeout

    def _setup_io_buffers(self):
        """Set up I/O buffers for capturing stdout and stderr."""
        return io.StringIO(), io.StringIO()

    def _reset_tracker(self):
        """Reset the resource tracker if available."""
        if "tracker" in self.globals and hasattr(self.globals["tracker"], "reset"):
            self.globals["tracker"].reset()

    def _compile_and_execute(self, code: str, local_vars: dict) -> bool:
        """Compile and execute the provided code.

        Args:
            code: The code to execute
            local_vars: Dictionary to store local variables

        Returns:
            bool: True if execution was successful, False otherwise
        """
        try:
            compiled = compile(code, "<string>", "exec")
            exec(compiled, self.globals, local_vars)  # noqa S102
            return True
        except SyntaxError as se:
            raise SyntaxError(
                f"Syntax error: {se.msg} (line {se.lineno}, offset {se.offset})"
            ) from se

    def _update_globals(self, local_vars: dict):
        """Update global variables with new ones from local execution."""
        for name, value in local_vars.items():
            if not name.startswith("_"):  # Skip internal variables
                self.globals[name] = value

    def _extract_error_message(self, error: Exception) -> str:
        """Extract a user-friendly error message from an exception."""
        if hasattr(error, "__module__") and error.__module__.startswith("pydantic"):
            return self._handle_pydantic_error(error)
        return f"{type(error).__name__}: {error!s}"

    def _handle_pydantic_error(self, error: ValidationError | Exception) -> str:
        """Handle Pydantic validation errors.

        Args:
            error: Either a Pydantic ValidationError or another Exception

        Returns:
            Formatted error message
        """
        if not hasattr(error, "errors") or not getattr(error, "errors", None):
            return str(error).split("\n")[0]

        field_errors = []
        for err in error.errors():  # type: ignore[union-attr]
            loc = ".".join(str(_loc) for _loc in err.get("loc", []))
            msg = err.get("msg", "Unknown error")
            field_errors.append(f"{loc}: {msg}" if loc else msg)
        return "; ".join(field_errors)

    def _get_resource_metrics(self) -> dict[str, Any]:
        """Get resource metrics from the tracker if available."""
        if "tracker" in self.globals and hasattr(
            self.globals["tracker"], "get_metrics"
        ):
            return self.globals["tracker"].get_metrics() or {}
        return {}

    def _extract_final_answer(self, resource_metrics: dict) -> Any:
        """Extract final answer from resource metrics if available."""
        for tool_call in resource_metrics.get("tool_calls", []):
            if tool_call.get("tool_name") == "final_answer" and tool_call.get(
                "success", False
            ):
                return tool_call.get("result")
        return None

    def _format_stdout(self, stdout: str, final_answer: Any) -> str:
        """Format the standard output, using final_answer if available."""
        if final_answer is not None and not stdout:
            try:
                return (
                    json.dumps(final_answer, indent=2)
                    if isinstance(final_answer, (dict, list))
                    else str(final_answer)
                )
            except (TypeError, ValueError):
                pass
        return stdout

    def execute(self, code: str) -> dict[str, Any]:
        """Execute code in this session.

        Returns:
            dict: A dictionary containing execution results and metadata
        """
        self.touch()
        stdout, stderr = self._setup_io_buffers()
        result = None
        error = None
        success = False

        self._reset_tracker()

        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                local_vars = {}
                if self._compile_and_execute(code, local_vars):
                    self._update_globals(local_vars)
                    success = True
        except Exception as e:
            error = self._extract_error_message(e)
            traceback.print_exc(file=stderr)

        # Process execution results
        resource_metrics = self._get_resource_metrics()
        final_answer = self._extract_final_answer(resource_metrics)
        stdout_str = self._format_stdout(stdout.getvalue().strip(), final_answer)
        stderr_str = stderr.getvalue().strip()

        return {
            "success": success,
            "result": result,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "error": error,
            "final_answer": final_answer,
            "metrics_per_tool_call": resource_metrics,
        }


class SessionManager:
    """Manages sessions in the sandbox."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get_session(self, session_id: str) -> Session:
        """Get or create a session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id)
        return self._sessions[session_id]

    def cleanup_expired(self, timeout: int) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        expired = [sid for sid, s in self._sessions.items() if s.is_expired(timeout)]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)
