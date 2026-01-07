"""Security manager for the sandbox environment."""

from __future__ import annotations

import ast
import re

from .exceptions import SandboxSecurityError, SandboxToolImportError


class SecurityManager:
    """Manages security checks for sandbox execution."""

    def __init__(
        self,
        allowed_imports: list[str] | None = None,
        blocked_patterns: list[str] | None = None,
        blocked_shell_commands: list[str] | None = None,
    ):
        self.allowed_imports = (
            set(allowed_imports) if allowed_imports is not None else set()
        )

        # Use defaults only if None is provided. If empty list is provided, it means no blocks.
        if blocked_patterns is None:
            self.blocked_patterns = [
                r"exec\(",
                r"eval\(",
                r"__import__",
                r"open\(",
                r"subprocess",
                r"os\.system",
                r"os\.popen",
            ]
        else:
            self.blocked_patterns = blocked_patterns

        if blocked_shell_commands is None:
            self.blocked_shell_commands = [
                "rm",
                "mv",
                "cp",
                "chmod",
                "chown",
                "wget",
                "curl",
                "nc",
                "bash",
                "sh",
            ]
        else:
            self.blocked_shell_commands = blocked_shell_commands

    def validate_code(self, code: str, skip_validation: bool = False) -> None:
        """Validate code against security rules.

        Args:
            code: The code to validate
            skip_validation: If True, skip validation (use with caution)

        Raises:
            SandboxSecurityViolation: If code violates security rules.
        """
        if skip_validation:
            return

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, code):
                raise SandboxSecurityError(f"Code contains blocked pattern: {pattern}")

        # Rest of the validation remains the same...
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # If syntax is invalid, let the execution fail naturally
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._check_import(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self._check_import(node.module)

    def _check_import(self, module_name: str) -> None:
        """Check if an import is allowed."""
        if not self.allowed_imports:
            return  # If no allowlist, allow all (or default deny? usually allow standard lib)

        # Simple check: is the top-level module allowed?
        top_level = module_name.split(".")[0]
        if top_level not in self.allowed_imports:
            raise SandboxToolImportError(f"Import not allowed: {module_name}")

    def validate_command(self, command: str) -> None:
        """Validate shell command."""
        cmd_parts = command.split()
        if not cmd_parts:
            return

        base_cmd = cmd_parts[0]
        if base_cmd in self.blocked_shell_commands:
            raise SandboxSecurityError(f"Shell command not allowed: {base_cmd}")
