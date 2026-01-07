"""Sandbox-specific exception taxonomy."""

from __future__ import annotations

from local_coding_assistant.core.exceptions import LocalAssistantError


class SandboxError(LocalAssistantError):
    """Base exception for sandbox subsystem errors."""

    subsystem = "sandbox"


class SandboxRuntimeError(SandboxError):
    """Raised for runtime or infrastructure level sandbox failures."""


class SandboxTimeoutError(SandboxRuntimeError):
    """Raised when sandbox execution exceeds the allowed time."""

    def __init__(self, message: str, *, return_code: int | None = None) -> None:
        self.return_code = return_code
        super().__init__(message)


class SandboxSecurityError(SandboxError):
    """Raised when sandbox security policies are violated."""


class SandboxToolImportError(SandboxSecurityError):
    """Raised when sandbox tool imports violate policy."""


class SandboxOutputFormatError(SandboxRuntimeError):
    """Raised when sandbox output cannot be parsed."""

    def __init__(
        self,
        message: str,
        *,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(message)
