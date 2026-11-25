"""Typed run options shared between front-ends and the RuntimeManager.

``RunOptions`` is the front-end-neutral contract for per-run configuration
options (model, agent mode, tool-call mode, sandbox session). The
RuntimeManager is the single place that translates these options into
config-layer session overrides, so every front-end (CLI, chat, future
server) applies identical semantics and config key knowledge stays in
the runtime layer instead of leaking into front-ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RunOptions:
    """Configuration-affecting options for one run.

    Attributes:
        model: LLM model name override (``llm.model_name``)
        agent_mode: Runtime agent mode (``runtime.agent_mode``)
        tool_call_mode: Tool call mode; ``"ptc"`` implies sandbox execution
            (``runtime.tool_call_mode`` + ``sandbox.*``)
        sandbox_session: Named sandbox session to reuse (``sandbox.session_id``)
        auto_approve: Auto-approve all tool calls (``tools_platform.auto_approve``)
    """

    model: str | None = None
    agent_mode: str | None = None
    tool_call_mode: str | None = None
    sandbox_session: str | None = None
    auto_approve: bool | None = None

    def to_session_overrides(self) -> dict[str, Any]:
        """Translate the options into config session overrides.

        Returns:
            Dot-notation override mapping for
            ``IConfigManager.set_session_overrides``; empty when no option
            is set.
        """
        overrides: dict[str, Any] = {}

        if self.agent_mode is not None:
            overrides["runtime.agent_mode"] = self.agent_mode
        if self.tool_call_mode is not None:
            overrides["runtime.tool_call_mode"] = self.tool_call_mode
            if self.tool_call_mode == "ptc":
                overrides["sandbox.enabled"] = True
                if self.sandbox_session:
                    overrides["sandbox.session_id"] = self.sandbox_session
                    overrides["sandbox.persistence"] = True
        if self.model is not None:
            overrides["llm.model_name"] = self.model
        if self.auto_approve is not None:
            overrides["tools_platform.auto_approve"] = self.auto_approve

        return overrides
