"""Local Coding Assistant - An extensible AI coding assistant framework."""

__version__ = "0.1.0"

# Core modules
from . import agent, cli, config, core, runtime, tools, utils

# Main LLM Manager
from .agent.llm_manager import LLMManager

__all__ = ["LLMManager", "agent", "cli", "config", "core", "runtime", "tools", "utils"]
