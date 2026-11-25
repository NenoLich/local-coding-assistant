"""Central tool registration for Phase 1 read-only navigation tools, Phase 2 workspace tools, and Phase 3b command tools.

This module provides functions to register tools for each phase.
"""

from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.tools.check_process_status import (
    register_check_process_status,
)
from local_coding_assistant.tools_platform.tools.create_file import register_create_file
from local_coding_assistant.tools_platform.tools.delete_file import register_delete_file
from local_coding_assistant.tools_platform.tools.get_symbol import register_get_symbol
from local_coding_assistant.tools_platform.tools.kill_process import (
    register_kill_process,
)
from local_coding_assistant.tools_platform.tools.match_paths import register_match_paths
from local_coding_assistant.tools_platform.tools.read_file import register_read_file
from local_coding_assistant.tools_platform.tools.replace_exact import (
    register_replace_exact,
)
from local_coding_assistant.tools_platform.tools.run_command import register_run_command
from local_coding_assistant.tools_platform.tools.search_symbols import (
    register_search_symbols,
)
from local_coding_assistant.tools_platform.tools.search_text import register_search_text


def register_phase1_tools(registry: ToolRegistry) -> None:
    """Register all Phase 1 read-only navigation tools.

    Args:
        registry: ToolRegistry instance
    """
    register_search_text(registry)
    register_match_paths(registry)
    register_read_file(registry)
    register_search_symbols(registry)
    register_get_symbol(registry)


def register_phase2_tools(registry: ToolRegistry) -> None:
    """Register all Phase 2 workspace and exact editing tools.

    Args:
        registry: ToolRegistry instance
    """
    register_replace_exact(registry)
    register_create_file(registry)
    register_delete_file(registry)


def register_phase3b_tools(registry: ToolRegistry) -> None:
    """Register all Phase 3b command execution tools.

    Args:
        registry: ToolRegistry instance
    """
    register_run_command(registry)
    register_check_process_status(registry)
    register_kill_process(registry)


def register_all_tools(registry: ToolRegistry) -> None:
    """Register all tools from all phases.

    Args:
        registry: ToolRegistry instance
    """
    register_phase1_tools(registry)
    register_phase2_tools(registry)
    register_phase3b_tools(registry)
