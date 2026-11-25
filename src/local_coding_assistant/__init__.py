"""Local Coding Assistant - An extensible AI coding assistant framework."""

from local_coding_assistant.config.env_manager import get_env_manager

# Load environment files implicitly on package import so env vars are
# available regardless of whether bootstrap() is called.
get_env_manager()

__version__ = "0.1.0"
