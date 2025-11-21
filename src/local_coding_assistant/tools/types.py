"""Shared data structures and enums for tool management."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolSource(str, Enum):
    """Source of the tool implementation."""

    BUILTIN = "builtin"
    EXTERNAL = "external"
    MCP = "mcp"


class ToolPermission(str, Enum):
    """Permissions required by tools."""

    READ = "read"
    WRITE = "write"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    COMPUTE = "compute"
    LOCAL = "local"
    SANDBOX = "sandbox"
    ENV_VARS = "env_vars"


class ToolCategory(str, Enum):
    """Categories for organizing tools."""

    MATH = "math"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    DATABASE = "database"
    SEARCH = "search"
    CODING = "coding"
    UTILITY = "utility"
    AI = "ai"
    OTHER = "other"


class ToolTag(str, Enum):
    """Tags for additional tool metadata."""

    RETRIEVAL = "retrieval"
    WEB = "web"
    MATH = "math"
    KNOWLEDGE = "knowledge"
    AI = "ai"
    DATA = "data"
    DEBUG = "debug"
    VERSION_CONTROL = "version_control"
    SECURITY = "security"
    UTILITY = "utility"


@dataclass
class ToolInfo:
    """Runtime representation of a tool's metadata and configuration.

    Attributes:
        name: Unique identifier for the tool
        class_name: Name of the tool's class (set during registration)
        description: Human-readable description of the tool
        category: Category the tool belongs to
        source: Source of the tool (builtin, external, mcp, etc.)
        permissions: List of permissions required by the tool
        tags: List of tags for the tool
        is_async: Whether the tool's run method is async
        supports_streaming: Whether the tool supports streaming output
        has_input_validation: Whether the tool validates its input
        has_output_validation: Whether the tool validates its output
        enabled: Whether the tool is enabled and should be loaded
        available: Whether the tool was successfully loaded and is ready to use
        endpoint: Endpoint URL for remote tools (MCP tools)
        provider: Provider name for MCP or external tools
        config: Tool-specific configuration options
    """

    name: str
    tool_class: type | None = None  # Store the actual tool class
    description: str = ""
    category: ToolCategory | None = None
    source: ToolSource | str = ToolSource.BUILTIN
    permissions: list[ToolPermission | str] = field(default_factory=list)
    tags: list[ToolTag | str] = field(default_factory=list)
    is_async: bool = False
    supports_streaming: bool = False
    has_input_validation: bool = False
    has_output_validation: bool = False
    enabled: bool = True
    available: bool = False
    endpoint: str | None = None
    provider: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert string enums to proper enum types."""
        if self.category is not None and isinstance(self.category, str):
            self.category = ToolCategory(self.category)
        if self.source is not None and isinstance(self.source, str):
            self.source = ToolSource(self.source)
        self.permissions = [
            ToolPermission(p) if isinstance(p, str) else p
            for p in (self.permissions or [])
        ]
        self.tags = [ToolTag(t) if isinstance(t, str) else t for t in (self.tags or [])]


class ToolExecutionRequest(BaseModel):
    """Request model for executing a tool.

    Attributes:
        tool_name: Name of the tool to execute
        payload: Input data for the tool
    """

    tool_name: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResponse(BaseModel):
    """Response model from tool execution.

    Attributes:
        tool_name: Name of the executed tool
        success: Whether the execution was successful
        result: Output from the tool (if successful)
        error_message: Error message (if execution failed)
        execution_time_ms: Time taken to execute the tool in milliseconds
    """

    tool_name: str
    success: bool
    result: dict[str, Any] | None = None
    error_message: str | None = None
    execution_time_ms: float | None = None


@dataclass
class ToolValidationResult:
    """Result of tool validation.

    Attributes:
        has_input_validation: Whether the tool has input validation
        has_output_validation: Whether the tool has output validation
    """

    has_input_validation: bool
    has_output_validation: bool
