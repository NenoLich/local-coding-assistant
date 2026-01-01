"""Shared data structures and enums for tool management."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolSource(str, Enum):
    """Source of the tool implementation."""

    BUILTIN = "builtin"
    EXTERNAL = "external"
    MCP = "mcp"
    SANDBOX = "sandbox"


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
    PTC = "ptc"
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
    FILESYSTEM = "filesystem"


@dataclass
class ToolInfo:
    """Runtime representation of a tool's metadata and configuration.

    Attributes:
        name: Unique identifier for the tool
        tool_class: Name of the tool's class (set during registration)
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
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )

    def __str__(self) -> str:
        """Return a JSON-formatted string representation of the ToolInfo with all fields.

        Returns:
            A JSON-formatted string showing all fields and their values with proper indentation.
        """
        # Create a clean dictionary of the object's attributes
        data = {}
        for field_name, field_value in self.__dict__.items():
            # Convert enums to their values for better readability
            if hasattr(field_value, "value"):
                data[field_name] = field_value.value
            elif isinstance(field_value, (list, tuple)):
                # Handle lists of enums
                data[field_name] = [
                    item.value if hasattr(item, "value") else item
                    for item in field_value
                ]
            else:
                data[field_name] = field_value

        # Convert to nicely formatted JSON with 2-space indentation
        return f"{self.__class__.__name__}:\n{json.dumps(data, indent=2, default=str)}"

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
    tool_type: str = "function"  # "function" or "code"
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResponse(BaseModel):
    """Response model from tool execution.

    Attributes:
        tool_name: Name of the executed tool
        success: Whether the execution was successful
        result: Output from the tool (if successful)
        error_message: Error message (if execution failed)
        execution_time_ms: Time taken to execute the tool in milliseconds
        format: Format of the response (e.g., 'text', 'markdown', 'json')
        metadata: Additional metadata about the response
        is_final: Whether this is the final response
    """

    tool_name: str
    success: bool
    result: Any | None = None
    error_message: str | None = None
    execution_time_ms: float | None = None
    stdout: str | None = None
    stderr: str | None = None
    files_created: list[str] | None = None
    files_modified: list[str] | None = None
    format: str = "text"
    metadata: dict[str, Any] = {}
    is_final: bool = False

    def dried_out(self) -> dict[str, Any]:
        output = {}
        for field_name, field_value in self.__dict__.items():
            if (
                field_name
                not in ["tool_name", "success", "format", "metadata", "is_final"]
                and field_value
            ):
                output[field_name] = field_value
        return output


@dataclass
class ToolValidationResult:
    """Result of tool validation.

    Attributes:
        has_input_validation: Whether the tool has input validation
        has_output_validation: Whether the tool has output validation
    """

    has_input_validation: bool
    has_output_validation: bool
