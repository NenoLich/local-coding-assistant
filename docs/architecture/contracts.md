# Data Contracts & Schemas

This document outlines the key data contracts and Pydantic models used throughout LOCCA. These contracts define the structure of data exchanged between components, ensuring type safety and consistent interfaces.

## Overview

LOCCA uses Pydantic models extensively for data validation, serialization, and type hints. All data contracts follow these principles:

- **Type Safety**: Full type annotations with modern Python syntax (`|` for unions)
- **Validation**: Automatic validation and error messages
- **Serialization**: JSON-compatible serialization for APIs and persistence
- **Documentation**: Clear field descriptions and examples

## Core Request/Response Contracts

### LLM Service Contracts

#### `LLMRequest` (`src/local_coding_assistant/agent/llm/models.py`)
Request structure for LLM calls:

```python
class LLMRequest(BaseModel):
    messages: list[LLMMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    tools: list[ToolDefinition] | None = None
    tool_choice: str | None = None
```

#### `LLMResponse` (`src/local_coding_assistant/agent/llm/models.py`)
Response structure from LLM calls:

```python
class LLMResponse(BaseModel):
    content: str
    usage: LLMUsage | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
```

### Tool System Contracts

#### `ToolExecutionRequest` (`src/local_coding_assistant/tools/types.py`)
Request structure for tool execution:

```python
class ToolExecutionRequest(BaseModel):
    tool_name: str
    tool_type: str = "function"  # "function" or "code"
    payload: dict[str, Any] = Field(default_factory=dict)
```

#### `ToolExecutionResponse` (`src/local_coding_assistant/tools/types.py`)
Response structure from tool execution:

```python
class ToolExecutionResponse(BaseModel):
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    success: bool
    result: Any | None = None
    error_message: str | None = None
    execution_time_ms: float | None = None
    is_final: bool = False
    envelope: ExecutionEnvelope | None = None
    tool_calls: list[ToolCallTrace] | None = None
    output: PresentationOutput | None = None
```

#### `ToolInfo` (`src/local_coding_assistant/tools/types.py`)
Tool metadata and configuration:

```python
@dataclass
class ToolInfo:
    name: str
    tool_class: type | None = None
    description: str = ""
    category: ToolCategory | None = None
    source: ToolSource | str = ToolSource.BUILTIN
    execution_mode: ToolExecutionMode | str = ToolExecutionMode.CLASSIC
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
```

### Agent Contracts

Agent implementations use standardized data contracts for input and output to ensure consistency and type safety.

#### `AgentRequest` (`src/local_coding_assistant/runtime/agent_types.py`)

Input contract for all agent execution modes:

```python
class AgentRequest(BaseModel):
    user_input: str
    session: SessionState
    agent_mode: str | None = None
    model_override: str | None = None
    temperature_override: float | None = None
    max_tokens_override: int | None = None
    tool_call_mode_override: str | None = None
    sandbox_session_override: str | None = None
    streaming: bool | None = None
    max_iterations: int | None = None
```

**Purpose**: Separates per-request parameters from global configuration defaults, enabling flexible overrides while maintaining consistent agent interfaces.

#### `RunReport` (`src/local_coding_assistant/runtime/reporting.py`)

Standardized output contract for all agent execution results:

```python
class RunReport(BaseModel):
    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4()}")
    session_id: str | None = None
    mode: str = "regular"
    status: str = "success"
    final_answer: str | None = None
    message: str | None = None
    finish_reason: str | None = None
    models_used: list[str] = Field(default_factory=list)
    tokens_used: int | None = None
    iterations: int | None = None
    frames: list[dict[str, Any]] | None = None
    history: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    metrics: RunMetrics | None = None
    errors: list[RunError] = Field(default_factory=list)
    events: list[RuntimeEvent] = Field(default_factory=list)
```

**Purpose**: Provides comprehensive, agent-agnostic reporting of execution results, including final answers, metrics, and agent-specific data.

The main agent implementation uses execution frames:

**Key Classes:**
- `FrameAgent` (`src/local_coding_assistant/agent/frame_agent.py`) - Main agent class
- `ExecutionFrame` (`src/local_coding_assistant/runtime/execution_types.py`) - Execution frame structure
- `ExecutionEvent` (`src/local_coding_assistant/runtime/events.py`) - Event streaming

**Note**: Frame Agent does not use traditional request/response contracts but operates through execution frames and events.

### Configuration Contracts

#### `ProviderConfig` (`src/local_coding_assistant/config/schemas.py`)
Provider configuration schema:

```python
class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key_env: str | None = None
    models: list[str]
    health_check_interval: int = 60
    timeout: int = 30
    retries: int = 3
```

#### `ToolConfig` (`src/local_coding_assistant/config/schemas.py`)
Tool configuration schema:

```python
class ToolConfig(BaseModel):
    name: str
    enabled: bool = True
    path: str | None = None
    module: str | None = None
    config: dict[str, Any] | None = None
```

### Runtime Contracts

#### `ToolSpec` (`src/local_coding_assistant/runtime/runtime_types.py`)
Structured tool representation for prompt composition:

```python
class ToolSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
```

#### `ExecutionMode` (`src/local_coding_assistant/runtime/runtime_types.py`)
Supported execution modes:

```python
class ExecutionMode(str, Enum):
    REASONING_ONLY = "reasoning_only"
    CLASSIC_TOOLS = "classic_tools"
    SANDBOX_PYTHON = "sandbox_python"
    SANDBOX_SHELL = "sandbox_shell"
```

### Session Management Contracts

#### `Session` (`src/local_coding_assistant/runtime/session.py`)
Session state:

```python
class Session(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    context: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
```

#### `SessionContext` (`src/local_coding_assistant/runtime/session.py`)
Context data within a session:

```python
class SessionContext(BaseModel):
    variables: dict[str, Any] = {}
    history: list[AgentInteraction] = []
    preferences: dict[str, Any] = {}
```

## Data Flow Patterns

### Primary Data Flows

1. **User Request Flow**:
   ```
   CLI/API → AgentRequest → Agent Loop → LLMRequest → Provider → LLMResponse → AgentResponse → User
   ```

2. **Tool Execution Flow**:
   ```
   Agent Loop → ToolCall → ExecutionRequest → Sandbox → ToolResult → Agent Loop
   ```

3. **Configuration Flow**:
   ```
   YAML/Env → ConfigManager → Validation → Component Configs → Runtime Components
   ```

### Key Transformation Points

- **Request Parsing**: Raw user input → structured `AgentRequest`
- **LLM Adaptation**: `AgentRequest` → provider-specific `LLMRequest`
- **Tool Invocation**: LLM tool calls → `ToolCall` → `ExecutionRequest`
- **Response Formatting**: Raw LLM output → structured `AgentResponse`

## Validation Rules

All contracts include validation rules:

- **Required Fields**: Clearly marked required vs optional
- **Type Constraints**: Strict type checking
- **Value Validation**: Custom validators for complex fields
- **Cross-field Validation**: Dependencies between fields

## Serialization Formats

- **JSON**: Primary serialization for APIs and persistence
- **YAML**: Configuration files
- **Internal**: Python objects with full type information

## Extension Points

Contracts are designed for extension:

- **Optional Fields**: Easy addition of new optional fields
- **Generic Types**: Support for custom data types
- **Inheritance**: Base contracts can be extended
- **Custom Validators**: Field-specific validation logic

## Best Practices

When working with contracts:

1. **Always use the models**: Prefer Pydantic models over plain dicts
2. **Validate early**: Parse and validate data at entry points
3. **Handle errors**: Provide meaningful error messages for validation failures
4. **Document changes**: Update this document when adding new contracts
5. **Test serialization**: Ensure models can be serialized/deserialized correctly
