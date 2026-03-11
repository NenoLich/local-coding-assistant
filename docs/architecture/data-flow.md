# Data Flow

This document describes how data flows through LOCCA's components and contracts.

## Request Processing Flow

```mermaid
graph TD
    A[User Input] --> B[CLI Parser]
    B --> C[Agent Request]
    C --> D[Agent Loop]
    D --> E[LLM Service]
    E --> F[Provider Router]
    F --> G[LLM Provider]
    G --> H[LLM Response]
    H --> I[Agent Response]
    I --> J[User Output]

    D --> K{Tool Needed?}
    K -->|Yes| L[Tool Call]
    L --> M[Execution Request]
    M --> N[Sandbox Manager]
    N --> O[Tool Execution]
    O --> P[Execution Result]
    P --> Q[Tool Result]
    Q --> D
    K -->|No| I
```

## Key Data Transformations

### Input Processing
- Raw CLI args → Structured `AgentRequest`
- User queries → Parsed intent and context

### LLM Interaction
- `AgentRequest` → `LLMRequest` with provider-specific formatting
- Raw LLM output → Parsed `LLMResponse` with metadata

### Tool Execution
- LLM tool calls → `ToolCall` objects
- `ToolCall` → `ToolExecutionRequest` using sandbox if available and enabled
- Tool output → `ToolExecutionResponse` with metrics

### Response Generation
- Agent reasoning + tool results → `AgentResponse`
- `AgentResponse` → Formatted user output

## State Management

### Session Data Flow
```mermaid
graph LR
    A[Session Start] --> B[Session Context]
    B --> C[Agent Interactions]
    C --> D[Context Updates]
    D --> E[Session Persistence]
    E --> F[Session End]

    C --> G[Memory Storage]
    G --> H[Future Retrieval]
```

### Configuration Flow
```mermaid
graph TD
    A[YAML Files] --> B[Config Parser]
    B --> C[Validation]
    C --> D[Merged Config]
    D --> E[Component Injection]
    E --> F[Runtime Components]
```

## Error Handling Flow

```mermaid
graph TD
    A[Error Occurs] --> B{Error Type}
    B -->|Provider| C[Retry/Fallback]
    B -->|Tool| D[Tool Recovery]
    B -->|System| E[System Recovery]
    C --> F[Alternative Path]
    D --> F
    E --> F
    F --> G[Continue Processing]
    F --> H[Graceful Failure]
```

## Performance Monitoring

### Metrics Collection
- Execution times for each component
- Resource usage (CPU, memory)
- API call statistics
- Error rates and recovery success

### Data Flow Tracking
- Request tracing through components
- Bottleneck identification
- Optimization opportunities

## Security Boundaries

### Data Isolation
- User data segregated by session
- Tool execution in sandboxed environments
- API keys encrypted and isolated

### Access Control
- Component-level permissions
- Sandbox resource limits
- Network access restrictions

## Extensibility Points

### Custom Data Flows
- Plugin interfaces for new components
- Custom data transformers
- Alternative routing logic

### Monitoring Integration
- Custom metrics collectors
- External logging systems
- Performance dashboards

## Next Steps

- [Components](components.md)
- [Contracts](../architecture/contracts.md)
