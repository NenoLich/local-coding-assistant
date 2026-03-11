# Agent Patterns

LOCCA implements advanced agent patterns for complex reasoning and task execution.

## Current Agent Implementation

### Frame Agent
**Recommended**: The primary agent implementation for all new projects.

Flexible execution framework for custom agent patterns:

- Configurable reasoning loops
- Extensible action handlers
- State management across iterations

## Deprecated Agent Implementations

!!! warning "Deprecated Agents"
    The following agents are deprecated and will be removed in a future version.
    Use FrameAgent for all new implementations.

### AgentLoop (Deprecated)
Legacy observe-plan-act-reflect loop implementation.

- **Status**: Deprecated
- **Replacement**: Use FrameAgent instead
- **Removal**: Planned for future version

### LangGraph Agent (Deprecated)
LangGraph-based agent implementation.

- **Status**: Deprecated  
- **Replacement**: Use FrameAgent instead
- **Removal**: Planned for future version

## Migration Guide

To migrate from deprecated agents to FrameAgent:

1. Replace `AgentLoop(...)` with `FrameAgent(...)`
2. Replace `LangGraphAgent(...)` with `FrameAgent(...)`
3. Update any agent-specific configuration
4. Test functionality with FrameAgent

## Agent Implementations

### Frame Agent
Flexible execution framework for custom agent patterns:

- Configurable reasoning loops
- Extensible action handlers
- State management across iterations

## Tool Integration

Agents seamlessly integrate with the tool system:

- Automatic tool discovery and selection
- Context-aware tool invocation
- Result processing and feedback loops

## Session Awareness

Agents maintain context across interactions:

- Persistent conversation history
- User preference learning
- Progressive task refinement

## Error Handling

Robust error recovery mechanisms:

- Graceful failure handling
- Retry logic with backoff
- Alternative strategy selection

## Customization

Extend and customize agent behavior:

- Custom reasoning patterns
- Specialized tool sets
- Domain-specific adaptations

## Examples

### Code Review Agent
```python
# Agent configured for code review tasks
agent = FrameAgent()
result = agent.run("Review this pull request for security issues")
```

### Documentation Agent
```python
# Agent specialized in documentation generation
agent = FrameAgent()
docs = agent.run("Generate API documentation for this codebase")
```

## Next Steps

- [Tool System](tool-system.md)
- [CLI Usage](cli-usage.md)
