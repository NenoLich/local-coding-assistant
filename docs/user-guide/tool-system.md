# Tool System

LOCCA's extensible tool system allows agents to perform actions beyond LLM capabilities.

## Tool Architecture

### Tool Definitions
Tools are defined with JSON schemas:

```python
class Calculator:
    name = "calculator"
    description = "Perform mathematical calculations"

    def run(self, expression: str) -> str:
        return str(eval(expression))
```

### Tool Registration
Tools are registered via configuration or code:

```yaml
# config/tools.local.yaml
tools:
  - name: calculator
    path: ./tools/calculator.py
    enabled: true
```

### Tool Execution
Tools run in secure sandbox environments:

- Isolated execution
- Resource limits
- Error handling

## Built-in Tools

### Math Tools
- Calculator for expressions
- Statistics functions
- Unit conversions

### File System Tools
- File reading/writing
- Directory operations
- Path manipulation

### System Tools
- Command execution
- Environment inspection
- Network operations

### Creating Tools

Tools do not need to inherit from a base `Tool` class, but must implement a `run` method and optionally a `stream` method. For detailed implementation details, refer to `src/local_coding_assistant/config/tool_loader.py`.

```python
# Example tool implementation
class MyCustomTool:
    """A custom tool that performs some operation."""

    def run(self, query: str, limit: int = 10) -> str:
        """Execute the tool's main logic.

        Args:
            query: The search query to process
            limit: Maximum number of results to return

        Returns:
            Formatted results as a string
        """
        # Tool implementation here
        return f"Results for '{query}' (limit: {limit})"
```

**Key Requirements:**
- Implement a `run` method (can be sync or async)
- Optionally implement a `stream` method for streaming responses
- Method parameters are automatically extracted for JSON schema generation
- Use type hints for proper parameter validation
- Docstrings are used to generate parameter descriptions

### Tool Validation
Tools are validated against schemas:

- Parameter type checking
- Required field validation
- Custom validators

## Tool Discovery

### Registry System
Tools are automatically discovered:

- Scan configured paths
- Load Python modules
- Validate implementations

### Dynamic Loading
Tools can be added at runtime:

```bash
locca tool add --name mytool --path ./mytool.py
locca tool reload
```

## Security

### Sandbox Execution
All tools run in isolated environments:

- Docker containers
- Resource constraints
- Network restrictions

### Input Validation
Strict input validation prevents malicious use.

## Performance

### Execution Statistics
Track tool performance:

- Execution time
- Resource usage
- Success rates

### Optimization
Tools are optimized for:

- Fast startup
- Low memory usage
- Concurrent execution

## Next Steps

- [Provider Management](provider-management.md)
- [CLI Usage](cli-usage.md)
