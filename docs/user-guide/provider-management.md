# Provider Management

LOCCA supports multiple LLM providers with automatic routing and fallback.

## Supported Providers

### OpenAI
```bash
locca provider add openai --api-key-env OPENAI_API_KEY --models gpt-4 gpt-3.5-turbo
```

### Anthropic
```bash
locca provider add anthropic --api-key-env ANTHROPIC_API_KEY --models claude-3-sonnet claude-3-haiku
```

### Local Models
```bash
locca provider add ollama --base-url http://localhost:11434 --models llama2 codellama
```

## Provider Configuration

### YAML Configuration
```yaml
# config/providers.local.yaml
providers:
  openai:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    models:
      gpt-3.5-turbo-instruct:
        supported_parameters:
        - max_tokens
        - temperature
        - top_p
      gpt-4o:
        supported_parameters:
        - max_tokens
        - temperature
        - top_p
    timeout: 30
    retries: 3
```

### CLI Management
```bash
# Add provider
locca provider add openai --api-key-env OPENAI_API_KEY

# List providers
locca provider list

# Remove provider
locca provider remove openai

# Validate configurations
locca provider validate
```

## Routing and Fallback

### Provider Selection
LOCCA automatically selects providers based on:

- Model availability
- Health status
- Performance metrics
- Cost preferences

### Fallback Logic
If primary provider fails:

1. Retry with same provider
2. Switch to backup provider
3. Degrade gracefully

## Health Monitoring

### Automatic Checks
Providers are monitored for:

- API availability
- Response times
- Error rates
- Token limits

### Manual Testing
```bash
# Test provider connectivity
locca provider test openai

# Check health status
locca provider health
```

## Cost Management

### Usage Tracking
Monitor API usage and costs:

- Token consumption
- Request counts
- Cost estimation

### Optimization
- Model selection based on cost
- Caching for repeated queries
- Batch processing

## Custom Providers

### Implementing Providers
Extend the Provider base class:

```python
from local_coding_assistant.providers.base import Provider

class MyProvider(Provider):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        # Implementation here
        pass
```

### Registration
Add custom providers to configuration.

## Security

### API Key Management
- Environment variables only
- No hardcoded keys
- Secure storage options

### Request Validation
- Input sanitization
- Rate limiting
- Abuse prevention

## Performance

### Connection Pooling
Efficient provider connections:

- Persistent connections
- Connection reuse
- Load balancing

### Caching
Response caching for:

- Identical requests
- Model metadata
- Provider configurations

## Next Steps

- [Tool System](tool-system.md)
- [CLI Usage](cli-usage.md)
