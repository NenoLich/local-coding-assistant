# OpenAI Integration Testing

This directory contains tests and scripts to verify that OpenAI integration is working correctly.

## Prerequisites

1. **OpenAI API Key**: You need a valid OpenAI API key
2. **Environment Setup**: Add your API key to `.env` file
3. **Dependencies**: Install OpenAI package: `pip install openai`
4. **Dotenv Support**: Tests automatically load `.env` files using python-dotenv

## Environment Variables

You can use either format in your `.env` file:

```bash
# Option 1: Standard OpenAI format
OPENAI_API_KEY=your_api_key_here

# Option 2: Structured config format
LOCCA_LLM__API_KEY=your_api_key_here
```

## Running Tests

### 1. Pytest Integration Tests

```bash
# Run all OpenAI integration tests
pytest tests/integration/test_openai_integration.py -v

# Run specific test
pytest tests/integration/test_openai_integration.py::TestOpenAIIntegration::test_openai_api_key_from_env -v

# Run with verbose output
pytest tests/integration/test_openai_integration.py -v -s
```

### 2. Standalone Integration Test

```bash
# Run the standalone test script
python test_openai_integration.py

# Run from project root
python -m local_coding_assistant.test_openai_integration
```

## Test Coverage

The integration tests cover:

- ✅ **Environment Loading**: Automatically loads `.env` files before checking API keys
- ✅ **API Key Loading**: Verifies API key is loaded from environment variables
- ✅ **LLMManager Creation**: Tests successful initialization with real API key
- ✅ **Simple Generation**: Makes actual API call to test functionality
- ✅ **Error Handling**: Tests behavior with invalid API keys
- ✅ **Configuration**: Verifies model and parameter configuration

## Expected Behavior

### When API Key is Valid:
- Tests should pass and show successful API responses
- Real API calls are made to OpenAI
- Response content is validated

### When API Key is Missing/Invalid:
- Tests skip gracefully if no API key is found
- Tests fail with appropriate error messages for invalid keys
- No real API calls are made

## Troubleshooting

### Common Issues:

1. **No API Key Found**
   ```bash
   pytest.skip("No OpenAI API key found in environment variables")
   ```

2. **Invalid API Key**
   ```bash
   AgentError: OpenAI API error: authentication
   ```

3. **Model Not Available**
   ```bash
   AgentError: OpenAI API error: model
   ```

4. **Network Issues**
   ```bash
   AgentError: OpenAI API error: connection
   ```

### Debug Tips:

1. **Check Environment Variables**:
   ```bash
   echo $OPENAI_API_KEY
   echo $LOCCA_LLM__API_KEY
   ```

2. **Test API Key Directly**:
   ```bash
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}' \
        https://api.openai.com/v1/chat/completions
   ```

3. **Check OpenAI Status**: Visit [status.openai.com](https://status.openai.com)

## Security Notes

⚠️ **Important**: The integration tests make real API calls and will consume your OpenAI credits. Run them judiciously.

🔒 **API Key Security**: Never commit API keys to version control. The `.env` file should be in `.gitignore`.
