"""Simple script to test OpenAI integration manually."""

import asyncio
import os
import sys
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from local_coding_assistant.agent.llm_manager import LLMConfig, LLMManager, LLMRequest


@pytest.mark.asyncio
async def test_openai_integration():
    """Test OpenAI integration with API key from environment."""
    print("🔧 Testing OpenAI Integration")
    print("=" * 50)

    # Load .env file first
    try:
        load_dotenv()
    except UnicodeDecodeError:
        # If .env file has encoding issues, skip loading it
        print("⚠️  Warning: .env file has encoding issues, skipping load")

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LOCCA_LLM__API_KEY")

    if not api_key:
        print("❌ No OpenAI API key found in environment variables")
        print("💡 Add either OPENAI_API_KEY or LOCCA_LLM__API_KEY to your .env file")
        return False

    print(
        f"✅ Found API key: {api_key[:10]}..."
        if len(api_key) > 10
        else f"✅ Found API key: {api_key}"
    )

    try:
        # Create LLMManager with real API key
        print("\n🚀 Creating LLMManager...")
        config = LLMConfig(
            model_name="gpt-3.5-turbo",
            provider="openai",
            api_key=api_key,
            max_tokens=100,
        )

        llm = LLMManager(config)
        print("✅ LLMManager created successfully")

        # Test a simple generation
        print("\n🤖 Testing API call...")
        request = LLMRequest(
            prompt="Write a very short hello world program in Python.",
            system_prompt="You are a helpful coding assistant. Keep responses brief.",
        )

        response = await llm.generate(request)

        print("✅ API call successful!")
        print(f"📝 Response: {response.content}")
        print(f"🎯 Model used: {response.model_used}")
        print(f"🔢 Tokens used: {response.tokens_used}")

        return True

    except Exception as e:
        print(f"❌ Error during OpenAI integration test: {e}")

        # Provide helpful error messages
        error_str = str(e).lower()
        if "authentication" in error_str or "invalid" in error_str:
            print(
                "💡 This looks like an API key issue. Please check your OpenAI API key."
            )
        elif "model" in error_str:
            print("💡 This looks like a model availability issue. Try gpt-3.5-turbo.")
        else:
            print("💡 Check your internet connection and OpenAI service status.")

        return False


def main():
    """Main function to run the test."""
    print("🧪 OpenAI Integration Test")
    print("=" * 50)

    success = asyncio.run(test_openai_integration())

    if success:
        print("\n🎉 OpenAI integration test PASSED!")
        print("✨ Your Local Coding Assistant is ready to use OpenAI!")
    else:
        print("\n❌ OpenAI integration test FAILED!")
        print("🔧 Please check your configuration and try again.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
