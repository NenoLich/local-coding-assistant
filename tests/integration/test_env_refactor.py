#!/usr/bin/env python3
"""Test script for refactored environment variable system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_env_loader():
    """Test the new EnvLoader functionality."""
    print("=== Testing EnvLoader ===")

    from local_coding_assistant.config.env_loader import EnvLoader

    # Test 1: Basic functionality
    env_loader = EnvLoader()
    config = env_loader.get_config_from_env()
    print(f"✓ EnvLoader created successfully")
    print(f"✓ Default config from env: {config}")

    # Test 2: Environment variable parsing
    import os

    test_env = {
        "LOCCA_LLM__MODEL_NAME": "gpt-4",
        "LOCCA_LLM__TEMPERATURE": "0.8",
        "LOCCA_RUNTIME__PERSISTENT_SESSIONS": "true",
    }

    old_env = {}
    for key in test_env:
        old_env[key] = os.environ.get(key)
        os.environ[key] = test_env[key]

    try:
        config = env_loader.get_config_from_env()
        print(f"✓ Environment parsing works: {config}")

        # Verify the parsed values
        assert config["llm"]["model_name"] == "gpt-4"
        assert config["llm"]["temperature"] == 0.8
        assert config["runtime"]["persistent_sessions"] == True
        print("✓ Type conversion works correctly")

    finally:
        # Restore environment
        for key in test_env:
            if old_env[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_env[key]

    return True


def test_bootstrap_integration():
    """Test bootstrap integration with new .env system."""
    print("\n=== Testing Bootstrap Integration ===")

    try:
        from local_coding_assistant.core.bootstrap import bootstrap

        ctx = bootstrap()
        print("✓ Bootstrap loads successfully with .env integration")

        # Check that configuration loaded properly
        runtime = ctx.get("runtime")
        assert runtime is not None
        print(f"✓ Runtime manager created with config: {type(runtime.config)}")

        return True

    except Exception as e:
        print(f"❌ Bootstrap error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Testing refactored environment variable system...\n")

    success = True
    success &= test_env_loader()
    success &= test_bootstrap_integration()

    if success:
        print("\n🎉 Environment variable refactoring completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
