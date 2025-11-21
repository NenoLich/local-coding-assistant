#!/usr/bin/env python3
"""Test script for refactored environment variable system."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_env_manager():
    """Test the EnvManager functionality."""
    print("=== Testing EnvManager ===")

    from local_coding_assistant.config.env_manager import EnvManager

    # Test 1: Basic functionality
    env_manager = EnvManager()
    config = env_manager.get_config_from_env()
    print("✓ EnvManager created successfully")
    print(f"✓ Default config from env: {config}")

    # Test 2: Environment variable parsing
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
        config = env_manager.get_config_from_env()
        print(f"✓ Environment parsing works: {config}")

        # Verify the parsed values
        assert config["llm"]["model_name"] == "gpt-4"
        assert config["llm"]["temperature"] == 0.8
        assert config["runtime"]["persistent_sessions"]
        print("✓ Type conversion works correctly")

        # Test prefix handling
        assert env_manager.with_prefix("TEST_KEY") == "LOCCA_TEST_KEY"
        assert env_manager.without_prefix("LOCCA_TEST_KEY") == "TEST_KEY"
        print("✓ Prefix handling works correctly")

        # Test environment variable management
        env_manager.set_env("API_KEY", "test123")
        assert os.environ["LOCCA_API_KEY"] == "test123"
        assert env_manager.get_env("API_KEY") == "test123"

        env_manager.unset_env("API_KEY")
        assert "LOCCA_API_KEY" not in os.environ
        print("✓ Environment variable management works")

        # Test get_all_env_vars
        all_vars = env_manager.get_all_env_vars()
        assert all(k.startswith("LOCCA_") for k in all_vars)
        print("✓ get_all_env_vars works correctly")

    finally:
        # Restore environment
        for key in test_env:
            if old_env.get(key) is None:
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

        # Access config through the config_manager
        config = runtime.config_manager.resolve()
        print(f"✓ Runtime manager created with config: {type(config)}")
        assert config is not None

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
    success &= test_env_manager()
    success &= test_bootstrap_integration()

    if success:
        print("\n🎉 Environment variable refactoring completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
