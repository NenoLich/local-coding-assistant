"""
Unit tests for core AppContext functionality.
"""

import pytest

from local_coding_assistant.core.app_context import AppContext


class TestAppContext:
    """Test AppContext functionality."""

    def test_initialization(self):
        """Test AppContext initialization."""
        ctx = AppContext()
        assert isinstance(ctx._resources, dict)
        assert len(ctx._resources) == 0

    def test_register_and_get(self):
        """Test registering and retrieving resources."""
        ctx = AppContext()

        # Register different types of resources
        ctx.register("string", "test_string")
        ctx.register("number", 42)
        ctx.register("list", [1, 2, 3])
        ctx.register("dict", {"key": "value"})
        ctx.register("none", None)

        # Test retrieval
        assert ctx.get("string") == "test_string"
        assert ctx.get("number") == 42
        assert ctx.get("list") == [1, 2, 3]
        assert ctx.get("dict") == {"key": "value"}
        assert ctx.get("none") is None

    def test_get_with_default(self):
        """Test get method with default values."""
        ctx = AppContext()

        # Test existing resource
        ctx.register("existing", "value")
        assert ctx.get("existing", "default") == "value"

        # Test non-existing resource
        assert ctx.get("nonexistent", "default") == "default"
        assert ctx.get("nonexistent", None) is None
        assert ctx.get("nonexistent", 42) == 42

    def test_get_nonexistent_without_default(self):
        """Test get method for non-existent resource without default."""
        ctx = AppContext()
        assert ctx.get("nonexistent") is None

    def test_dictionary_access(self):
        """Test dictionary-style access."""
        ctx = AppContext()
        ctx.register("test", "value")

        # Test __getitem__
        assert ctx["test"] == "value"

        # Test __contains__
        assert "test" in ctx
        assert "nonexistent" not in ctx

    def test_dictionary_access_missing_key(self):
        """Test dictionary-style access for missing key."""
        ctx = AppContext()

        with pytest.raises(KeyError):
            _ = ctx["nonexistent"]

    def test_resource_overwrite(self):
        """Test overwriting existing resources."""
        ctx = AppContext()

        ctx.register("test", "original")
        assert ctx.get("test") == "original"

        ctx.register("test", "updated")
        assert ctx.get("test") == "updated"

    def test_multiple_resources(self):
        """Test managing multiple resources."""
        ctx = AppContext()

        resources = {
            "service1": "Service One",
            "service2": "Service Two",
            "config": {"setting": "value"},
            "manager": None,
        }

        for name, resource in resources.items():
            ctx.register(name, resource)

        # Verify all resources
        for name, expected in resources.items():
            assert ctx.get(name) == expected
            assert name in ctx

        # Verify total count
        assert len(ctx._resources) == len(resources)

    def test_resource_isolation(self):
        """Test that resources don't interfere with each other."""
        ctx = AppContext()

        # Register resources with different types
        ctx.register("mutable_list", [1, 2, 3])
        ctx.register("mutable_dict", {"a": 1})

        original_list = ctx.get("mutable_list")
        original_dict = ctx.get("mutable_dict")

        # Type checker doesn't know these are not None, so assert they exist
        assert original_list is not None
        assert original_dict is not None

        # Modify retrieved objects
        original_list.append(4)
        original_dict["b"] = 2

        # Verify changes are reflected (shallow copy behavior)
        assert ctx.get("mutable_list") == [1, 2, 3, 4]
        assert ctx.get("mutable_dict") == {"a": 1, "b": 2}

    def test_empty_context_operations(self):
        """Test operations on empty context."""
        ctx = AppContext()

        # All get operations should return None or default
        assert ctx.get("any") is None
        assert ctx.get("any", "default") == "default"

        # No resources should exist
        assert len(ctx._resources) == 0
        assert "any" not in ctx

    def test_context_methods_consistency(self):
        """Test consistency between different access methods."""
        ctx = AppContext()
        ctx.register("test", "value")

        # All methods should return the same result
        assert ctx.get("test") == ctx["test"]
        assert ctx.get("test", "default") == ctx["test"]

        # Contains should match
        assert ("test" in ctx) == (ctx.get("test") is not None)

    def test_resource_naming(self):
        """Test resource naming conventions."""
        ctx = AppContext()

        # Test various valid names
        valid_names = [
            "simple",
            "with_numbers123",
            "with_underscores_and_123",
            "mixed_Case_123",
        ]

        for name in valid_names:
            ctx.register(name, f"value_{name}")
            assert ctx.get(name) == f"value_{name}"

        # Verify all names work
        for name in valid_names:
            assert name in ctx


class TestAppContextEdgeCases:
    """Test AppContext edge cases."""

    def test_register_none_resource(self):
        """Test registering None as a resource."""
        ctx = AppContext()

        ctx.register("none_resource", None)
        assert ctx.get("none_resource") is None
        assert "none_resource" in ctx

    def test_register_empty_string(self):
        """Test registering empty string."""
        ctx = AppContext()

        ctx.register("", "empty_name")
        assert ctx.get("") == "empty_name"
        assert "" in ctx

    def test_get_with_none_default(self):
        """Test get with None as default."""
        ctx = AppContext()

        # Non-existent resource with None default
        assert ctx.get("missing", None) is None

        # Existing None resource with None default
        ctx.register("none_resource", None)
        assert ctx.get("none_resource", "default") is None

    def test_chained_registration(self):
        """Test registering resources in sequence."""
        ctx = AppContext()

        # Register in sequence
        ctx.register("first", 1)
        ctx.register("second", 2)
        ctx.register("third", 3)

        # Verify all are accessible
        assert ctx.get("first") == 1
        assert ctx.get("second") == 2
        assert ctx.get("third") == 3

        # Verify order doesn't matter for access
        resources = {}
        for name in ["first", "second", "third"]:
            resources[name] = ctx.get(name)

        assert resources == {"first": 1, "second": 2, "third": 3}


class TestAppContextIntegration:
    """Test AppContext integration scenarios."""

    def test_bootstrap_like_usage(self):
        """Test usage pattern similar to bootstrap."""
        ctx = AppContext()

        # Simulate bootstrap registration
        ctx.register("llm", "MockLLMManager")
        ctx.register("tools", "MockToolManager")
        ctx.register("runtime", "MockRuntimeManager")
        ctx.register("config", {"test": "config"})

        # Test service access patterns
        assert ctx.get("llm") == "MockLLMManager"
        assert ctx.get("tools") == "MockToolManager"
        assert ctx.get("runtime") == "MockRuntimeManager"
        assert ctx.get("config") == {"test": "config"}

        # Test service availability checks
        assert "llm" in ctx
        assert "tools" in ctx
        assert "runtime" in ctx
        assert "nonexistent" not in ctx

        # Test dictionary-style access
        assert ctx["llm"] == "MockLLMManager"

    def test_service_dependency_simulation(self):
        """Test simulating service dependencies."""
        ctx = AppContext()

        # Register services in dependency order
        ctx.register("config", {"database_url": "sqlite:///test.db"})
        ctx.register("database", "MockDatabase")
        ctx.register("llm", "MockLLM")
        ctx.register("tools", "MockTools")

        # Simulate a service that depends on others
        def get_llm_service():
            config = ctx.get("config")
            database = ctx.get("database")
            return f"LLMService(config={config}, db={database})"

        def get_tool_service():
            llm = ctx.get("llm")
            return f"ToolService(llm={llm})"

        # Test dependency resolution
        llm_service = get_llm_service()
        tool_service = get_tool_service()

        assert "config=" in llm_service
        assert "db=" in llm_service
        assert "llm=" in tool_service

    def test_context_lifecycle(self):
        """Test context lifecycle operations."""
        ctx = AppContext()

        # Initial state
        assert len(ctx._resources) == 0

        # Registration phase
        ctx.register("service1", "Service1")
        ctx.register("service2", "Service2")
        assert len(ctx._resources) == 2

        # Usage phase
        assert ctx.get("service1") == "Service1"
        assert ctx.get("service2") == "Service2"

        # Modification phase
        ctx.register("service1", "UpdatedService1")
        assert ctx.get("service1") == "UpdatedService1"
        assert len(ctx._resources) == 2  # Still same count

        # Final state
        assert "service1" in ctx
        assert "service2" in ctx
        assert "service3" not in ctx
