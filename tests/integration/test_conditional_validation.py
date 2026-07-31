"""Integration tests for conditional validation system."""

import os

import pytest

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.config.env_manager import get_env_manager
from local_coding_assistant.core.system_registry import system_capability_registry


@pytest.fixture(autouse=True)
def cleanup_env_vars():
    """Clean up LOCCA environment variables before and after each test."""
    # Store original values
    original_env = {}
    for key in list(os.environ.keys()):
        if key.startswith("LOCCA_"):
            original_env[key] = os.environ[key]

    # Clear all LOCCA environment variables
    for key in list(os.environ.keys()):
        if key.startswith("LOCCA_"):
            del os.environ[key]

    yield

    # Restore original values
    for key, value in original_env.items():
        os.environ[key] = value

    # Clean up any new LOCCA variables that might have been added
    for key in list(os.environ.keys()):
        if key.startswith("LOCCA_") and key not in original_env:
            del os.environ[key]


@pytest.fixture(autouse=True)
def reset_system_registry():
    """Reset the global system capability registry before each test."""
    # Clear all capabilities and modules
    system_capability_registry.capabilities.clear()
    system_capability_registry.module_status.clear()
    system_capability_registry.pending_validations.clear()
    # Don't clear capability_to_settings as it's populated during config loading


class TestSystemDegradation:
    """Test system degradation and backward resolution when capabilities unregister."""

    @pytest.fixture
    def config_manager(self):
        """Create a fresh ConfigManager for each test."""
        env_manager = get_env_manager()
        return ConfigManager(env_manager=env_manager)

    def test_unregister_ptc_capability_falls_back_to_classic(self, config_manager):
        """Test that unregistering sandbox capability falls back from ptc to classic."""
        # Set PTC mode and sandbox enabled
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "ptc"
        os.environ["LOCCA_SANDBOX__ENABLED"] = "true"

        # Load config
        config_manager.load_global_config()

        # Register both modules to enable PTC and sandbox
        config_manager.register_module(
            "tool_manager",
            {
                "python_execution": True,
                "tool_count": 12,
            },
        )
        config_manager.register_module(
            "sandbox_manager",
            {
                "sandbox_available": True,
            },
        )

        # Verify PTC and sandbox are active
        assert config_manager.global_config.runtime.tool_call_mode == "ptc"
        assert config_manager.global_config.sandbox.enabled == True

        # Unregister sandbox capability (should fallback both settings)
        affected = config_manager.unregister_capability(["sandbox_available"])

        # Should affect both the tool_call_mode and sandbox.enabled settings
        assert len(affected) == 2

        affected_settings = dict(affected)
        assert "runtime.tool_call_mode" in affected_settings
        assert "sandbox.enabled" in affected_settings
        assert affected_settings["runtime.tool_call_mode"] == "ptc"
        assert affected_settings["sandbox.enabled"] == True

        # Verify fallback to classic and sandbox disabled
        assert config_manager.global_config.runtime.tool_call_mode == "classic"
        assert config_manager.global_config.sandbox.enabled == False

    def test_unregister_tool_manager_falls_back_to_reasoning_only(self, config_manager):
        """Test that unregistering tool_manager falls back from classic to reasoning_only."""
        # Set classic mode
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "classic"

        # Load config
        config_manager.load_global_config()

        # Register tool_manager to enable classic
        config_manager.register_module(
            "tool_manager",
            {
                "python_execution": True,
                "tool_count": 12,
            },
        )

        # Verify classic is active
        assert config_manager.global_config.runtime.tool_call_mode == "classic"

        # Unregister tool_manager module (should fallback to reasoning_only)
        affected = config_manager.unregister_capability(["tool_manager"])

        # Should affect the tool_call_mode setting (may return multiple possible values)
        assert len(affected) >= 1

        # Check that classic is in the affected settings
        affected_dict = dict(affected)
        assert "runtime.tool_call_mode" in affected_dict
        assert "classic" in [
            v for k, v in affected_dict.items() if k == "runtime.tool_call_mode"
        ]

        # Verify fallback to reasoning_only
        assert config_manager.global_config.runtime.tool_call_mode == "reasoning_only"

    def test_unregister_sandbox_capability_disables_sandbox_setting(
        self, config_manager
    ):
        """Test that unregistering sandbox capability disables sandbox.enabled."""
        # Set sandbox enabled
        os.environ["LOCCA_SANDBOX__ENABLED"] = "true"

        # Load config
        config_manager.load_global_config()

        # Register sandbox_manager to enable sandbox
        config_manager.register_module(
            "sandbox_manager",
            {
                "sandbox_available": True,
            },
        )

        # Verify sandbox is enabled
        assert config_manager.global_config.sandbox.enabled == True

        # Unregister sandbox capability
        affected = config_manager.unregister_capability(["sandbox_available"])

        # Should affect the sandbox.enabled setting (may also affect other settings)
        assert len(affected) >= 1

        # Check that sandbox.enabled is in the affected settings
        affected_dict = dict(affected)
        assert "sandbox.enabled" in affected_dict
        assert True in [v for k, v in affected_dict.items() if k == "sandbox.enabled"]

        # Verify fallback to False
        assert config_manager.global_config.sandbox.enabled == False

    def test_unregister_multiple_capabilities_affects_multiple_settings(
        self, config_manager
    ):
        """Test unregistering multiple capabilities affects multiple dependent settings."""
        # Set both PTC mode and sandbox enabled
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "ptc"
        os.environ["LOCCA_SANDBOX__ENABLED"] = "true"

        # Load config
        config_manager.load_global_config()

        # Register both modules
        config_manager.register_module(
            "tool_manager",
            {
                "python_execution": True,
                "tool_count": 12,
            },
        )
        config_manager.register_module(
            "sandbox_manager",
            {
                "sandbox_available": True,
            },
        )

        # Verify both settings are active
        assert config_manager.global_config.runtime.tool_call_mode == "ptc"
        assert config_manager.global_config.sandbox.enabled == True

        # Unregister both sandbox capabilities
        affected = config_manager.unregister_capability(
            ["sandbox_available", "sandbox_manager"]
        )

        # Should affect both settings
        assert len(affected) == 2
        setting_names = [k for k, v in affected.items()]
        assert "runtime.tool_call_mode" in setting_names
        assert "sandbox.enabled" in setting_names

        # Verify both settings fell back
        assert config_manager.global_config.runtime.tool_call_mode == "classic"
        assert config_manager.global_config.sandbox.enabled == False

    def test_unregister_nonexistent_capability_no_effect(self, config_manager):
        """Test that unregistering non-existent capability has no effect."""
        # Load config
        config_manager.load_global_config()

        # Get initial state
        initial_tool_mode = config_manager.global_config.runtime.tool_call_mode
        initial_sandbox_enabled = config_manager.global_config.sandbox.enabled

        # Unregister non-existent capability
        affected = config_manager.unregister_capability(["nonexistent_capability"])

        # Should have no effect
        assert len(affected) == 0
        assert config_manager.global_config.runtime.tool_call_mode == initial_tool_mode
        assert config_manager.global_config.sandbox.enabled == initial_sandbox_enabled

    def test_unregister_capability_adds_to_pending_validations(self, config_manager):
        """Test that unregistering capabilities adds settings to pending validations."""
        # Set PTC mode
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "ptc"

        # Load config
        config_manager.load_global_config()

        # Register both modules to enable PTC
        config_manager.register_module(
            "tool_manager",
            {
                "python_execution": True,
                "tool_count": 12,
            },
        )
        config_manager.register_module(
            "sandbox_manager",
            {
                "sandbox_available": True,
            },
        )

        # Verify PTC is active and no pending validations
        assert config_manager.global_config.runtime.tool_call_mode == "ptc"
        status = config_manager.get_system_status()
        assert len(status["pending_validations"]) == 0

        # Unregister sandbox capability (should add to pending validations)
        config_manager.unregister_capability(["sandbox_available"])

        # Check that PTC setting was added to pending validations
        status = config_manager.get_system_status()
        pending_validations = status["pending_validations"]

        # Should have pending validation for tool_call_mode=ptc
        ptc_pending = any(
            pv["setting"] == "runtime.tool_call_mode" and pv["value"] == "ptc"
            for pv in pending_validations
        )
        assert ptc_pending, "Expected pending validation for runtime.tool_call_mode=ptc"

    def test_reregister_capability_resolves_pending_validations(self, config_manager):
        """Test that re-registering capabilities resolves pending validations."""
        # Set PTC mode
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "ptc"
        os.environ["LOCCA_SANDBOX__ENABLED"] = "false"

        # Load config
        config_manager.load_global_config()

        # Register both modules to enable PTC
        config_manager.register_module(
            "tool_manager",
            {
                "python_execution": True,
                "tool_count": 12,
            },
        )
        config_manager.register_module(
            "sandbox_manager",
            {
                "sandbox_available": True,
            },
        )

        # Verify PTC is active
        assert config_manager.global_config.runtime.tool_call_mode == "ptc"

        # Unregister sandbox capability (should fallback and add to pending)
        config_manager.unregister_capability(["sandbox_available"])
        assert config_manager.global_config.runtime.tool_call_mode == "classic"

        # Re-register sandbox capability (should resolve pending and restore PTC)
        resolved = config_manager.register_capability(["sandbox_available"])

        # Should resolve the pending validation
        assert len(resolved) == 1
        assert "runtime.tool_call_mode" in resolved
        assert resolved["runtime.tool_call_mode"] == "ptc"

        # Verify PTC is restored
        assert config_manager.global_config.runtime.tool_call_mode == "ptc"

        # Verify no more pending validations
        status = config_manager.get_system_status()
        assert len(status["pending_validations"]) == 0

    def test_partial_capability_unregistration(self, config_manager):
        """Test partial unregistration when only some dependencies are removed."""
        # Set PTC mode
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "ptc"

        # Load config
        config_manager.load_global_config()

        # Register both modules to enable PTC
        config_manager.register_module(
            "tool_manager",
            {
                "python_execution": True,
                "tool_count": 12,
            },
        )
        config_manager.register_module(
            "sandbox_manager",
            {
                "sandbox_available": True,
            },
        )

        # Verify PTC is active
        assert config_manager.global_config.runtime.tool_call_mode == "ptc"

        # Unregister only python_execution capability (keep modules)
        affected = config_manager.unregister_capability(["python_execution"])

        # Should affect tool_call_mode and fallback to classic
        assert len(affected) == 1
        assert "runtime.tool_call_mode" in affected
        assert config_manager.global_config.runtime.tool_call_mode == "classic"

    def test_system_degradation_chain(self, config_manager):
        """Test complete system degradation chain from PTC to reasoning_only."""
        # Set PTC mode
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "ptc"

        # Load config
        config_manager.load_global_config()

        # Register both modules to enable PTC
        config_manager.register_module(
            "tool_manager",
            {
                "python_execution": True,
                "tool_count": 12,
            },
        )
        config_manager.register_module(
            "sandbox_manager",
            {
                "sandbox_available": True,
            },
        )

        # Verify PTC is active
        assert config_manager.global_config.runtime.tool_call_mode == "ptc"

        # First degradation: remove sandbox (PTC -> classic)
        config_manager.unregister_capability(["sandbox_available"])
        assert config_manager.global_config.runtime.tool_call_mode == "classic"

        # Second degradation: remove tool_manager (classic -> reasoning_only)
        config_manager.unregister_capability(["tool_manager"])
        assert config_manager.global_config.runtime.tool_call_mode == "reasoning_only"

        # Verify pending validations for both removed states
        status = config_manager.get_system_status()
        pending_validations = status["pending_validations"]

        # Should have pending for both ptc and classic
        ptc_pending = any(
            pv["setting"] == "runtime.tool_call_mode" and pv["value"] == "ptc"
            for pv in pending_validations
        )
        classic_pending = any(
            pv["setting"] == "runtime.tool_call_mode" and pv["value"] == "classic"
            for pv in pending_validations
        )

        assert ptc_pending, "Expected pending validation for ptc"
        assert not classic_pending, "Not expected pending validation for classic"


class TestConditionalValidation:
    """Test conditional validation system with different setting values."""

    @pytest.fixture
    def config_manager(self):
        """Create a fresh ConfigManager for each test."""
        env_manager = get_env_manager()
        return ConfigManager(env_manager=env_manager)

    def test_ptc_mode_with_all_dependencies(self, config_manager):
        """Test PTC mode when all dependencies are available."""
        # Set PTC mode
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "ptc"

        # Load config (should defer validation)
        config_manager.load_global_config()
        tool_call_mode = config_manager.global_config.runtime.tool_call_mode
        assert tool_call_mode == "reasoning_only"  # fallback

        # Register tool_manager (should resolve ptc -> classic fallback, but keep ptc pending)
        resolved = config_manager.register_module(
            "tool_manager",
            {
                "python_execution": True,
                "tool_count": 12,
            },
        )
        assert len(resolved) == 1  # fallback "classic" resolved
        assert resolved["runtime.tool_call_mode"] == "classic"

        # Register sandbox_manager (should resolve pending ptc -> ptc)
        resolved = config_manager.register_module(
            "sandbox_manager",
            {
                "sandbox_available": True,
            },
        )
        assert len(resolved) == 1  # Should resolve pending ptc
        assert resolved["runtime.tool_call_mode"] == "ptc"

    def test_ptc_mode_missing_sandbox(self, config_manager):
        """Test PTC mode when sandbox is not available."""
        # Set PTC mode
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "ptc"

        config_manager.load_global_config()

        # Register tool_manager but not sandbox_manager
        resolved = config_manager.register_module(
            "tool_manager",
            {
                "python_execution": True,
                "tool_count": 12,
            },
        )
        assert len(resolved) == 1  # Should resolve ptc -> classic immediately
        assert "runtime.tool_call_mode" in resolved
        assert resolved["runtime.tool_call_mode"] == "classic"
        assert config_manager.global_config.runtime.tool_call_mode == "classic"

        # Register sandbox_manager as unavailable
        resolved = config_manager.register_module(
            "sandbox_manager",
            {
                "sandbox_available": False,
            },
        )
        assert len(resolved) == 0  # No new resolutions - pending already resolved

    def test_classic_mode_only_needs_tool_manager(self, config_manager):
        """Test classic mode only needs tool_manager."""
        # Set classic mode
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "classic"

        config_manager.load_global_config()

        # Register tool_manager (should resolve classic)
        resolved = config_manager.register_module(
            "tool_manager",
            {
                "python_execution": True,
                "tool_count": 12,
            },
        )
        assert len(resolved) == 1  # Should resolve tool_call_mode
        assert "runtime.tool_call_mode" in resolved
        assert config_manager.global_config.runtime.tool_call_mode == "classic"

    def test_reasoning_only_no_dependencies(self, config_manager):
        """Test reasoning_only mode needs no dependencies."""
        # Set reasoning_only mode
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "reasoning_only"

        config_manager.load_global_config()
        assert (
            config_manager.global_config.runtime.tool_call_mode == "reasoning_only"
        )  # Should validate immediately

        # Should have no pending validations since reasoning_only needs no dependencies
        status = config_manager.get_system_status()
        assert len(status["pending_validations"]) == 0

    def test_sandbox_enabled_true_when_available(self, config_manager):
        """Test sandbox.enabled=True when sandbox is available."""
        # Set sandbox enabled
        os.environ["LOCCA_SANDBOX__ENABLED"] = "true"

        config_manager.load_global_config()
        assert (
            config_manager.global_config.sandbox.enabled == False
        )  # Should defer validation

        # Register sandbox_manager as available
        resolved = config_manager.register_module(
            "sandbox_manager",
            {
                "sandbox_available": True,
            },
        )
        assert len(resolved) == 1  # Should resolve sandbox.enabled
        assert "sandbox.enabled" in resolved
        assert config_manager.global_config.sandbox.enabled == True

    def test_sandbox_enabled_false_when_unavailable(self, config_manager):
        """Test sandbox.enabled=True when sandbox is not available."""
        # Set sandbox enabled
        os.environ["LOCCA_SANDBOX__ENABLED"] = "true"

        config_manager.load_global_config()
        assert (
            config_manager.global_config.sandbox.enabled == False
        )  # Should defer validation

        # Register sandbox_manager as unavailable
        resolved = config_manager.register_module(
            "sandbox_manager",
            {
                "sandbox_available": False,
            },
        )
        assert len(resolved) == 0  # Should not resolve (validation fails)
        assert config_manager.global_config.sandbox.enabled == False

    def test_fallback_chain_ptc_to_classic_to_reasoning(self, config_manager):
        """Test fallback chain: ptc -> classic -> reasoning_only."""
        # Set PTC mode
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "ptc"

        config_manager.load_global_config()

        # Register only tool_manager (should fallback to classic)
        resolved = config_manager.register_module(
            "tool_manager",
            {
                "python_execution": True,
                "tool_count": 12,
            },
        )
        assert len(resolved) == 1  # Should fallback to classic
        assert "runtime.tool_call_mode" in resolved
        assert config_manager.global_config.runtime.tool_call_mode == "classic"

    def test_system_status_transparency(self, config_manager):
        """Test that system status shows value-specific dependencies."""
        # Set PTC mode
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "ptc"

        config_manager.load_global_config()
        status = config_manager.get_system_status()

        # Find the pending validation for runtime.tool_call_mode with value 'ptc'
        ptc_pending = None
        for pv in status["pending_validations"]:
            if pv["setting"] == "runtime.tool_call_mode" and pv["value"] == "ptc":
                ptc_pending = pv
                break

        assert ptc_pending is not None, (
            "Should have pending validation for runtime.tool_call_mode = 'ptc'"
        )

        # Check the dependencies for the PTC value
        ptc_deps = ptc_pending["dependencies"]

        # Should have the unified dependencies list
        assert isinstance(ptc_deps, list)
        assert "tool_manager" in ptc_deps
        assert "sandbox_manager" in ptc_deps
        assert "python_execution" in ptc_deps
        assert "sandbox_available" in ptc_deps


class TestPendingValidationOverride:
    """Test pending validation override behavior when user explicitly changes settings."""

    @pytest.fixture
    def config_manager(self):
        """Create a fresh ConfigManager for each test."""
        env_manager = get_env_manager()
        return ConfigManager(env_manager=env_manager)

    def test_pending_validation_removed_when_user_sets_valid_value(
        self, config_manager
    ):
        """Test that pending validations are removed when user explicitly sets a valid value.

        This tests the edge case where:
        1. User sets an invalid value that gets added to pending validations
        2. User then explicitly changes to a valid value
        3. When dependencies become available, the pending validation should not override
           the user's explicit valid choice
        """
        # Step 1: Set an invalid value that requires missing dependencies
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "ptc"

        # Load config - this should add 'ptc' to pending validations
        config_manager.load_global_config()

        # Verify ptc is in pending validations (requires tool_manager, sandbox_manager)
        status = config_manager.get_system_status()
        pending_validations = status["pending_validations"]
        ptc_pending = any(
            pv["setting"] == "runtime.tool_call_mode" and pv["value"] == "ptc"
            for pv in pending_validations
        )
        assert ptc_pending, "ptc should be in pending validations"

        # Step 2: User explicitly changes to a valid value (reasoning_only - no dependencies)
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "reasoning_only"

        # This should remove pending validations for runtime.tool_call_mode
        config_manager.load_global_config()

        # Verify pending validations are removed
        status = config_manager.get_system_status()
        pending_validations = status["pending_validations"]
        tool_call_pending = any(
            pv["setting"] == "runtime.tool_call_mode" for pv in pending_validations
        )
        assert not tool_call_pending, (
            "Pending validations should be removed when user sets valid value"
        )

        # Step 3: Simulate dependencies becoming available
        # Register the missing modules
        config_manager.register_module("tool_manager")
        resolved = config_manager.register_module(
            "sandbox_manager", capabilities={"sandbox_available": True}
        )

        # Step 4: Verify the user's explicit choice is preserved
        # The config should still be 'reasoning_only', not overridden to 'ptc'
        assert config_manager.global_config.runtime.tool_call_mode == "reasoning_only"

        # No pending validations should have been resolved
        assert "runtime.tool_call_mode" not in resolved, (
            "No pending validations should have been resolved"
        )

    def test_pending_validation_preserved_when_user_keeps_invalid_value(
        self, config_manager
    ):
        """Test that pending validations are preserved when user keeps invalid value.

        This tests the normal case where:
        1. User sets an invalid value that gets added to pending validations
        2. User keeps the invalid value
        3. When dependencies become available, the pending validation should resolve
           and the value should be applied
        """
        # Step 1: Set an invalid value that requires missing dependencies
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "ptc"

        # Load config - this should add 'ptc' to pending validations
        config_manager.load_global_config()

        # Verify ptc is in pending validations
        status = config_manager.get_system_status()
        pending_validations = status["pending_validations"]
        ptc_pending = any(
            pv["setting"] == "runtime.tool_call_mode" and pv["value"] == "ptc"
            for pv in pending_validations
        )
        assert ptc_pending, "ptc should be in pending validations"

        # Step 2: User keeps the same value (no change)
        # Don't modify environment, just reload to simulate keeping the same value
        config_manager.load_global_config()

        # Step 3: Simulate dependencies becoming available
        # Register the missing modules
        tool_resolved = config_manager.register_module(
            "tool_manager", capabilities={"python_execution": True}
        )
        sandbox_resolved = config_manager.register_module(
            "sandbox_manager", capabilities={"sandbox_available": True}
        )

        # The ptc validation should be resolved in the second call when all dependencies are available
        assert "runtime.tool_call_mode" in tool_resolved, (
            "Should have resolved to classic fallback"
        )
        assert tool_resolved["runtime.tool_call_mode"] == "classic", (
            "Should have resolved to classic fallback"
        )
        assert "runtime.tool_call_mode" in sandbox_resolved, (
            "ptc pending validation should have been resolved"
        )
        assert sandbox_resolved["runtime.tool_call_mode"] == "ptc", (
            "Should have resolved to ptc"
        )

        # Step 4: Verify the pending validation resolved and applied the value
        # The config should now be 'ptc' (resolved from pending)
        assert config_manager.global_config.runtime.tool_call_mode == "ptc"

        # Verify no pending validations remain for runtime.tool_call_mode
        status = config_manager.get_system_status()
        pending_validations = status["pending_validations"]
        tool_call_pending = any(
            pv["setting"] == "runtime.tool_call_mode" for pv in pending_validations
        )
        assert not tool_call_pending, (
            "No pending validations should remain for runtime.tool_call_mode"
        )

    def test_pending_validation_preserved_during_capability_unregister(
        self, config_manager
    ):
        """Test that current config values are preserved during capability unregister.

        This tests that when capabilities are unregistered, the system doesn't
        corrupt the current configuration state.
        """
        # Step 1: Set PTC mode and register all dependencies to make it valid
        os.environ["LOCCA_RUNTIME__TOOL_CALL_MODE"] = "ptc"

        config_manager.load_global_config()

        # Register all required capabilities to make ptc valid
        # Note: The system resolves to fallback when dependencies are missing,
        # but keeps ptc pending for future resolution
        tool_resolved = config_manager.register_module(
            "tool_manager", capabilities={"python_execution": True}
        )
        sandbox_resolved = config_manager.register_module(
            "sandbox_manager", capabilities={"sandbox_available": True}
        )

        # Verify ptc is now active (pending validation resolved when all dependencies became available)
        assert config_manager.global_config.runtime.tool_call_mode == "ptc"

        # Step 2: Unregister a capability
        # This should trigger revalidation and potentially change the config
        config_manager.unregister_capability(["python_execution"])

        # Step 3: Verify the system handled the capability removal appropriately
        # The config might change to a fallback value, but it should be a valid state
        current_value = config_manager.global_config.runtime.tool_call_mode
        assert current_value in ["classic", "reasoning_only"], (
            f"Should be a valid fallback, got {current_value}"
        )
