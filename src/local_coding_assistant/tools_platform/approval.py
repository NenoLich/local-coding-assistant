"""Approval service for tool execution policy.

All approval logic is expressed as rules with explicit priorities.
Base rules from tool_policies.yaml + project overrides from tool_policies.local.yaml.
Session overrides modify rules in-place at runtime.
"""

from __future__ import annotations

import fnmatch
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from local_coding_assistant.config.schemas import MatcherConfig
from local_coding_assistant.tools_platform.contracts import (
    ApprovalResponse,
    ToolDecision,
    ToolDecisionOutcome,
    ToolRiskClass,
)

if TYPE_CHECKING:
    from local_coding_assistant.core.protocols import IConfigManager
    from local_coding_assistant.tools_platform.workspace import WorkspaceManager


class PolicySource(Enum):
    """Source of a policy rule."""

    CONFIG = "config"
    USER_APPROVAL = "user_approval"


class Matcher(ABC):
    """Base class for argument matchers."""

    @abstractmethod
    def matches(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        workspace_manager: WorkspaceManager | None = None,
    ) -> bool:
        """Check if arguments match this matcher."""
        pass


@dataclass
class AnyArgs(Matcher):
    """Matches any arguments."""

    def matches(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        workspace_manager: WorkspaceManager | None = None,
    ) -> bool:
        return True


@dataclass
class CommandPattern(Matcher):
    """Matches run_command argv as glob pattern."""

    pattern: str

    def matches(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        workspace_manager: WorkspaceManager | None = None,
    ) -> bool:
        if tool_id != "run_command":
            return False
        argv = arguments.get("argv", [])
        if not argv:
            return False
        cmd_str = " ".join(argv)
        if cmd_str == self.pattern:
            return True
        if fnmatch.fnmatch(cmd_str, self.pattern):
            return True
        if self.pattern.endswith("*"):
            prefix = self.pattern[:-1]
            if cmd_str.startswith(prefix):
                return True
        return False


@dataclass
class PathOutsideWorkspace(Matcher):
    """Matches paths outside workspace root."""

    def matches(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        workspace_manager: WorkspaceManager | None = None,
    ) -> bool:
        # Check common path argument names
        path = arguments.get("path") or arguments.get("pattern") or arguments.get("cwd")
        if not path or not workspace_manager:
            return False
        return not workspace_manager.is_path_inside_workspace(path)


@dataclass
class ArgEquals(Matcher):
    """Matches specific argument value."""

    arg_name: str
    expected_value: Any

    def matches(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        workspace_manager: WorkspaceManager | None = None,
    ) -> bool:
        return arguments.get(self.arg_name) == self.expected_value


@dataclass
class ArgPattern(Matcher):
    """Matches argument with glob pattern."""

    arg_name: str
    pattern: str

    def matches(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        workspace_manager: WorkspaceManager | None = None,
    ) -> bool:
        val = arguments.get(self.arg_name)
        if val is None:
            return False
        val_str = str(val)
        if val_str == self.pattern:
            return True
        if fnmatch.fnmatch(val_str, self.pattern):
            return True
        if self.pattern.endswith("*"):
            prefix = self.pattern[:-1]
            if val_str.startswith(prefix):
                return True
        return False


@dataclass
class PolicyRule:
    """A single approval policy rule."""

    id: str
    tool_id: str | list[str]
    matcher: Matcher
    outcome: ToolDecisionOutcome
    description: str = ""
    source: PolicySource = PolicySource.CONFIG
    priority: int = 100

    def matches_tool(self, tool_id: str) -> bool:
        """Check if this rule applies to the given tool."""
        if isinstance(self.tool_id, list):
            return tool_id in self.tool_id
        if self.tool_id == "*":
            return True
        return self.tool_id == tool_id

    def matches(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        workspace_manager: WorkspaceManager | None = None,
    ) -> bool:
        """Check if rule matches the tool and arguments."""
        if not self.matches_tool(tool_id):
            return False
        return self.matcher.matches(tool_id, arguments, workspace_manager)


@dataclass
class ConfirmationToken:
    """User confirmation token for approval."""

    token: str
    call_id: str
    session_id: str
    rule_id: str
    created_at: datetime
    expires_at: datetime
    used: bool = False
    approval_response: ApprovalResponse | None = None
    # The rule that matched and triggered ASK
    matched_rule: PolicyRule | None = None
    # The arguments that triggered the match
    matched_arguments: dict[str, Any] | None = None


class ApprovalService:
    """Reusable allow/deny/ask policy engine.

    Rules are loaded from tool_policies.yaml, merged with project overrides
    from tool_policies.local.yaml, and evaluated in priority order.
    Session overrides modify rules in-place at runtime.
    """

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        config_manager: IConfigManager,
    ) -> None:
        """Initialize the approval service."""
        self._workspace_manager = workspace_manager
        self._config_manager = config_manager
        self._rules: list[PolicyRule] = []
        self._tokens: dict[str, ConfirmationToken] = {}

        # Load rules from config (already merged by ConfigManager)
        self._load_rules()

    @property
    def _config(self) -> Any:
        """Dynamic config property that reads current config including session overrides."""
        return self._config_manager.global_config.tools_platform

    def _load_rules(self) -> None:
        """Load rules from config (already merged with project overrides)."""
        self._rules.clear()

        rules_config = self._config.rules

        for rule_data in rules_config:
            matcher = self._build_matcher(rule_data.matcher)
            self._rules.append(
                PolicyRule(
                    id=rule_data.id,
                    tool_id=rule_data.tool_id,
                    matcher=matcher,
                    outcome=ToolDecisionOutcome(rule_data.outcome),
                    description=rule_data.description,
                    source=PolicySource.CONFIG
                    if rule_data.source == "config"
                    else PolicySource.USER_APPROVAL,
                    priority=rule_data.priority,
                )
            )

        # Sort by priority (highest first)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def _build_matcher(self, matcher_data: MatcherConfig) -> Matcher:
        """Build matcher from config data."""
        m_type = matcher_data.type

        if m_type == "any_args":
            return AnyArgs()
        elif m_type == "command_pattern":
            return CommandPattern(matcher_data.pattern or "")
        elif m_type == "path_outside_workspace":
            return PathOutsideWorkspace()
        elif m_type == "arg_equals":
            return ArgEquals(matcher_data.arg or "", matcher_data.value)
        elif m_type == "arg_pattern":
            return ArgPattern(matcher_data.arg or "", matcher_data.pattern or "")
        else:
            return AnyArgs()

    def evaluate(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        backend: str,
        session_id: str,
        call_id: str,
        risk_class: ToolRiskClass = ToolRiskClass.LOW,
    ) -> ToolDecision:
        """Evaluate a tool call against approval rules in priority order."""
        # Auto-approve mode: skip all rules and allow immediately
        if self._config.auto_approve:
            return ToolDecision(
                outcome=ToolDecisionOutcome.ALLOW,
                reason="Auto-approve mode enabled",
                policy_rule="auto_approve",
            )

        # Check all rules in priority order (already sorted)
        for rule in self._rules:
            if rule.matches(tool_id, arguments, self._workspace_manager):
                if rule.outcome == ToolDecisionOutcome.ASK:
                    return self._create_ask_decision(
                        rule, arguments, call_id, session_id
                    )
                return ToolDecision(
                    outcome=rule.outcome,
                    reason=rule.description or f"Rule matched: {rule.id}",
                    policy_rule=rule.id,
                )

        # Should never reach here due to fallback rule, but safety net
        return ToolDecision(
            outcome=ToolDecisionOutcome.DENY,
            reason="No rule matched (unexpected)",
            policy_rule="no_match",
        )

    def _create_ask_decision(
        self,
        rule: PolicyRule,
        arguments: dict[str, Any],
        call_id: str,
        session_id: str,
    ) -> ToolDecision:
        """Create ASK decision with confirmation token."""
        non_interactive = self._config.non_interactive
        if non_interactive:
            return ToolDecision(
                outcome=ToolDecisionOutcome.DENY,
                reason=f"Non-interactive mode: {rule.description}",
                policy_rule=rule.id,
            )

        token = self._create_confirmation_token(call_id, session_id, rule.id)
        # Store matched rule and arguments for confirmation handling
        confirmation = self._tokens[token]
        confirmation.matched_rule = rule
        confirmation.matched_arguments = arguments

        # Create proposed override rule for preview
        proposed_rule = self._create_override_rule(rule, arguments)

        return ToolDecision(
            outcome=ToolDecisionOutcome.ASK,
            reason=rule.description,
            policy_rule=rule.id,
            confirmation_token=token,
            expiry=datetime.now(UTC) + timedelta(minutes=5),
            proposed_rule=proposed_rule,
        )

    def _create_confirmation_token(
        self,
        call_id: str,
        session_id: str,
        rule_id: str,
    ) -> str:
        """Create a confirmation token."""
        token = secrets.token_urlsafe(32)
        confirmation = ConfirmationToken(
            token=token,
            call_id=call_id,
            session_id=session_id,
            rule_id=rule_id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        self._tokens[token] = confirmation
        return token

    def confirm(
        self,
        token: str,
        session_id: str,
        approval_response: ApprovalResponse = ApprovalResponse.YES_ONCE,
    ) -> bool:
        """Confirm a tool call using a confirmation token."""
        if token not in self._tokens:
            return False

        confirmation = self._tokens[token]

        if confirmation.session_id != session_id:
            return False

        if datetime.now(UTC) > confirmation.expires_at:
            return False

        if confirmation.used:
            return False

        confirmation.approval_response = approval_response
        confirmation.used = True

        matched_rule = confirmation.matched_rule

        if not matched_rule:
            return False

        if approval_response == ApprovalResponse.YES_ONCE:
            # Just allow this call, no persistent change
            return True

        # Override the matched rule in-place
        matched_rule.outcome = ToolDecisionOutcome.ALLOW
        matched_rule.source = PolicySource.USER_APPROVAL

        if approval_response == ApprovalResponse.YES_ALWAYS_SESSION:
            # In-memory only, no persistence
            pass

        elif approval_response == ApprovalResponse.YES_ALWAYS_PROJECT:
            # Persist to tool_policies.local.yaml via ConfigManager
            self._config_manager.persist_tool_policies_overrides(
                rule_id=matched_rule.id,
                outcome=matched_rule.outcome.value,
                description=matched_rule.description,
            )

        return True

    def _create_override_rule(
        self, matched_rule: PolicyRule, arguments: dict[str, Any]
    ) -> PolicyRule:
        """Create proposed override rule from matched rule and actual arguments.

        Generalizes the matcher based on the arguments that triggered the match.
        """
        generalized_matcher = self._generalize_matcher(matched_rule.matcher, arguments)

        return PolicyRule(
            id=matched_rule.id,
            tool_id=matched_rule.tool_id,
            matcher=generalized_matcher,
            outcome=ToolDecisionOutcome.ALLOW,
            description=f"Override: {matched_rule.description}",
            source=PolicySource.USER_APPROVAL,
            priority=matched_rule.priority,
        )

    def _generalize_matcher(
        self, matcher: Matcher, arguments: dict[str, Any]
    ) -> Matcher:
        """Generalize matcher based on actual arguments."""
        if isinstance(matcher, CommandPattern):
            return matcher

        if isinstance(matcher, PathOutsideWorkspace):
            path = (
                arguments.get("path")
                or arguments.get("pattern")
                or arguments.get("cwd", "")
            )
            if path:
                from pathlib import Path

                p = Path(path)
                if p.name and p.name != ".":
                    return (
                        PathOutsideWorkspace()
                    )  # Keep the same matcher for simplicity
            return matcher

        if isinstance(matcher, ArgEquals):
            val = arguments.get(matcher.arg_name)
            if val is not None:
                return ArgEquals(matcher.arg_name, val)
            return matcher

        if isinstance(matcher, ArgPattern):
            return matcher

        return AnyArgs()
