"""Command prefix pattern matcher with glob support and subcommand tree logic.

Provides hierarchical command matching to differentiate between safe and dangerous
subcommands (e.g., git status vs git reset --hard, python -m pytest vs python -c).
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import tomli


class PolicyAction(str, Enum):
    """Policy action for command patterns."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class CommandPattern:
    """A command pattern with glob support.

    Attributes:
        pattern: Command pattern (e.g., "git diff*", "git status", "pytest*")
        action: Policy action for this pattern
        description: Human-readable description
        priority: Higher priority patterns are checked first
    """

    pattern: str
    action: PolicyAction
    description: str = ""
    priority: int = 100


@dataclass
class CommandPolicy:
    """Command approval policy loaded from configuration.

    Attributes:
        default_action: Default action when no patterns match
        allow_patterns: Patterns that are allowed
        ask_patterns: Patterns that require approval
        deny_patterns: Patterns that are denied
    """

    default_action: PolicyAction = PolicyAction.ASK
    allow_patterns: list[CommandPattern] = field(default_factory=list)
    ask_patterns: list[CommandPattern] = field(default_factory=list)
    deny_patterns: list[CommandPattern] = field(default_factory=list)

    def get_all_patterns(self) -> list[CommandPattern]:
        """Get all patterns sorted by priority (highest first)."""
        all_patterns = self.allow_patterns + self.ask_patterns + self.deny_patterns
        return sorted(all_patterns, key=lambda p: p.priority, reverse=True)

    @classmethod
    def from_toml(cls, path: str | Path) -> CommandPolicy:
        """Load policy from a TOML configuration file.

        Args:
            path: Path to the TOML config file

        Returns:
            CommandPolicy instance
        """
        with open(path, "rb") as f:
            data = tomli.load(f)

        policy_data = data.get("policy", {})

        default_action = PolicyAction(policy_data.get("default_action", "ask"))

        def parse_patterns(
            pattern_list: list[str], action: PolicyAction, base_priority: int
        ) -> list[CommandPattern]:
            patterns = []
            for i, pattern_str in enumerate(pattern_list):
                patterns.append(
                    CommandPattern(
                        pattern=pattern_str,
                        action=action,
                        description=f"{action.value.capitalize()}: {pattern_str}",
                        priority=base_priority - i,
                    )
                )
            return patterns

        allow_patterns = parse_patterns(
            policy_data.get("allow", []), PolicyAction.ALLOW, 200
        )
        ask_patterns = parse_patterns(policy_data.get("ask", []), PolicyAction.ASK, 150)
        deny_patterns = parse_patterns(
            policy_data.get("deny", []), PolicyAction.DENY, 250
        )

        return cls(
            default_action=default_action,
            allow_patterns=allow_patterns,
            ask_patterns=ask_patterns,
            deny_patterns=deny_patterns,
        )


class CommandPatternMatcher:
    """Matches commands against policy patterns using glob patterns.

    Supports:
    - Glob patterns (*, ?) for flexible matching
    - Hierarchical subcommand matching
    - Priority-based pattern evaluation
    - Session and project whitelists for user approvals
    """

    def __init__(self, policy: CommandPolicy | None = None) -> None:
        """Initialize the pattern matcher.

        Args:
            policy: Command policy to use (creates default if None)
        """
        self._policy = policy or CommandPolicy()
        self._session_whitelist: set[str] = set()
        self._project_whitelist: set[str] = set()

    @classmethod
    def from_toml(cls, path: str | Path) -> CommandPatternMatcher:
        """Create a matcher from a TOML configuration file.

        Args:
            path: Path to the TOML config file

        Returns:
            CommandPatternMatcher instance
        """
        policy = CommandPolicy.from_toml(path)
        return cls(policy)

    def match(self, argv: list[str]) -> tuple[PolicyAction, str]:  # noqa: C901
        """Match a command against policy patterns.

        Args:
            argv: Command and arguments as a list

        Returns:
            Tuple of (action, matching_pattern_description)
        """
        if not argv:
            return self._policy.default_action, "Empty command"

        # Convert argv to command string for matching
        command_str = self._argv_to_command_string(argv)

        # Check session whitelist first (highest priority)
        for pattern in self._session_whitelist:
            if self._matches_pattern(command_str, pattern):
                return PolicyAction.ALLOW, f"Session whitelisted: {pattern}"

        # Check project whitelist
        for pattern in self._project_whitelist:
            if self._matches_pattern(command_str, pattern):
                return PolicyAction.ALLOW, f"Project whitelisted: {pattern}"

        # Check deny patterns (highest priority in policy)
        for pattern in sorted(
            self._policy.deny_patterns, key=lambda p: p.priority, reverse=True
        ):
            if self._matches_pattern(command_str, pattern.pattern):
                return PolicyAction.DENY, pattern.description or pattern.pattern

        # Check allow patterns
        for pattern in sorted(
            self._policy.allow_patterns, key=lambda p: p.priority, reverse=True
        ):
            if self._matches_pattern(command_str, pattern.pattern):
                return PolicyAction.ALLOW, pattern.description or pattern.pattern

        # Check ask patterns
        for pattern in sorted(
            self._policy.ask_patterns, key=lambda p: p.priority, reverse=True
        ):
            if self._matches_pattern(command_str, pattern.pattern):
                return PolicyAction.ASK, pattern.description or pattern.pattern

        # No pattern matched, use default
        return self._policy.default_action, "Default policy"

    def add_session_whitelist(self, pattern: str) -> None:
        """Add a pattern to the session whitelist (in-memory).

        Args:
            pattern: Pattern to whitelist (e.g., "git diff*")
        """
        self._session_whitelist.add(pattern)

    def add_project_whitelist(self, pattern: str) -> None:
        """Add a pattern to the project whitelist (persistent).

        Args:
            pattern: Pattern to whitelist (e.g., "git diff*")
        """
        self._project_whitelist.add(pattern)

    def clear_session_whitelist(self) -> None:
        """Clear the session whitelist."""
        self._session_whitelist.clear()

    def _argv_to_command_string(self, argv: list[str]) -> str:
        """Convert argv list to command string for pattern matching.

        Args:
            argv: Command and arguments

        Returns:
            Command string (e.g., "git status", "python -m pytest")
        """
        if not argv:
            return ""
        return " ".join(argv)

    def _matches_pattern(self, command_str: str, pattern: str) -> bool:
        """Check if a command string matches a pattern.

        Args:
            command_str: Command string to match
            pattern: Glob pattern to match against

        Returns:
            True if pattern matches
        """
        # Handle exact match
        if command_str == pattern:
            return True

        # Handle glob pattern matching
        if fnmatch.fnmatch(command_str, pattern):
            return True

        # Handle prefix matching for patterns ending with *
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            if command_str.startswith(prefix):
                return True

        return False


def create_default_policy() -> CommandPolicy:
    """Create a default command policy with safe patterns.

    Returns:
        Default CommandPolicy with common safe/dangerous patterns
    """
    return CommandPolicy(
        default_action=PolicyAction.ASK,
        allow_patterns=[
            CommandPattern(
                pattern="git diff*",
                action=PolicyAction.ALLOW,
                description="Git diff commands are safe reads",
                priority=150,
            ),
            CommandPattern(
                pattern="git status",
                action=PolicyAction.ALLOW,
                description="Git status is a safe read",
                priority=150,
            ),
            CommandPattern(
                pattern="git log*",
                action=PolicyAction.ALLOW,
                description="Git log commands are safe reads",
                priority=150,
            ),
            CommandPattern(
                pattern="git show*",
                action=PolicyAction.ALLOW,
                description="Git show commands are safe reads",
                priority=150,
            ),
            CommandPattern(
                pattern="git branch*",
                action=PolicyAction.ALLOW,
                description="Git branch commands are safe reads",
                priority=150,
            ),
            CommandPattern(
                pattern="cargo check",
                action=PolicyAction.ALLOW,
                description="Cargo check is safe",
                priority=150,
            ),
            CommandPattern(
                pattern="cargo test*",
                action=PolicyAction.ALLOW,
                description="Cargo test commands are safe",
                priority=150,
            ),
            CommandPattern(
                pattern="cargo build*",
                action=PolicyAction.ALLOW,
                description="Cargo build commands are safe",
                priority=150,
            ),
            CommandPattern(
                pattern="pytest*",
                action=PolicyAction.ALLOW,
                description="Pytest commands are safe",
                priority=150,
            ),
            CommandPattern(
                pattern="ruff check*",
                action=PolicyAction.ALLOW,
                description="Ruff check commands are safe",
                priority=150,
            ),
            CommandPattern(
                pattern="ruff format*",
                action=PolicyAction.ALLOW,
                description="Ruff format commands are safe",
                priority=150,
            ),
            CommandPattern(
                pattern="ls",
                action=PolicyAction.ALLOW,
                description="ls is a safe read",
                priority=150,
            ),
            CommandPattern(
                pattern="cat",
                action=PolicyAction.ALLOW,
                description="cat is a safe read",
                priority=150,
            ),
            CommandPattern(
                pattern="head",
                action=PolicyAction.ALLOW,
                description="head is a safe read",
                priority=150,
            ),
            CommandPattern(
                pattern="tail",
                action=PolicyAction.ALLOW,
                description="tail is a safe read",
                priority=150,
            ),
            CommandPattern(
                pattern="grep*",
                action=PolicyAction.ALLOW,
                description="grep commands are safe reads",
                priority=150,
            ),
            CommandPattern(
                pattern="find*",
                action=PolicyAction.ALLOW,
                description="find commands are safe reads",
                priority=150,
            ),
        ],
        ask_patterns=[
            CommandPattern(
                pattern="git commit*",
                action=PolicyAction.ASK,
                description="Git commit requires approval",
                priority=180,
            ),
            CommandPattern(
                pattern="git push*",
                action=PolicyAction.ASK,
                description="Git push requires approval",
                priority=180,
            ),
            CommandPattern(
                pattern="git pull*",
                action=PolicyAction.ASK,
                description="Git pull requires approval",
                priority=180,
            ),
            CommandPattern(
                pattern="git reset*",
                action=PolicyAction.ASK,
                description="Git reset requires approval",
                priority=180,
            ),
            CommandPattern(
                pattern="git checkout*",
                action=PolicyAction.ASK,
                description="Git checkout requires approval",
                priority=180,
            ),
            CommandPattern(
                pattern="git revert*",
                action=PolicyAction.ASK,
                description="Git revert requires approval",
                priority=180,
            ),
            CommandPattern(
                pattern="npm install*",
                action=PolicyAction.ASK,
                description="npm install requires approval",
                priority=180,
            ),
            CommandPattern(
                pattern="pip install*",
                action=PolicyAction.ASK,
                description="pip install requires approval",
                priority=180,
            ),
            CommandPattern(
                pattern="rm *",
                action=PolicyAction.ASK,
                description="rm commands require approval",
                priority=180,
            ),
            CommandPattern(
                pattern="cp *",
                action=PolicyAction.ASK,
                description="cp commands require approval",
                priority=180,
            ),
            CommandPattern(
                pattern="mv *",
                action=PolicyAction.ASK,
                description="mv commands require approval",
                priority=180,
            ),
            CommandPattern(
                pattern="python*",
                action=PolicyAction.ASK,
                description="Python commands require approval (risk of arbitrary code)",
                priority=180,
            ),
            CommandPattern(
                pattern="node*",
                action=PolicyAction.ASK,
                description="Node commands require approval (risk of arbitrary code)",
                priority=180,
            ),
            CommandPattern(
                pattern="bash*",
                action=PolicyAction.ASK,
                description="Bash commands require approval",
                priority=180,
            ),
        ],
        deny_patterns=[
            CommandPattern(
                pattern="sudo *",
                action=PolicyAction.DENY,
                description="sudo commands are denied",
                priority=250,
            ),
            CommandPattern(
                pattern="systemctl*",
                action=PolicyAction.DENY,
                description="systemctl commands are denied",
                priority=250,
            ),
            CommandPattern(
                pattern="*id_rsa*",
                action=PolicyAction.DENY,
                description="Commands accessing id_rsa are denied",
                priority=250,
            ),
            CommandPattern(
                pattern=":(){ :|:& };:",
                action=PolicyAction.DENY,
                description="Fork bomb denied",
                priority=250,
            ),
            CommandPattern(
                pattern="apt*",
                action=PolicyAction.DENY,
                description="Package manager commands are denied",
                priority=250,
            ),
            CommandPattern(
                pattern="apt-get*",
                action=PolicyAction.DENY,
                description="Package manager commands are denied",
                priority=250,
            ),
            CommandPattern(
                pattern="yum*",
                action=PolicyAction.DENY,
                description="Package manager commands are denied",
                priority=250,
            ),
            CommandPattern(
                pattern="dnf*",
                action=PolicyAction.DENY,
                description="Package manager commands are denied",
                priority=250,
            ),
        ],
    )
