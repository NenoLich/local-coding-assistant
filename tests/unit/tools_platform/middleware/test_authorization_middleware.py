"""Unit tests for AuthorizationMiddleware.

Covers approval evaluation, confirmation token handling, and approval
response propagation described in plan_docs/2026_09/tools_platform_tests.md.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.tools_platform.approval import ApprovalService, PolicySource
from local_coding_assistant.tools_platform.contracts import (
    ApprovalResponse,
    ToolCallRequest,
    ToolDecisionOutcome,
    ToolRiskClass,
    ValidatedToolCall,
)
from local_coding_assistant.tools_platform.middleware import MiddlewareContext
from local_coding_assistant.tools_platform.middleware.prepare.authorization import (
    AuthorizationMiddleware,
)
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.tools.read_file import ReadFileInput
from local_coding_assistant.tools_platform.tools.register import register_all_tools
from local_coding_assistant.tools_platform.tools.run_command import RunCommandInput
from local_coding_assistant.tools_platform.workspace import WorkspaceManager
from local_coding_assistant.tools_platform.workspace.models import PathPolicy

SESSION_ID = "sess_1"
CALL_ID = "call_1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace_manager(tmp_path: Path) -> WorkspaceManager:
    path_policy = PathPolicy(
        workspace_root=tmp_path,
        allowed_patterns=["**"],
        denied_patterns=[],
        allow_symlinks=False,
        max_depth=None,
    )
    return WorkspaceManager(
        workspace_id="test-ws",
        workspace_root=tmp_path,
        path_policy=path_policy,
        config_manager=None,
    )


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_all_tools(registry)
    return registry


def _make_config_manager(tmp_path: Path) -> ConfigManager:
    """Config manager bound to the real tool policies with isolated persistence."""
    cm = ConfigManager(config_paths=["config/tool_policies.yaml"])
    cfg = cm.load_global_config()
    cfg.tools_platform.workspace_root = str(tmp_path)
    cfg.tools_platform.tool_policies_local_path = str(
        tmp_path / "tool_policies.local.yaml"
    )
    return cm


def _request(
    tool_id: str = "read_file",
    call_id: str = CALL_ID,
    session_id: str = SESSION_ID,
    arguments: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_id=tool_id,
        call_id=call_id,
        session_id=session_id,
        frame_id="frame_1",
        raw_arguments=arguments or {},
        workspace_id="test-ws",
        **kwargs,
    )


def _validated_call(
    registry: ToolRegistry,
    request: ToolCallRequest,
    input_model: Any,
) -> ValidatedToolCall:
    tool_definition = registry.get_tool(request.tool_id)
    assert tool_definition is not None
    return ValidatedToolCall(
        request=request,
        tool_definition=tool_definition,
        input_model=input_model,
    )


def _context(
    registry: ToolRegistry,
    approval_service: ApprovalService,
    workspace_manager: WorkspaceManager,
    request: ToolCallRequest,
    input_model: Any,
) -> MiddlewareContext:
    context = MiddlewareContext(
        request=request,
        registry=registry,
        approval_service=approval_service,
        workspace_manager=workspace_manager,
        backend=Mock(),
        scheduler=Mock(),
    )
    context.backend.get_backend_type.return_value = "host"
    context.validated_call = _validated_call(registry, request, input_model)
    return context


async def _prepare(
    middleware: AuthorizationMiddleware,
    context: MiddlewareContext,
):
    await middleware.prepare(context)
    return context.decision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    return _make_registry()


@pytest.fixture
def workspace_manager(tmp_path: Path) -> WorkspaceManager:
    return _make_workspace_manager(tmp_path)


@pytest.fixture
def config_manager(tmp_path: Path) -> ConfigManager:
    return _make_config_manager(tmp_path)


@pytest.fixture
def approval_service(
    workspace_manager: WorkspaceManager,
    config_manager: ConfigManager,
) -> ApprovalService:
    return ApprovalService(
        workspace_manager=workspace_manager,
        config_manager=config_manager,
    )


@pytest.fixture
def middleware(approval_service: ApprovalService) -> AuthorizationMiddleware:
    return AuthorizationMiddleware(approval_service)


@pytest.fixture
def outside_file(tmp_path: Path) -> Path:
    """A file outside the workspace root (triggers boundary rules)."""
    outside = tmp_path.parent / f"outside_{tmp_path.name}.txt"
    outside.write_text("secret")
    return outside


def _ask_token(
    approval_service: ApprovalService,
    outside_file: Path,
    session_id: str = SESSION_ID,
) -> str:
    """Obtain a real confirmation token by asking for an out-of-workspace read."""
    decision = approval_service.evaluate(
        tool_id="read_file",
        arguments={"path": str(outside_file)},
        backend="host",
        session_id=session_id,
        call_id=CALL_ID,
        risk_class=ToolRiskClass.SAFE,
    )
    assert decision.outcome == ToolDecisionOutcome.ASK
    assert decision.confirmation_token is not None
    return decision.confirmation_token


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authorize_call_sets_decision_from_approval_service(
    middleware: AuthorizationMiddleware,
    registry: ToolRegistry,
    approval_service: ApprovalService,
    workspace_manager: WorkspaceManager,
):
    """prepare() stores the ApprovalService decision on the context."""
    request = _request(arguments={"path": "notes.txt"})
    context = _context(
        registry,
        approval_service,
        workspace_manager,
        request,
        ReadFileInput(path="notes.txt"),
    )

    decision = await _prepare(middleware, context)

    assert decision is not None
    assert decision.outcome == ToolDecisionOutcome.ALLOW
    assert decision.policy_rule == "base_read_file"
    assert decision.reason == "Base policy for read_file"
    assert context.decision is decision


@pytest.mark.asyncio
async def test_authorize_call_denies_when_rule_denies(
    middleware: AuthorizationMiddleware,
    registry: ToolRegistry,
    approval_service: ApprovalService,
    workspace_manager: WorkspaceManager,
):
    """A deny rule produces a DENY decision with the matching rule id."""
    request = _request(
        tool_id="run_command",
        arguments={"argv": ["sudo", "reboot"]},
    )
    context = _context(
        registry,
        approval_service,
        workspace_manager,
        request,
        RunCommandInput(argv=["sudo", "reboot"]),
    )

    decision = await _prepare(middleware, context)

    assert decision is not None
    assert decision.outcome == ToolDecisionOutcome.DENY
    assert decision.policy_rule == "cmd_deny_sudo"


@pytest.mark.asyncio
async def test_authorize_call_asks_with_confirmation_token(
    middleware: AuthorizationMiddleware,
    registry: ToolRegistry,
    approval_service: ApprovalService,
    workspace_manager: WorkspaceManager,
    outside_file: Path,
):
    """An ASK decision carries a confirmation token for the retry."""
    request = _request(arguments={"path": str(outside_file)})
    context = _context(
        registry,
        approval_service,
        workspace_manager,
        request,
        ReadFileInput(path=str(outside_file)),
    )

    decision = await _prepare(middleware, context)

    assert decision is not None
    assert decision.outcome == ToolDecisionOutcome.ASK
    assert decision.policy_rule == "boundary_read_outside"
    assert decision.confirmation_token is not None


@pytest.mark.asyncio
async def test_authorize_call_uses_validated_input_model_arguments(
    middleware: AuthorizationMiddleware,
    registry: ToolRegistry,
    approval_service: ApprovalService,
    workspace_manager: WorkspaceManager,
):
    """Evaluation uses the validated input model, not the raw arguments."""
    request = _request(arguments={"path": "notes.txt"})
    context = _context(
        registry,
        approval_service,
        workspace_manager,
        request,
        ReadFileInput(path="notes.txt"),
    )
    evaluate = Mock(wraps=approval_service.evaluate)
    approval_service.evaluate = evaluate  # type: ignore[method-assign]

    await _prepare(middleware, context)

    assert evaluate.call_count == 1
    kwargs = evaluate.call_args.kwargs
    assert kwargs["tool_id"] == "read_file"
    assert kwargs["arguments"]["path"] == "notes.txt"
    assert kwargs["backend"] == "host"
    assert kwargs["session_id"] == SESSION_ID
    assert kwargs["call_id"] == CALL_ID
    assert kwargs["risk_class"] == ToolRiskClass.SAFE


@pytest.mark.asyncio
async def test_authorize_call_raises_without_validated_call(
    middleware: AuthorizationMiddleware,
    registry: ToolRegistry,
    approval_service: ApprovalService,
    workspace_manager: WorkspaceManager,
):
    """prepare() refuses to run without a validated call."""
    request = _request(arguments={"path": "notes.txt"})
    context = _context(
        registry,
        approval_service,
        workspace_manager,
        request,
        ReadFileInput(path="notes.txt"),
    )
    context.validated_call = None

    with pytest.raises(ValueError, match="Validated call not set in context"):
        await _prepare(middleware, context)


# ---------------------------------------------------------------------------
# Confirmation tokens
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authorize_call_with_valid_confirmation_token_allows(
    middleware: AuthorizationMiddleware,
    registry: ToolRegistry,
    approval_service: ApprovalService,
    workspace_manager: WorkspaceManager,
    outside_file: Path,
):
    """A valid token for the same session allows the call."""
    token = _ask_token(approval_service, outside_file)

    request = _request(
        call_id="call_2",
        arguments={"path": str(outside_file)},
        confirmation_token=token,
    )
    context = _context(
        registry,
        approval_service,
        workspace_manager,
        request,
        ReadFileInput(path=str(outside_file)),
    )

    decision = await _prepare(middleware, context)

    assert decision is not None
    assert decision.outcome == ToolDecisionOutcome.ALLOW
    assert decision.policy_rule == "confirmation_token"


@pytest.mark.asyncio
async def test_authorize_call_with_expired_confirmation_token_denies(
    middleware: AuthorizationMiddleware,
    registry: ToolRegistry,
    approval_service: ApprovalService,
    workspace_manager: WorkspaceManager,
    outside_file: Path,
):
    """An expired token is rejected."""
    token = _ask_token(approval_service, outside_file)
    approval_service._tokens[token].expires_at = datetime.now(UTC) - timedelta(
        minutes=1
    )

    request = _request(
        call_id="call_2",
        arguments={"path": str(outside_file)},
        confirmation_token=token,
    )
    context = _context(
        registry,
        approval_service,
        workspace_manager,
        request,
        ReadFileInput(path=str(outside_file)),
    )

    decision = await _prepare(middleware, context)

    assert decision is not None
    assert decision.outcome == ToolDecisionOutcome.DENY
    assert decision.policy_rule == "confirmation_token_invalid"


@pytest.mark.asyncio
async def test_authorize_call_with_confirmation_token_for_other_session_denies(
    middleware: AuthorizationMiddleware,
    registry: ToolRegistry,
    approval_service: ApprovalService,
    workspace_manager: WorkspaceManager,
    outside_file: Path,
):
    """A token issued for another session is rejected."""
    token = _ask_token(approval_service, outside_file, session_id="other_session")

    request = _request(
        call_id="call_2",
        session_id=SESSION_ID,
        arguments={"path": str(outside_file)},
        confirmation_token=token,
    )
    context = _context(
        registry,
        approval_service,
        workspace_manager,
        request,
        ReadFileInput(path=str(outside_file)),
    )

    decision = await _prepare(middleware, context)

    assert decision is not None
    assert decision.outcome == ToolDecisionOutcome.DENY
    assert decision.policy_rule == "confirmation_token_invalid"


@pytest.mark.asyncio
async def test_authorize_call_with_already_used_confirmation_token_denies(
    middleware: AuthorizationMiddleware,
    registry: ToolRegistry,
    approval_service: ApprovalService,
    workspace_manager: WorkspaceManager,
    outside_file: Path,
):
    """A token can only be used once."""
    token = _ask_token(approval_service, outside_file)

    first_request = _request(
        call_id="call_2",
        arguments={"path": str(outside_file)},
        confirmation_token=token,
    )
    first_context = _context(
        registry,
        approval_service,
        workspace_manager,
        first_request,
        ReadFileInput(path=str(outside_file)),
    )
    first_decision = await _prepare(middleware, first_context)
    assert first_decision is not None
    assert first_decision.outcome == ToolDecisionOutcome.ALLOW

    second_request = _request(
        call_id="call_3",
        arguments={"path": str(outside_file)},
        confirmation_token=token,
    )
    second_context = _context(
        registry,
        approval_service,
        workspace_manager,
        second_request,
        ReadFileInput(path=str(outside_file)),
    )

    decision = await _prepare(middleware, second_context)

    assert decision is not None
    assert decision.outcome == ToolDecisionOutcome.DENY
    assert decision.policy_rule == "confirmation_token_invalid"


# ---------------------------------------------------------------------------
# Approval responses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authorize_call_with_yes_always_session_allow_overrides_rule_in_memory(
    middleware: AuthorizationMiddleware,
    registry: ToolRegistry,
    approval_service: ApprovalService,
    workspace_manager: WorkspaceManager,
    outside_file: Path,
):
    """YES_ALWAYS_SESSION allows the call and relaxes the matched rule in memory."""
    token = _ask_token(approval_service, outside_file)

    request = _request(
        call_id="call_2",
        arguments={"path": str(outside_file)},
        confirmation_token=token,
        approval_response=ApprovalResponse.YES_ALWAYS_SESSION,
    )
    context = _context(
        registry,
        approval_service,
        workspace_manager,
        request,
        ReadFileInput(path=str(outside_file)),
    )

    decision = await _prepare(middleware, context)

    assert decision is not None
    assert decision.outcome == ToolDecisionOutcome.ALLOW
    assert decision.policy_rule == "confirmation_token"

    rule = next(r for r in approval_service._rules if r.id == "boundary_read_outside")
    assert rule.outcome == ToolDecisionOutcome.ALLOW
    assert rule.source == PolicySource.USER_APPROVAL

    # The relaxed rule applies to later evaluations too.
    later = approval_service.evaluate(
        tool_id="read_file",
        arguments={"path": str(outside_file)},
        backend="host",
        session_id=SESSION_ID,
        call_id="call_3",
    )
    assert later.outcome == ToolDecisionOutcome.ALLOW


@pytest.mark.asyncio
async def test_authorize_call_with_yes_always_project_persists_override(
    middleware: AuthorizationMiddleware,
    registry: ToolRegistry,
    approval_service: ApprovalService,
    workspace_manager: WorkspaceManager,
    config_manager: ConfigManager,
    tmp_path: Path,
    outside_file: Path,
):
    """YES_ALWAYS_PROJECT allows the call and persists the override to local.yaml."""
    token = _ask_token(approval_service, outside_file)

    request = _request(
        call_id="call_2",
        arguments={"path": str(outside_file)},
        confirmation_token=token,
        approval_response=ApprovalResponse.YES_ALWAYS_PROJECT,
    )
    context = _context(
        registry,
        approval_service,
        workspace_manager,
        request,
        ReadFileInput(path=str(outside_file)),
    )

    decision = await _prepare(middleware, context)

    assert decision is not None
    assert decision.outcome == ToolDecisionOutcome.ALLOW
    assert decision.policy_rule == "confirmation_token"

    rule = next(r for r in approval_service._rules if r.id == "boundary_read_outside")
    assert rule.outcome == ToolDecisionOutcome.ALLOW
    assert rule.source == PolicySource.USER_APPROVAL

    local_file = tmp_path / "tool_policies.local.yaml"
    assert local_file.exists()
    with open(local_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    overrides = data.get("tools_platform", {}).get("project_overrides", [])
    assert len(overrides) == 1
    assert overrides[0]["base_rule_id"] == "boundary_read_outside"
    assert overrides[0]["outcome"] == "allow"
