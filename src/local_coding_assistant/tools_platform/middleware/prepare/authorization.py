"""Authorization middleware for permission and approval evaluation.

This module implements the authorization middleware that maps tool
effects to permissions and evaluates the reusable ApprovalService.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_coding_assistant.tools_platform.contracts import (
    ApprovalResponse,
    ToolCallRequest,
    ToolDecision,
    ToolDecisionOutcome,
    ValidatedToolCall,
)
from local_coding_assistant.tools_platform.middleware.base import (
    MiddlewareContext,
    PrepareMiddleware,
)

if TYPE_CHECKING:
    from local_coding_assistant.tools_platform.approval import ApprovalService


class AuthorizationMiddleware(PrepareMiddleware):
    """Middleware for permission and approval evaluation.

    This middleware:
    - Maps tool effects to permissions
    - Evaluates the reusable ApprovalService
    - Returns allow/deny/ask decisions
    - Handles confirmation tokens for retry after ASK
    """

    def __init__(self, approval_service: ApprovalService) -> None:
        """Initialize authorization middleware.

        Args:
            approval_service: Approval policy engine
        """
        self.approval_service = approval_service

    async def prepare(self, context: MiddlewareContext) -> None:
        """Check permissions and get approval decision.

        Args:
            context: Middleware context

        Raises:
            ValueError: If validated call not set in context
        """
        validated_call = context.validated_call
        if validated_call is None:
            raise ValueError("Validated call not set in context")

        decision = await self._authorize_call(validated_call, context)
        context.decision = decision

    async def _authorize_call(
        self,
        validated_call: ValidatedToolCall,
        context: MiddlewareContext,
    ) -> ToolDecision:
        """Check permissions and get approval decision.

        Args:
            validated_call: Validated tool call
            context: Middleware context

        Returns:
            Approval decision
        """
        request = validated_call.request

        # Handle confirmation token from previous ASK decision
        if request.confirmation_token:
            return await self._handle_confirmation_token(request)

        # Extract arguments from input model
        arguments = (
            validated_call.input_model.model_dump()
            if hasattr(validated_call.input_model, "model_dump")
            else dict(validated_call.input_model)
        )

        return self.approval_service.evaluate(
            tool_id=request.tool_id,
            arguments=arguments,
            backend=context.backend.get_backend_type(),
            session_id=request.session_id,
            call_id=request.call_id,
            risk_class=validated_call.tool_definition.risk_class,
        )

    async def _handle_confirmation_token(
        self, request: ToolCallRequest
    ) -> ToolDecision:
        """Verify confirmation token and return appropriate decision.

        Args:
            request: Tool call request with confirmation token

        Returns:
            ALLOW if token valid, DENY if invalid/expired
        """
        token = request.confirmation_token
        if not token:
            return ToolDecision(
                outcome=ToolDecisionOutcome.DENY,
                reason="Missing confirmation token",
                policy_rule="confirmation_token_missing",
            )
        approval_response = request.approval_response or ApprovalResponse.YES_ONCE

        confirmed = self.approval_service.confirm(
            token=token,
            session_id=request.session_id,
            approval_response=approval_response,
        )

        if confirmed:
            return ToolDecision(
                outcome=ToolDecisionOutcome.ALLOW,
                reason="User approved via confirmation token",
                policy_rule="confirmation_token",
            )
        else:
            return ToolDecision(
                outcome=ToolDecisionOutcome.DENY,
                reason="Invalid or expired confirmation token",
                policy_rule="confirmation_token_invalid",
            )
