"""Typed action / observation / state models for SupportSphere.

All models inherit from openenv base Pydantic classes to ensure
compatibility with the OpenEnv server + client serialization pipeline.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
class SupportSphereAction(Action):
    """An action the support agent can take on a ticket.

    Attributes:
        action_type: Discriminator selecting the operation.
        payload: Free-form dict carrying action-specific data
                 (e.g. ``{"message": "...", "amount": 499}``).
    """

    action_type: Literal[
        "view_student",
        "reply",
        "issue_refund",
        "escalate",
        "close_ticket",
        "suggest_resource",
        "ask_clarification",
    ] = Field(description="The type of support action to execute")
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific key/value data (message, amount, etc.)",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
class SupportSphereObservation(Observation):
    """What the agent sees after each step.

    Inherits ``done``, ``reward``, and ``metadata`` from the openenv
    ``Observation`` base class automatically.
    """

    current_ticket_id: str = Field(description="Unique ticket identifier")
    ticket_summary: str = Field(description="Human-readable description of the issue")
    student_profile_snippet: str = Field(
        description="Relevant student profile information"
    )
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full conversation turns so far [{role, content}, ...]",
    )
    available_actions_hint: str = Field(
        default="view_student, reply, issue_refund, escalate, close_ticket, suggest_resource, ask_clarification",
        description="Hint string listing valid action_types",
    )
    knowledge_base_snippet: Optional[str] = Field(
        default=None,
        description="Relevant policy / KB excerpt for the current ticket context",
    )
    system_time: int = Field(
        default=0, description="Current step count within the episode"
    )


# ---------------------------------------------------------------------------
# State (server-side only, never sent to agent)
# ---------------------------------------------------------------------------
class SupportSphereState(State):
    """Internal server state for a single episode.

    ``episode_id`` and ``step_count`` are inherited from ``State``.
    """

    task_name: str = Field(default="easy", description="easy | medium | hard")
    current_ticket_idx: int = Field(
        default=0, description="Index into the ticket queue"
    )
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full agent conversation log for grading",
    )
    identity_verified: bool = Field(
        default=False,
        description="Whether the agent has executed an identity verification step",
    )
    refund_attempted: bool = Field(
        default=False,
        description="Whether a refund action was attempted",
    )
    escalated: bool = Field(
        default=False,
        description="Whether the ticket was escalated",
    )
