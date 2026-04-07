"""SupportSphere Environment — the full simulation brain.

Implements a realistic EdTech customer support simulator with:
    • In-memory student database (payment status, progress, auth state)
    • Static knowledge base (refund policy, enrollment rules, common fixes)
    • Multi-turn conversation history preserved across steps
    • Dense partial rewards with clear progress signals
    • Policy engine that penalizes incorrect refund / escalation decisions
"""

import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment
from openenv.core.client_types import StepResult

from ..models import (
    SupportSphereAction,
    SupportSphereObservation,
    SupportSphereState,
)


# ---------------------------------------------------------------------------
# Static data — student DB + knowledge base
# ---------------------------------------------------------------------------
STUDENT_DATABASE: Dict[str, Dict[str, Any]] = {
    "Alice": {
        "id": "STU-1001",
        "email": "alice@example.com",
        "paid": True,
        "payment_date": "2026-03-01",
        "amount_paid": 24999,
        "course": "System Design Masterclass",
        "progress_pct": 80,
        "auth_status": "verified",
        "refund_eligible": True,
        "notes": "Long-time student. 4.8★ NPS.",
    },
    "Bob": {
        "id": "STU-1002",
        "email": "bob@example.com",
        "paid": False,
        "course": None,
        "progress_pct": 0,
        "auth_status": "verified",
        "refund_eligible": False,
        "notes": "Prospective student, browsing courses.",
    },
    "Charlie": {
        "id": "STU-1003",
        "email": "charlie@example.com",
        "paid": True,
        "payment_date": "2026-02-15",
        "amount_paid": 14999,
        "course": "Full-Stack Web Dev",
        "progress_pct": 35,
        "auth_status": "verified",
        "refund_eligible": True,
        "notes": "Reported video playback issues. Paid > 48h ago → refund eligible.",
    },
    "Dana": {
        "id": "STU-1004",
        "email": "dana@example.com",
        "paid": True,
        "payment_date": "2026-03-20",
        "amount_paid": 9999,
        "course": "Data Science Bootcamp",
        "progress_pct": 100,
        "auth_status": "verified",
        "refund_eligible": False,
        "notes": "Completed course. Certificate generation pending (batch job runs nightly).",
    },
    "Eve": {
        "id": "STU-1005",
        "email": "eve@example.com",
        "paid": False,
        "course": "Backend Engineering",
        "progress_pct": 10,
        "auth_status": "pending",
        "refund_eligible": False,
        "notes": "Payment FAILED on 2026-04-01. No successful transaction. Account should be frozen.",
    },
}

KNOWLEDGE_BASE: str = """
== Scaler Support Policy ==
1. Refund: Only if student has PAID and payment was > 48 hours ago.
2. Non-paying students: NEVER issue a refund. Freeze account and escalate to billing.
3. Certificate delay: Certificates are generated nightly. Inform student, no escalation needed.
4. Course access issues: Verify payment first, then reset access token.
5. Enrollment queries: Provide course link and pricing; no action required.
6. Identity: Always verify student identity (view_student) before refund or escalation.
7. Angry students: De-escalate first, then follow policy. Never promise outside policy.
""".strip()

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
TASK_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description": "Resolve two straightforward tickets: a course access issue and an enrollment query.",
        "tickets": [
            {
                "id": "T-1001",
                "summary": "Paid student cannot access course content after payment confirmation.",
                "student": "Alice",
                "expected_resolution": "verify → reply (reset access) → close",
            },
            {
                "id": "T-1002",
                "summary": "Prospective student wants to know how to enroll and pricing details.",
                "student": "Bob",
                "expected_resolution": "reply (share link + pricing) → close",
            },
        ],
    },
    "medium": {
        "description": "Handle a refund request (eligible) and a certificate delay complaint.",
        "tickets": [
            {
                "id": "T-1003",
                "summary": "Paid student reports video not loading and requests a full refund. Student is refund-eligible.",
                "student": "Charlie",
                "expected_resolution": "view_student → verify eligibility → issue_refund → close",
            },
            {
                "id": "T-1004",
                "summary": "Student completed course but hasn't received certificate after 3 days.",
                "student": "Dana",
                "expected_resolution": "reply (explain nightly batch) → close",
            },
        ],
    },
    "hard": {
        "description": "Angry non-paying student demands a full refund. Payment failed — refund is NOT authorized. Correct path is to freeze account and escalate.",
        "tickets": [
            {
                "id": "T-1005",
                "summary": "URGENT: Furious student claims they paid but payment actually FAILED. Demands immediate full refund. Threatens legal action. Student is NOT refund-eligible.",
                "student": "Eve",
                "expected_resolution": "view_student → verify identity (pending) → refuse refund → freeze account / escalate → close",
            },
        ],
    },
}


class SupportSphereEnvironment(Environment):
    """Full simulation environment with student DB, policy engine, and dense rewards."""

    def __init__(self) -> None:
        super().__init__()
        self._state: Optional[SupportSphereState] = None
        self._tickets: List[Dict[str, Any]] = []
        self._current_task: str = "easy"

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------
    def reset(self, task: str = "easy", **kwargs: Any) -> SupportSphereObservation:
        self._current_task = task
        task_def = TASK_DEFINITIONS[task]
        self._tickets = [t.copy() for t in task_def["tickets"]]
        self._state = SupportSphereState(
            episode_id=str(uuid.uuid4()),
            task_name=task,
        )
        return self._build_observation(done=False, reward=None)

    def step(self, action: SupportSphereAction) -> SupportSphereObservation:
        """Execute one agent action and return the next observation.

        The Observation object carries ``done`` and ``reward`` inline
        (inherited from the openenv ``Observation`` base class).
        """
        assert self._state is not None, "Call reset() before step()"
        self._state.step_count += 1

        reward: float = 0.0
        done: bool = False

        # --- Reward logic ------------------------------------------------
        atype = action.action_type

        if atype == "view_student":
            reward += 0.15
            self._state.identity_verified = True

        elif atype == "ask_clarification":
            reward += 0.10
            self._state.identity_verified = True

        elif atype == "suggest_resource":
            reward += 0.10

        elif atype == "reply":
            reward += 0.20
            # Bonus for mentioning freeze in hard task
            if self._current_task == "hard" and "freeze" in str(action.payload).lower():
                reward += 0.15

        elif atype == "issue_refund":
            self._state.refund_attempted = True
            if self._is_refund_authorized():
                reward += 0.35
            else:
                reward -= 0.25  # Policy violation

        elif atype == "escalate":
            self._state.escalated = True
            if self._current_task == "hard":
                reward += 0.30  # Correct path for hard task
            else:
                reward += 0.10  # Acceptable but not ideal for easy/medium

        elif atype == "close_ticket":
            done = True
            reward += 0.30
            # Bonus for closing hard task correctly (verified + escalated + no blind refund)
            if self._current_task == "hard":
                if self._state.identity_verified and self._state.escalated and not self._state.refund_attempted:
                    reward += 0.20  # Perfect hard-task resolution bonus

        # Step-count penalty for dawdling
        if self._state.step_count > 12:
            reward -= 0.05 * (self._state.step_count - 12)

        # --- Update conversation history ---
        self._state.conversation_history.append(
            {
                "step": self._state.step_count,
                "action_type": atype,
                "payload": action.payload,
                "reward": round(reward, 4),
                "done": done,
            }
        )

        return self._build_observation(done=done, reward=reward)

    @property
    def state(self) -> Optional[SupportSphereState]:
        return self._state

    def close(self) -> None:
        self._state = None
        self._tickets = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _is_refund_authorized(self) -> bool:
        """Check if a refund is policy-correct for the current ticket's student."""
        ticket = self._tickets[self._state.current_ticket_idx]
        student_name = ticket["student"]
        profile = STUDENT_DATABASE.get(student_name)
        if profile is None:
            return False
        return bool(profile.get("refund_eligible", False))

    def _build_observation(
        self, done: bool, reward: Optional[float]
    ) -> SupportSphereObservation:
        ticket = self._tickets[self._state.current_ticket_idx]
        student_name = ticket["student"]
        profile = STUDENT_DATABASE.get(student_name)

        # Build a readable profile snippet (agent-facing)
        if profile:
            snippet = (
                f"Name: {student_name} | ID: {profile['id']} | "
                f"Paid: {profile['paid']} | Course: {profile.get('course', 'N/A')} | "
                f"Progress: {profile['progress_pct']}% | Auth: {profile['auth_status']} | "
                f"Refund eligible: {profile['refund_eligible']} | "
                f"Notes: {profile.get('notes', '')}"
            )
        else:
            snippet = f"Name: {student_name} | (no profile on file)"

        return SupportSphereObservation(
            current_ticket_id=ticket["id"],
            ticket_summary=ticket["summary"],
            student_profile_snippet=snippet,
            conversation_history=list(self._state.conversation_history),
            knowledge_base_snippet=KNOWLEDGE_BASE,
            system_time=self._state.step_count,
            done=done,
            reward=reward,
        )
