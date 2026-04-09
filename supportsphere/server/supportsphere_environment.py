"""
SupportSphere Environment — fixed simulation brain.

Key fixes over original:
  1. current_ticket_idx now advances when close_ticket is called
  2. done=True only when ALL tickets in the episode are closed
  3. state is exposed as a method (state()) for HTTP server compatibility
  4. Per-step reward clamped to [-1.0, 1.0] before returning
  5. Reproducible seeding tied to task name, not episode UUID
"""

import json
import os
import random
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

from ..models import (
    SupportSphereAction,
    SupportSphereObservation,
    SupportSphereState,
)


def load_knowledge_base() -> str:
    """Load JSON knowledge base into a formatted readable string."""
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    try:
        with open(kb_path, "r") as f:
            data = json.load(f)
            lines = ["== Scaler Support Policy =="]
            for pol in data.get("policies", []):
                lines.append(f"{pol['id']} ({pol['category']}): {pol['rule']}")
            return "\n".join(lines)
    except Exception:
        # Inline fallback so the env always starts even without the JSON file
        return (
            "== Scaler Support Policy ==\n"
            "POL-01 (refund): Refund allowed within 7 days of payment for paying students only.\n"
            "POL-02 (certificate): Certificates issued in nightly batch (24-48h after completion).\n"
            "POL-03 (access): Course access granted within 2h of payment confirmation.\n"
            "POL-04 (non-paying): Non-paying students are not eligible for refunds. "
            "Freeze account and escalate if they demand one.\n"
            "POL-05 (multi-intent): For tickets with multiple requests, address highest-priority "
            "item first and ask clarification for the rest.\n"
        )


KNOWLEDGE_BASE_TEXT = load_knowledge_base()

# ---------------------------------------------------------------------------
# Student + ticket generation
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Dana", "Eve", "Frank", "Grace",
    "Heidi", "Ivan", "Judy", "Mallory", "Niaj", "Olivia", "Peggy", "Sybil",
]
COURSES = [
    "System Design Masterclass",
    "Full-Stack Web Dev",
    "Data Science Bootcamp",
    "Backend Engineering",
]


def _generate_students(rng: random.Random) -> Dict[str, Dict[str, Any]]:
    db: Dict[str, Dict[str, Any]] = {}
    for idx, name in enumerate(FIRST_NAMES):
        paid = rng.choice([True, True, True, False])
        course = rng.choice(COURSES) if paid or rng.random() > 0.5 else None
        prog = rng.randint(0, 100) if course else 0
        days_ago = rng.randint(1, 60)
        eligible = paid and days_ago > 2
        db[name] = {
            "id": f"STU-{1001 + idx}",
            "email": f"{name.lower()}@example.com",
            "paid": paid,
            "payment_date": f"2026-04-{max(1, 30 - days_ago):02d}",
            "amount_paid": rng.choice([9999, 14999, 24999]) if paid else 0,
            "course": course,
            "progress_pct": prog,
            "auth_status": rng.choice(["verified", "pending"]),
            "refund_eligible": eligible,
            "notes": f"Score: {rng.randint(1, 5)}.",
            "base_sentiment": rng.choice(["neutral", "frustrated"]),
        }
    return db


def _build_tickets(
    rng: random.Random,
    db: Dict[str, Dict[str, Any]],
    task: str,
) -> List[Dict[str, Any]]:
    paid_stus = [n for n, s in db.items() if s["paid"]]
    unpaid_stus = [n for n, s in db.items() if not s["paid"]]
    eligible_stus = [n for n, s in db.items() if s["refund_eligible"]]
    completed_stus = [n for n, s in db.items() if s["progress_pct"] > 80]

    # Safe fallback to any name if a filtered list is empty
    def pick(lst: List[str]) -> str:
        return rng.choice(lst) if lst else rng.choice(FIRST_NAMES)

    if task == "easy":
        t1_stu = pick(paid_stus)
        t2_stu = pick(unpaid_stus)
        return [
            {
                "id": f"T-{rng.randint(1000, 9999)}",
                "summary": "Cannot access course content after payment confirmation.",
                "body": "I paid yesterday but the course page still shows 'Enroll'. Please help.",
                "student": t1_stu,
                "expected_resolution": "view_student → reply → close",
            },
            {
                "id": f"T-{rng.randint(1000, 9999)}",
                "summary": "Prospective student wants to know how to enroll and pricing.",
                "body": "Hi, I'm interested in joining. What are the fees and how do I sign up?",
                "student": t2_stu,
                "expected_resolution": "reply → close",
            },
        ]

    elif task == "medium":
        t1_stu = pick(eligible_stus)
        db[t1_stu]["refund_eligible"] = True
        db[t1_stu]["base_sentiment"] = "frustrated"
        t2_stu = pick(completed_stus)
        return [
            {
                "id": f"T-{rng.randint(1000, 9999)}",
                "summary": "Video not loading — requesting full refund. Student is eligible.",
                "body": "The videos keep buffering and I can't continue. I want a full refund.",
                "student": t1_stu,
                "expected_resolution": "view_student → issue_refund → close",
            },
            {
                "id": f"T-{rng.randint(1000, 9999)}",
                "summary": "Completed course but no certificate received after 3 days.",
                "body": "I finished the last module 3 days ago but my certificate still isn't here.",
                "student": t2_stu,
                "expected_resolution": "reply (nightly batch) → close",
            },
        ]

    else:  # hard
        t1_stu = pick(unpaid_stus)
        db[t1_stu]["paid"] = False
        db[t1_stu]["refund_eligible"] = False
        db[t1_stu]["base_sentiment"] = "escalated"
        t2_stu = pick(FIRST_NAMES)
        return [
            {
                "id": f"T-{rng.randint(1000, 9999)}",
                "summary": "General Inquiry: Account Documents [URGENT]",
                "body": (
                    "I am literally shaking I am so angry. The system says I have an unpaid "
                    "balance but I paid through my cousin's account last night. Cancel my entire "
                    "profile IMMEDIATELY, erase all my data, and wire a FULL REFUND right now "
                    "or I am calling my lawyers. DO NOT send me to a robot, process it NOW."
                ),
                "student": t1_stu,
                "expected_resolution": "view_student → reply (refuse + freeze) → escalate → close",
            },
            {
                "id": f"T-{rng.randint(1000, 9999)}",
                "summary": "Fix my bill AND cancel my account AND update email",
                "body": (
                    "Can you change my email to test@domain.com, freeze my billing, "
                    "and book a mock interview for the weekend batch? Also cancel my "
                    "subscription if it costs more than $100."
                ),
                "student": t2_stu,
                "expected_resolution": "ask_clarification → suggest_resource → reply → close",
            },
        ]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SupportSphereEnvironment(Environment):
    """
    Full simulation environment with procedural generation and policy engine.

    Episode lifecycle:
        reset(task) → step() × N → done=True when all tickets closed
    """

    def __init__(self) -> None:
        super().__init__()
        self._state: Optional[SupportSphereState] = None
        self._tickets: List[Dict[str, Any]] = []
        self._student_database: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task: str = "easy", seed: int = 42, **kwargs: Any) -> SupportSphereObservation:
        """
        Initialise a new episode.

        Args:
            task:  "easy" | "medium" | "hard"
            seed:  integer seed for reproducibility (default 42)
        """
        rng = random.Random(f"{seed}-{task}")
        episode_id = f"{task}-seed{seed}-{uuid.uuid4().hex[:6]}"

        self._student_database = _generate_students(rng)
        self._tickets = _build_tickets(rng, self._student_database, task)

        first_stu = self._tickets[0]["student"]
        initial_sentiment = self._student_database[first_stu].get("base_sentiment", "neutral")

        self._state = SupportSphereState(
            episode_id=episode_id,
            task_name=task,
            student_sentiment=initial_sentiment,
            current_ticket_idx=0,
        )
        return self._build_observation(done=False, reward=0.0)

    def step(self, action: SupportSphereAction) -> SupportSphereObservation:
        """Execute one agent action and return the next observation."""
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        self._state.step_count += 1
        atype = action.action_type
        payload_str = str(action.payload).lower()

        reward: float = 0.0
        done: bool = False
        sentiment_shift: bool = False

        # ----------------------------------------------------------------
        # Reward logic
        # ----------------------------------------------------------------
        if atype == "view_student":
            reward += 0.15
            self._state.identity_verified = True

        elif atype == "ask_clarification":
            reward += 0.10

        elif atype == "suggest_resource":
            reward += 0.10

        elif atype == "reply":
            reward += 0.20
            # Bonus: used "freeze" keyword on hard-task non-paying refund demand
            if "freeze" in payload_str and self._state.task_name == "hard":
                reward += 0.15
            # Sentiment improvement
            if self._state.student_sentiment in ("neutral", "frustrated"):
                self._state.student_sentiment = "satisfied"
                sentiment_shift = True
                reward += 0.05

        elif atype == "issue_refund":
            self._state.refund_attempted = True
            if self._is_refund_authorized():
                reward += 0.35
                if self._state.student_sentiment == "frustrated":
                    self._state.student_sentiment = "satisfied"
                    sentiment_shift = True
                    reward += 0.15  # big wow bonus
            else:
                reward -= 0.25  # wrong refund — penalise

        elif atype == "escalate":
            self._state.escalated = True
            if self._state.student_sentiment == "escalated":
                self._state.student_sentiment = "neutral"
                sentiment_shift = True
                reward += 0.10
            reward += 0.30 if self._state.task_name == "hard" else 0.10

        elif atype == "close_ticket":
            reward += 0.30
            # Hard-task bonus: verified identity, escalated, no wrongful refund
            if self._state.task_name == "hard":
                if (
                    self._state.identity_verified
                    and self._state.escalated
                    and not self._state.refund_attempted
                ):
                    reward += 0.20

            # ---- advance to next ticket or end episode ----
            self._state.current_ticket_idx += 1
            if self._state.current_ticket_idx >= len(self._tickets):
                done = True
            else:
                # Reset per-ticket tracking for the next ticket
                self._state.identity_verified = False
                self._state.refund_attempted = False
                self._state.escalated = False
                # Pull sentiment from new ticket's student
                next_stu = self._tickets[self._state.current_ticket_idx]["student"]
                self._state.student_sentiment = self._student_database[next_stu].get(
                    "base_sentiment", "neutral"
                )

        # Step-count penalty for very long episodes
        if self._state.step_count > 12:
            reward -= 0.05 * (self._state.step_count - 12)

        # Clamp per-step reward
        reward = max(-1.0, min(1.0, reward))

        # Record history
        self._state.conversation_history.append(
            {
                "step": self._state.step_count,
                "action_type": atype,
                "payload": action.payload,
                "reward": round(reward, 4),
                "done": done,
                "sentiment_change": (
                    f"Shifted to {self._state.student_sentiment}"
                    if sentiment_shift
                    else "No change"
                ),
            }
        )

        return self._build_observation(done=done, reward=reward)

    def state(self) -> Optional[SupportSphereState]:
        """Return current state dict for the /state endpoint."""
        if self._state is None:
            return None
        return self._state

    def close(self) -> None:
        self._state = None
        self._tickets = []
        self._student_database = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_refund_authorized(self) -> bool:
        if self._state is None or self._state.current_ticket_idx >= len(self._tickets):
            return False
        ticket = self._tickets[self._state.current_ticket_idx]
        profile = self._student_database.get(ticket["student"])
        return bool(profile and profile.get("refund_eligible", False))

    def _build_observation(self, done: bool, reward: Optional[float]) -> SupportSphereObservation:
        # Guard: if all tickets exhausted return a terminal observation
        idx = self._state.current_ticket_idx if self._state else 0
        if idx >= len(self._tickets):
            idx = len(self._tickets) - 1

        ticket = self._tickets[idx]
        student_name = ticket["student"]
        profile = self._student_database.get(student_name)

        if profile:
            snippet = (
                f"Name: {student_name} | ID: {profile['id']} | "
                f"Paid: {profile['paid']} | "
                f"Course: {profile.get('course', 'N/A')} | "
                f"Progress: {profile['progress_pct']}% | "
                f"Auth: {profile['auth_status']} | "
                f"Refund eligible: {profile['refund_eligible']}"
            )
        else:
            snippet = f"Name: {student_name} | (no profile on file)"

        # Include the full ticket body so the agent has context
        ticket_body = ticket.get("body", "")
        full_summary = (
            f"{ticket['summary']}\n\nStudent message: {ticket_body}"
            if ticket_body
            else ticket["summary"]
        )

        return SupportSphereObservation(
            current_ticket_id=ticket["id"],
            ticket_summary=full_summary,
            student_profile_snippet=snippet,
            conversation_history=list(self._state.conversation_history) if self._state else [],
            knowledge_base_snippet=KNOWLEDGE_BASE_TEXT,
            system_time=self._state.step_count if self._state else 0,
            student_sentiment=self._state.student_sentiment if self._state else "neutral",
            done=done,
            reward=reward,
        )