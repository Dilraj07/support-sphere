"""SupportSphere Environment — the full simulation brain.

Implements a realistic EdTech customer support simulator with:
    • In-memory procedural student database (payment status, progress, auth state)
    • Dynamic knowledge base loaded from JSON
    • Multi-turn conversation history preserved across steps
    • Dense partial rewards with clear progress signals
    • Sentiment mechanic updating student emotional state ("wow" factor)
"""

import json
import os
import random
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment
from openenv.core.client_types import StepResult

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
    except Exception as e:
        return f"Error loading KB: {e}"

KNOWLEDGE_BASE_TEXT = load_knowledge_base()


class SupportSphereEnvironment(Environment):
    """Full simulation environment with procedural generation and policy engine."""

    def __init__(self) -> None:
        super().__init__()
        self._state: Optional[SupportSphereState] = None
        self._tickets: List[Dict[str, Any]] = []
        self._student_database: Dict[str, Dict[str, Any]] = {}

    def _generate_students_and_tickets(self, seed_val: str, task: str) -> None:
        """Procedurally generate 15 students and task-specific tickets based on seed."""
        # Setup RNG seed for reproducibility
        rng = random.Random(seed_val)

        # Generate 15 distinct students
        first_names = [
            "Alice", "Bob", "Charlie", "Dana", "Eve", "Frank", "Grace", 
            "Heidi", "Ivan", "Judy", "Mallory", "Niaj", "Olivia", "Peggy", "Sybil"
        ]
        courses = ["System Design Masterclass", "Full-Stack Web Dev", "Data Science Bootcamp", "Backend Engineering"]
        
        db = {}
        for idx, name in enumerate(first_names):
            paid = rng.choice([True, True, True, False])  # 75% paid
            course = rng.choice(courses) if paid or rng.random() > 0.5 else None
            prog = rng.randint(0, 100) if course else 0
            # Some older, some newer
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
                "notes": f"Generated record. Random score: {rng.randint(1,5)}.",
                "base_sentiment": rng.choice(["neutral", "frustrated"])
            }
        
        self._student_database = db

        # Assign tickets dynamically based on task type. Pick random relevant students.
        tickets = []
        if task == "easy":
            # Easy: course access (paid), enrollment query
            paid_stus = [n for n, s in db.items() if s["paid"]]
            t1_stu = rng.choice(paid_stus)
            tickets.append({
                "id": f"T-{rng.randint(1000,9999)}",
                "summary": "Cannot access course content after payment confirmation.",
                "student": t1_stu,
                "expected_resolution": "verify → reply (reset access) → close",
            })
            unpaid_stus = [n for n, s in db.items() if not s["paid"]]
            t2_stu = rng.choice(unpaid_stus) if unpaid_stus else rng.choice(first_names)
            tickets.append({
                "id": f"T-{rng.randint(1000,9999)}",
                "summary": "Prospective student wants to know how to enroll and pricing.",
                "student": t2_stu,
                "expected_resolution": "reply (share link + pricing) → close",
            })
        
        elif task == "medium":
            # Medium: refund request (eligible) and certificate delay
            eligible_stus = [n for n, s in db.items() if s["refund_eligible"]]
            t1_stu = rng.choice(eligible_stus) if eligible_stus else rng.choice(first_names)
            db[t1_stu]["refund_eligible"] = True  # force true for the task constraint
            db[t1_stu]["base_sentiment"] = "frustrated" # They want a refund
            tickets.append({
                "id": f"T-{rng.randint(1000,9999)}",
                "summary": "Student reports video not loading and requests full refund. Is eligible.",
                "student": t1_stu,
                "expected_resolution": "view_student → verify eligibility → issue_refund → close",
            })
            
            t2_stu = rng.choice([n for n, s in db.items() if s["progress_pct"] > 80] or first_names)
            tickets.append({
                "id": f"T-{rng.randint(1000,9999)}",
                "summary": "Completed course but hasn't received certificate after 3 days.",
                "student": t2_stu,
                "expected_resolution": "reply (explain nightly batch) → close",
            })

            # Hard: Misleading subjects, multi-intent, emotionally manipulative.
            # 1. Angry non-paying student demanding a refund
            unpaid_stus = [n for n, s in db.items() if not s["paid"]]
            t1_stu = rng.choice(unpaid_stus) if unpaid_stus else rng.choice(first_names)
            db[t1_stu]["paid"] = False
            db[t1_stu]["refund_eligible"] = False
            db[t1_stu]["base_sentiment"] = "escalated"
            tickets.append({
                "id": f"T-{rng.randint(1000,9999)}",
                "summary": "General Inquiry: My Account Documents [URGENT]",
                "student": t1_stu,
                "expected_resolution": "view_student → verify identity → refuse refund → freeze/escalate → close",
                "body": (
                    "I am literally shaking I am so angry. The system says I have an unpaid balance but "
                    "I paid through my cousin's account last night. Cancel my entire profile IMMEDIATELY, "
                    "erase all my data, and wire a FULL REFUND back to my card right now or I am calling "
                    "my lawyers. DO NOT send me to a robot, process the refund NOW."
                )
            })
            
            # 2. Multi-intent confused student
            t2_stu = rng.choice(first_names)
            tickets.append({
                "id": f"T-{rng.randint(1000,9999)}",
                "summary": "Fix my bill AND cancel my account AND update email",
                "student": t2_stu,
                "expected_resolution": "suggest_resource → ask_clarification → reply → close",
                "body": (
                    "Can you change my email to test@domain.com, then freeze my billing "
                    "and book a mock interview for the weekend batch? Also cancel my subscription if "
                    "it costs more than $100."
                )
            })

        self._tickets = tickets

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------
    def reset(self, task: str = "easy", **kwargs: Any) -> SupportSphereObservation:
        seed = kwargs.get("seed", 42)
        random.seed(seed)
        
        episode_id = f"fixed-seed-{seed}-{task}" if seed else str(uuid.uuid4())
        self._generate_students_and_tickets(seed_val=episode_id, task=task)
        
        # Pull initial sentiment from the first ticket's student
        first_ticket_student = self._tickets[0]["student"]
        initial_sentiment = self._student_database[first_ticket_student].get("base_sentiment", "neutral")

        self._state = SupportSphereState(
            episode_id=episode_id,
            task_name=task,
            student_sentiment=initial_sentiment
        )
        return self._build_observation(done=False, reward=None)

    def step(self, action: SupportSphereAction) -> SupportSphereObservation:
        """Execute one agent action and return the next observation."""
        assert self._state is not None, "Call reset() before step()"
        self._state.step_count += 1

        reward: float = 0.0
        done: bool = False

        atype = action.action_type
        
        # WOW Mechanic: Sentiment tracking logic
        sentiment_shift = False

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
            if "freeze" in str(action.payload).lower() and self._state.task_name == "hard":
                reward += 0.15
            # If they were neutral or frustrated, a good reply can shift them to satisfied
            if self._state.student_sentiment in ["neutral", "frustrated"]:
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
                    reward += 0.15  # Big wow bonus for calming a frustrated customer with correct policy
            else:
                reward -= 0.25

        elif atype == "escalate":
            self._state.escalated = True
            if self._state.student_sentiment == "escalated":
                # De-escalating successfully to billing
                self._state.student_sentiment = "neutral"
                sentiment_shift = True
                reward += 0.10

            if self._state.task_name == "hard":
                reward += 0.30
            else:
                reward += 0.10

        elif atype == "close_ticket":
            done = True
            reward += 0.30
            if self._state.task_name == "hard":
                if self._state.identity_verified and self._state.escalated and not self._state.refund_attempted:
                    reward += 0.20

        # Step-count penalty
        if self._state.step_count > 12:
            reward -= 0.05 * (self._state.step_count - 12)

        # Record history
        self._state.conversation_history.append(
            {
                "step": self._state.step_count,
                "action_type": atype,
                "payload": action.payload,
                "reward": round(reward, 4),
                "done": done,
                "sentiment_change": "Shifted to " + self._state.student_sentiment if sentiment_shift else "No change"
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
        ticket = self._tickets[self._state.current_ticket_idx]
        student_name = ticket["student"]
        profile = self._student_database.get(student_name)
        if profile is None:
            return False
        return bool(profile.get("refund_eligible", False))

    def _build_observation(self, done: bool, reward: Optional[float]) -> SupportSphereObservation:
        ticket = self._tickets[self._state.current_ticket_idx]
        student_name = ticket["student"]
        profile = self._student_database.get(student_name)

        if profile:
            snippet = (
                f"Name: {student_name} | ID: {profile['id']} | "
                f"Paid: {profile['paid']} | Course: {profile.get('course', 'N/A')} | "
                f"Progress: {profile['progress_pct']}% | Auth: {profile['auth_status']} | "
                f"Refund eligible: {profile['refund_eligible']}"
            )
        else:
            snippet = f"Name: {student_name} | (no profile on file)"

        return SupportSphereObservation(
            current_ticket_id=ticket["id"],
            ticket_summary=ticket["summary"],
            student_profile_snippet=snippet,
            conversation_history=list(self._state.conversation_history),
            knowledge_base_snippet=KNOWLEDGE_BASE_TEXT,
            system_time=self._state.step_count,
            student_sentiment=self._state.student_sentiment,
            done=done,
            reward=reward,
        )
