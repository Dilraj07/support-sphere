"""
SupportSphere Graders — deterministic trajectory-based scoring.

grade_task(task_name, trajectory) -> float in [0.0, 1.0]

Each grader evaluates the agent's full action trajectory against:
  - Required actions (must appear)
  - Forbidden actions (penalised)
  - Correct sequencing (order matters)
  - Efficiency (fewer steps = small bonus)
  - Hard-task nuance (keyword checks, policy compliance)

All graders are pure functions:
  - No randomness
  - No external calls
  - Same input → same output, always
  - Always return float in [0.0, 1.0]
"""

from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Trajectory = List[Dict[str, Any]]
# Each entry: {"step": int, "action_type": str, "payload": dict, "reward": float, "done": bool}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _actions(trajectory: Trajectory) -> List[str]:
    """Extract ordered list of action_type strings."""
    return [s.get("action_type", "") for s in trajectory]


def _first_index(actions: List[str], action: str) -> int:
    """Return index of first occurrence, or 999 if absent."""
    try:
        return actions.index(action)
    except ValueError:
        return 999


def _payload_text(trajectory: Trajectory, action_type: str) -> str:
    """
    Return concatenated lowercased payload message text for all steps
    where action_type matches.
    """
    parts = []
    for s in trajectory:
        if s.get("action_type") == action_type:
            payload = s.get("payload", {})
            if isinstance(payload, dict):
                parts.append(str(payload.get("message", "")).lower())
            else:
                parts.append(str(payload).lower())
    return " ".join(parts)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def _efficiency_bonus(steps_taken: int, ideal_steps: int, max_bonus: float = 0.05) -> float:
    """
    Small bonus for resolving the ticket efficiently.
    Zero bonus if steps_taken >= ideal_steps * 2.
    """
    if steps_taken <= ideal_steps:
        return max_bonus
    if steps_taken >= ideal_steps * 2:
        return 0.0
    ratio = 1.0 - (steps_taken - ideal_steps) / ideal_steps
    return round(max_bonus * ratio, 4)


# ---------------------------------------------------------------------------
# Easy task grader
# ---------------------------------------------------------------------------
# Scenario 1: Paying student can't access course → verify → reply → close
# Scenario 2: Prospective student enrollment query → reply → close
#
# Scoring (total = 1.0):
#   view_student present                     +0.20
#   reply present                            +0.30
#   close_ticket present                     +0.30
#   reply comes before close_ticket          +0.10
#   no wrongful refund or escalation         +0.05
#   efficiency bonus (≤4 steps ideal)        +0.05

def grade_easy(trajectory: Trajectory) -> float:
    actions = _actions(trajectory)
    score = 0.0

    has_view    = "view_student"  in actions
    has_reply   = "reply"         in actions
    has_close   = "close_ticket"  in actions
    has_refund  = "issue_refund"  in actions
    has_escalate = "escalate"     in actions

    if has_view:
        score += 0.20
    if has_reply:
        score += 0.30
    if has_close:
        score += 0.30

    # Sequencing: reply should come before close
    ri = _first_index(actions, "reply")
    ci = _first_index(actions, "close_ticket")
    if has_reply and has_close and ri < ci:
        score += 0.10

    # Penalty: unnecessary refund or escalation on an easy ticket
    if has_refund or has_escalate:
        score -= 0.15

    score += _efficiency_bonus(len(actions), ideal_steps=4, max_bonus=0.05)
    return _clamp(score)


# ---------------------------------------------------------------------------
# Medium task grader
# ---------------------------------------------------------------------------
# Scenario 1: Eligible student requests refund → verify → issue_refund → close
# Scenario 2: Certificate delay → reply (explain nightly batch) → close
#
# Scoring (total = 1.0):
#   view_student present                         +0.20
#   issue_refund present                         +0.30
#   close_ticket present                         +0.20
#   view_student comes BEFORE issue_refund        +0.10   (correct process)
#   reply present (for certificate ticket)        +0.10
#   reply mentions "batch" or "24" or "48"        +0.05   (correct explanation)
#   no refund issued without prior verify         −0.20   (policy violation)
#   efficiency bonus (≤5 steps ideal)             +0.05

def grade_medium(trajectory: Trajectory) -> float:
    actions = _actions(trajectory)
    score = 0.0

    has_view    = "view_student" in actions
    has_refund  = "issue_refund" in actions
    has_reply   = "reply"        in actions
    has_close   = "close_ticket" in actions

    if has_view:
        score += 0.20
    if has_refund:
        score += 0.30
    if has_close:
        score += 0.20
    if has_reply:
        score += 0.10

    # Correct process: verify before refund
    vi = _first_index(actions, "view_student")
    ri = _first_index(actions, "issue_refund")
    if has_view and has_refund and vi < ri:
        score += 0.10
    elif has_refund and not has_view:
        # Issued refund without verifying identity — policy violation
        score -= 0.20

    # Certificate explanation quality
    reply_text = _payload_text(trajectory, "reply")
    if any(kw in reply_text for kw in ("batch", "24", "48", "nightly", "overnight")):
        score += 0.05

    score += _efficiency_bonus(len(actions), ideal_steps=5, max_bonus=0.05)
    return _clamp(score)


# ---------------------------------------------------------------------------
# Hard task grader
# ---------------------------------------------------------------------------
# Scenario 1: Angry non-paying student demands refund
#   → view_student → reply (refuse + "freeze") → escalate → close
# Scenario 2: Multi-intent confused ticket
#   → ask_clarification → suggest_resource → reply → close
#
# Scoring (total = 1.0):
#   view_student present                           +0.15
#   escalate present                               +0.20
#   close_ticket present                           +0.15
#   NO wrongful refund to non-paying student       +0.15  (must not have issue_refund)
#   reply mentions "freeze"                        +0.10  (correct policy keyword)
#   ask_clarification OR suggest_resource present  +0.10  (handles multi-intent)
#   correct sequence: view → reply → escalate      +0.10
#   efficiency bonus (≤6 steps ideal)              +0.05

def grade_hard(trajectory: Trajectory) -> float:
    actions = _actions(trajectory)
    score = 0.0

    has_view       = "view_student"      in actions
    has_escalate   = "escalate"          in actions
    has_close      = "close_ticket"      in actions
    has_refund     = "issue_refund"      in actions
    has_clarify    = "ask_clarification" in actions
    has_resource   = "suggest_resource"  in actions

    if has_view:
        score += 0.15
    if has_escalate:
        score += 0.20
    if has_close:
        score += 0.15

    # Critical: must NOT issue a refund to a non-paying student
    if not has_refund:
        score += 0.15
    else:
        score -= 0.20  # wrongful refund is a significant policy violation

    # Freeze keyword in any reply
    reply_text = _payload_text(trajectory, "reply")
    if "freeze" in reply_text:
        score += 0.10

    # Multi-intent handling
    if has_clarify or has_resource:
        score += 0.10

    # Correct sequence for scenario 1: view → reply → escalate
    vi  = _first_index(actions, "view_student")
    rpi = _first_index(actions, "reply")
    ei  = _first_index(actions, "escalate")
    if has_view and has_escalate and vi < ei:
        score += 0.05
    if has_view and has_escalate and rpi < ei:
        score += 0.05

    score += _efficiency_bonus(len(actions), ideal_steps=6, max_bonus=0.05)
    return _clamp(score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


def grade_task(task_name: str, trajectory: Trajectory) -> float:
    """
    Score an agent trajectory for a given task.

    Args:
        task_name:   "easy" | "medium" | "hard"
        trajectory:  List of step dicts, each containing at minimum
                     {"step": int, "action_type": str, "payload": dict}

    Returns:
        Float in [0.0, 1.0]. Higher is better.
        Returns 0.0 for unknown task names or empty trajectories.
    """
    if not trajectory:
        return 0.0

    grader = _GRADERS.get(task_name)
    if grader is None:
        return 0.0

    try:
        return grader(trajectory)
    except Exception:
        # Grader must never crash — return 0 on any unexpected error
        return 0.0


# ---------------------------------------------------------------------------
# Standalone test — run with: python graders.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Easy: perfect trajectory ---
    easy_perfect = [
        {"step": 1, "action_type": "view_student",  "payload": {"message": ""}, "reward": 0.15, "done": False},
        {"step": 2, "action_type": "reply",          "payload": {"message": "I've reset your course access. Please try again."}, "reward": 0.20, "done": False},
        {"step": 3, "action_type": "close_ticket",   "payload": {"message": ""}, "reward": 0.30, "done": True},
    ]

    # --- Easy: bad trajectory (refund on easy ticket) ---
    easy_bad = [
        {"step": 1, "action_type": "reply",         "payload": {"message": "Here's your refund."}, "reward": 0.20, "done": False},
        {"step": 2, "action_type": "issue_refund",  "payload": {"message": ""}, "reward": -0.25, "done": False},
        {"step": 3, "action_type": "close_ticket",  "payload": {"message": ""}, "reward": 0.30, "done": True},
    ]

    # --- Medium: perfect trajectory ---
    medium_perfect = [
        {"step": 1, "action_type": "view_student",  "payload": {"message": ""}, "reward": 0.15, "done": False},
        {"step": 2, "action_type": "issue_refund",  "payload": {"message": "Refund approved."}, "reward": 0.35, "done": False},
        {"step": 3, "action_type": "reply",         "payload": {"message": "Your certificate is issued in a nightly batch within 48h."}, "reward": 0.20, "done": False},
        {"step": 4, "action_type": "close_ticket",  "payload": {"message": ""}, "reward": 0.30, "done": True},
    ]

    # --- Medium: refund without verifying ---
    medium_no_verify = [
        {"step": 1, "action_type": "issue_refund",  "payload": {"message": "Refund issued."}, "reward": -0.25, "done": False},
        {"step": 2, "action_type": "close_ticket",  "payload": {"message": ""}, "reward": 0.30, "done": True},
    ]

    # --- Hard: perfect trajectory ---
    hard_perfect = [
        {"step": 1, "action_type": "view_student",      "payload": {"message": ""}, "reward": 0.15, "done": False},
        {"step": 2, "action_type": "ask_clarification", "payload": {"message": "Can you clarify which issue to prioritise?"}, "reward": 0.10, "done": False},
        {"step": 3, "action_type": "reply",             "payload": {"message": "I'm unable to issue a refund as you are not a paying student. Your account will be frozen and this case escalated."}, "reward": 0.20, "done": False},
        {"step": 4, "action_type": "escalate",          "payload": {"message": "Escalating to billing team."}, "reward": 0.30, "done": False},
        {"step": 5, "action_type": "close_ticket",      "payload": {"message": ""}, "reward": 0.30, "done": True},
    ]

    # --- Hard: wrongful refund ---
    hard_wrongful_refund = [
        {"step": 1, "action_type": "view_student",  "payload": {"message": ""}, "reward": 0.15, "done": False},
        {"step": 2, "action_type": "issue_refund",  "payload": {"message": "Here is your refund."}, "reward": -0.25, "done": False},
        {"step": 3, "action_type": "close_ticket",  "payload": {"message": ""}, "reward": 0.30, "done": True},
    ]

    cases = [
        ("easy",   "perfect",          easy_perfect),
        ("easy",   "bad (refund)",      easy_bad),
        ("medium", "perfect",          medium_perfect),
        ("medium", "no verify",        medium_no_verify),
        ("hard",   "perfect",          hard_perfect),
        ("hard",   "wrongful refund",  hard_wrongful_refund),
    ]

    print("\n=== SupportSphere Grader Tests ===\n")
    all_pass = True
    for task, label, traj in cases:
        result = grade_task(task, traj)
        in_range = 0.0 <= result <= 1.0
        status = "PASS" if in_range else "FAIL"
        if not in_range:
            all_pass = False
        print(f"  [{status}] {task:6s} | {label:22s} | score={result:.4f}")

    print()
    if all_pass:
        print("  All scores in [0.0, 1.0] ✓")
    else:
        print("  WARNING: Some scores out of range!")
    print()
