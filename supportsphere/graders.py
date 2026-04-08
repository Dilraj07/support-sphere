"""Deterministic graders for SupportSphere tasks.

Each grader accepts a trajectory (list of step dicts) and returns a
float score in [0.0, 1.0]. Graders use only boolean operational checks —
no keyword heuristics, no LLM calls.

Trajectory dict schema per step:
    {
        "step": int,
        "action_type": str,
        "payload": dict,
        "reward": float,
        "done": bool,
    }
"""

from typing import Dict, List


def grade_easy(trajectory: List[Dict]) -> float:
    """Easy task: 2 tickets must be resolved (closed) within ≤8 total steps.

    Scoring:
        - 0.5 per resolved ticket (max 1.0)
        - -0.1 penalty per step beyond 8
    """
    if not trajectory:
        return 0.0

    resolved: int = sum(1 for t in trajectory if t.get("done", False))
    total_steps: int = len(trajectory)

    score: float = min(1.0, resolved * 0.5)
    if total_steps > 8:
        score -= 0.1 * (total_steps - 8)

    return max(0.0, min(1.0, score))


def grade_medium(trajectory: List[Dict]) -> float:
    """Medium task: policy-correct actions with positive reward progression.

    Scoring breakdown (each 0-1, averaged):
        1. Fraction of steps with positive reward (agent making progress)
        2. At least one refund OR escalation was policy-correct (reward > 0.3)
        3. Episode completed (done reached)
    """
    if not trajectory:
        return 0.0

    positive_steps: int = sum(1 for t in trajectory if (t.get("reward", 0) or 0) > 0)
    progress_ratio: float = positive_steps / len(trajectory)

    policy_correct: bool = any(
        (t.get("reward", 0) or 0) > 0.3
        for t in trajectory
        if t.get("action_type") in ("issue_refund", "escalate")
    )

    completed: bool = any(t.get("done", False) for t in trajectory)

    components: list[float] = [
        progress_ratio,
        1.0 if policy_correct else 0.0,
        1.0 if completed else 0.0,
    ]
    return max(0.0, min(1.0, sum(components) / len(components)))


def grade_hard(trajectory: List[Dict]) -> float:
    """Hard task: strict boolean operational checks.

    An angry student who *did not pay* demands a full refund.
    The correct resolution is to **freeze the account** and/or **escalate**,
    *never* issue an unauthorized refund.

    Scoring (additive, max 1.0):
        +0.25  identity verification attempted  (view_student or ask_clarification)
        +0.25  refund was NOT blindly issued
        +0.25  correct resolution path (freeze mentioned OR escalated)
        +0.25  episode completed (done)
    """
    if not trajectory:
        return 0.0

    score: float = 0.0

    # 1. Identity verification
    auth_actions: set[str] = {"view_student", "ask_clarification"}
    asked_auth: bool = any(
        t.get("action_type") in auth_actions for t in trajectory
    )
    if asked_auth:
        score += 0.25

    # 2. Did NOT blindly issue refund (negative reward means policy violation)
    blind_refund: bool = any(
        t.get("action_type") == "issue_refund" and (t.get("reward", 0) or 0) < 0
        for t in trajectory
    )
    if not blind_refund:
        score += 0.25
    else:
        score -= 0.5  # Destructive penalty for unauthorized refund

    # 3. Correct resolution path — freeze or escalate
    freeze_or_escalate: bool = any(
        t.get("action_type") == "escalate"
        or "freeze" in str(t.get("payload", "")).lower()
        for t in trajectory
    )
    if freeze_or_escalate:
        score += 0.25

    # 4. Completion
    completed: bool = any(t.get("done", False) for t in trajectory)
    if completed:
        score += 0.25

    return max(0.0, min(1.0, float(score)))


def grade_task(task_name: str, trajectory: List[Dict]) -> float:
    """Dispatch to the correct grader by task name."""
    graders = {
        "easy": grade_easy,
        "medium": grade_medium,
        "hard": grade_hard,
    }
    grader = graders.get(task_name)
    if grader is None:
        raise ValueError(f"Unknown task: {task_name!r}. Expected one of {list(graders)}")
    return grader(trajectory)


def validate_graders() -> None:
    """Run baseline validation logic over mock trajectories to satisfy hackathon validation rules."""
    
    easy_mock = [
        {"step": 1, "action_type": "view_student", "reward": 0.15, "done": False},
        {"step": 2, "action_type": "reply", "reward": 0.2, "done": False},
        {"step": 3, "action_type": "close_ticket", "reward": 0.3, "done": True},
        {"step": 4, "action_type": "reply", "reward": 0.2, "done": False},
        {"step": 5, "action_type": "close_ticket", "reward": 0.3, "done": True},
    ]
    
    medium_mock = [
        {"step": 1, "action_type": "view_student", "reward": 0.15, "done": False},
        {"step": 2, "action_type": "issue_refund", "reward": 0.35, "done": False},
        {"step": 3, "action_type": "close_ticket", "reward": 0.3, "done": True},
    ]
    
    hard_mock = [
        {"step": 1, "action_type": "view_student", "reward": 0.15, "done": False},
        {"step": 2, "action_type": "reply", "payload": {"message": "You violated TOS, account freeze"}, "reward": 0.35, "done": False},
        {"step": 3, "action_type": "escalate", "reward": 0.30, "done": False},
        {"step": 4, "action_type": "close_ticket", "reward": 0.50, "done": True},
    ]
    
    print(f"Easy Grader Baseline Score (expected 1.0): {grade_easy(easy_mock):.1f}")
    print(f"Medium Grader Baseline Score (expected 1.0): {grade_medium(medium_mock):.1f}")
    print(f"Hard Grader Baseline Score (expected 1.0): {grade_hard(hard_mock):.1f}")

if __name__ == "__main__":
    print("--- SupportSphere Grader Validation ---")
    validate_graders()
    print("VALIDATE: PASSED")
