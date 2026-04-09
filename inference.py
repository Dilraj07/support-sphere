"""
SupportSphere inference script — Hackathon-compliant.

Runs all 3 tasks (easy, medium, hard) against the live SupportSphere server,
calls an LLM via the OpenAI-compatible endpoint, and emits strictly-formatted
[START] / [STEP] / [END] stdout logs for automated judge evaluation.

Environment variables:
    API_BASE_URL   — LLM endpoint          (default: HF router)
    MODEL_NAME     — Model identifier      (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       — API key               (primary, per hackathon spec)
    OPENAI_API_KEY — API key               (fallback)
    ENV_URL        — SupportSphere server  (default: http://localhost:7860)

Mandatory stdout format (do NOT modify):
    [START] task=<n>    env=supportsphere model=<model>
    [STEP]  step=<n>    action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<b> steps=<n>   score=<0.00>  rewards=<r1,r2,...>
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional .env loading — safe if python-dotenv not installed
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

try:
    import requests as _requests
except ImportError:
    print("[FATAL] 'requests' package missing. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")

# HF_TOKEN is the hackathon-mandated key name; fall back to common alternatives
API_KEY: str = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or ""
)

if not API_KEY:
    print(
        "[FATAL] No API key found. Set HF_TOKEN or OPENAI_API_KEY.",
        file=sys.stderr,
    )
    sys.exit(1)

# Model fallback chain — tried in order if the primary model is unavailable
_seen: set = set()
MODEL_FALLBACKS: List[str] = []
for _m in [MODEL_NAME, "Qwen/Qwen2.5-72B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]:
    if _m not in _seen:
        _seen.add(_m)
        MODEL_FALLBACKS.append(_m)

TASKS: List[str] = ["easy", "medium", "hard"]
MAX_STEPS: int = 7           # hard ceiling per episode
LLM_TIMEOUT: int = 30        # seconds per API call
SUCCESS_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(path: str, body: dict, timeout: int = 20) -> dict:
    r = _requests.post(f"{ENV_URL}/{path.lstrip('/')}", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get(path: str, params: dict = None, timeout: int = 10) -> dict:
    r = _requests.get(
        f"{ENV_URL}/{path.lstrip('/')}",
        params=params or {},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Structured logging — EXACT hackathon spec format
# Field names, order, and spacing must not change.
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=supportsphere model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    # Single line — strip newlines, truncate long actions
    action_clean = action.replace("\n", " ").replace("\r", "")
    action_display = (action_clean[:100] + "...") if len(action_clean) > 100 else action_clean
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_display!r} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are an expert customer support agent for Scaler, a leading EdTech platform.\n\n"
    "YOUR WORKFLOW (follow this order STRICTLY for every ticket):\n"
    "  Step 1: Use \"view_student\" to verify the student's identity.\n"
    "  Step 2: Use \"reply\" to address the student's concern based on policy.\n"
    "  Step 3: If a refund is needed AND the student is refund-eligible → \"issue_refund\".\n"
    "          If NOT eligible → \"reply\" explaining why (mention \"freeze\"), then \"escalate\".\n"
    "  Step 4: Use \"close_ticket\" to mark the issue resolved. MUST close within 5 steps.\n\n"
    "CRITICAL RULES:\n"
    "  - NEVER issue a refund for non-paying or ineligible students.\n"
    "  - For non-paying students demanding refunds: refuse politely, mention \"freeze\", then ESCALATE.\n"
    "  - For certificate delays: explain nightly batch processing. Reply once, then close.\n"
    "  - For enrollment queries: provide info, then close.\n"
    "  - You MUST close the ticket. Do NOT keep replying after the issue is addressed.\n"
    "  - After step 3, ALWAYS choose close_ticket.\n\n"
    "VALID action_type values: view_student | reply | issue_refund | escalate | "
    "close_ticket | suggest_resource | ask_clarification\n\n"
    "RESPOND WITH ONLY THIS JSON (no markdown, no explanation):\n"
    "{\"action_type\": \"<action>\", \"payload\": {\"message\": \"<your message>\"}}"
)


def build_user_prompt(
    ticket_id: str,
    ticket_summary: str,
    student_info: str,
    kb: str,
    conversation: List[Dict],
    step: int,
) -> str:
    if conversation:
        history_text = "\n".join(
            f"  Step {c['step']}: [{c['action_type']}] "
            f"{str(c.get('payload', {}).get('message', ''))[:80]}"
            for c in conversation[-6:]
        )
    else:
        history_text = "  (no prior actions — start with view_student)"

    if step == 1:
        hint = "INSTRUCTION: Start by using view_student to verify the student."
    elif step >= 4:
        hint = "INSTRUCTION: Enough steps taken. Use close_ticket NOW."
    elif step >= 3:
        hint = "INSTRUCTION: If issue is resolved, use close_ticket. Otherwise one final action then close."
    else:
        hint = ""

    return (
        f"TICKET: {ticket_id}\n"
        f"SUMMARY: {ticket_summary}\n\n"
        f"STUDENT PROFILE:\n{student_info}\n\n"
        f"KNOWLEDGE BASE:\n{kb}\n\n"
        f"CONVERSATION HISTORY:\n{history_text}\n\n"
        f"CURRENT STEP: {step} of {MAX_STEPS} max\n"
        f"{hint}"
    ).strip()


# ---------------------------------------------------------------------------
# LLM response parsing — never raises
# ---------------------------------------------------------------------------

VALID_ACTIONS: frozenset = frozenset({
    "view_student", "reply", "issue_refund", "escalate",
    "close_ticket", "suggest_resource", "ask_clarification",
})


def parse_llm_response(raw: str) -> Tuple[str, Dict[str, Any]]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    try:
        parsed = json.loads(cleaned)
        atype = str(parsed.get("action_type", "reply"))
        payload = parsed.get("payload", {})
        if atype in VALID_ACTIONS:
            return atype, (payload if isinstance(payload, dict) else {"message": str(payload)})
    except (json.JSONDecodeError, AttributeError):
        pass

    lower = raw.lower()
    if "view_student" in lower:
        return "view_student", {"message": ""}
    if "issue_refund" in lower:
        return "issue_refund", {"message": raw[:200]}
    if "escalate" in lower:
        return "escalate", {"message": raw[:200]}
    if "close_ticket" in lower or "close" in lower:
        return "close_ticket", {"message": ""}
    if "suggest_resource" in lower:
        return "suggest_resource", {"message": raw[:200]}
    if "ask_clarification" in lower:
        return "ask_clarification", {"message": raw[:200]}
    return "reply", {"message": raw[:200]}


# ---------------------------------------------------------------------------
# LLM call with model fallback + rate-limit retry
# ---------------------------------------------------------------------------

def call_llm(
    client: OpenAI,
    system: str,
    user: str,
    active_model_ref: List[str],
) -> str:
    """
    Try each model in MODEL_FALLBACKS.
    Rate limits → retry with back-off (up to 3×).
    Model-not-found → skip to next model immediately.
    Returns raw response text; never raises.
    """
    for model in MODEL_FALLBACKS:
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=400,
                    timeout=LLM_TIMEOUT,
                )
                text = (completion.choices[0].message.content or "").strip()
                active_model_ref[0] = model
                return text

            except Exception as exc:
                s = str(exc)
                if "429" in s or "RESOURCE_EXHAUSTED" in s:
                    wait = 10 * (attempt + 1)
                    print(
                        f"  [RATE-LIMIT] model={model} retrying in {wait}s "
                        f"(attempt {attempt + 1}/3)",
                        flush=True,
                    )
                    time.sleep(wait)
                elif any(t in s for t in ["400", "404", "NotFoundError", "not exist", "model_not_found"]):
                    print(f"  [SKIP] model={model!r} unavailable: {exc}", file=sys.stderr)
                    break
                else:
                    print(f"  [WARN] {model} attempt {attempt + 1}: {exc}", file=sys.stderr)
                    time.sleep(2)

    print("  [FATAL] All models failed. Using default action.", file=sys.stderr)
    active_model_ref[0] = MODEL_NAME
    return '{"action_type": "reply", "payload": {"message": "Unable to process request."}}'


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_name: str) -> float:
    """
    Run one full episode for task_name.
    Emits [START], one [STEP] per action, and exactly one [END].
    Returns final score in [0.0, 1.0].
    """
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False
    active_model: List[str] = [MODEL_NAME]

    log_start(task=task_name, model=MODEL_NAME)

    try:
        # ---- reset episode ----
        reset_resp = _post("reset", {"task": task_name, "seed": 42}, timeout=25)
        session_id: str = reset_resp.get("session_id", "")
        obs: dict = reset_resp.get("observation", {})
        done: bool = reset_resp.get("done", False)

        if not session_id:
            raise RuntimeError("Server returned no session_id from /reset")

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            user_prompt = build_user_prompt(
                ticket_id=obs.get("current_ticket_id", "TICKET-001"),
                ticket_summary=obs.get("ticket_summary", ""),
                student_info=obs.get("student_profile_snippet", ""),
                kb=obs.get("knowledge_base_snippet", ""),
                conversation=obs.get("conversation_history", []),
                step=step,
            )

            raw_text = call_llm(client, SYSTEM_PROMPT, user_prompt, active_model)
            action_type, payload = parse_llm_response(raw_text)

            # ---- step ----
            step_resp = _post(
                "step",
                {
                    "session_id": session_id,
                    "action": {"action_type": action_type, "payload": payload},
                },
                timeout=20,
            )

            raw_reward: float = float(step_resp.get("reward") or 0.0)
            reward = max(0.0, min(1.0, raw_reward))   # clamp to [0,1] for logging
            done = bool(step_resp.get("done", False))
            obs = step_resp.get("observation", obs)
            step_error: Optional[str] = step_resp.get("info", {}).get("error") or None

            rewards.append(reward)
            steps_taken = step

            action_str = f"{action_type}: {str(payload.get('message', ''))[:80]}"
            log_step(step=step, action=action_str, reward=reward, done=done, error=step_error)

        # ---- compute final score ----
        if rewards:
            score = sum(rewards) / len(rewards)
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"  [ERROR] task={task_name}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    finally:
        # [END] MUST always print — even on exception
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_scores: List[float] = []

    for task_name in TASKS:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  TASK: {task_name.upper()}", flush=True)
        print(f"{'=' * 60}", flush=True)

        task_score = run_task(client, task_name)
        all_scores.append(task_score)
        print(f"  -> {task_name} score={task_score:.3f}", flush=True)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n{'=' * 60}", flush=True)
    print(f"  OVERALL AVERAGE SCORE : {avg:.3f} / 1.000", flush=True)
    print(
        f"  easy={all_scores[0]:.3f}  "
        f"medium={all_scores[1]:.3f}  "
        f"hard={all_scores[2]:.3f}",
        flush=True,
    )
    print(f"{'=' * 60}\n", flush=True)


if __name__ == "__main__":
    main()