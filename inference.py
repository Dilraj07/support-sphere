"""SupportSphere inference script -- Hackathon-compliant.

Runs all 3 tasks (easy, medium, hard) against the local environment,
calls an LLM via the OpenAI-compatible endpoint, and logs
structured [START] / [STEP] / [END] output for judges.

Environment variables (auto-loaded from .env):
    OPENAI_API_KEY  -- API key (injected by validator)
    API_BASE_URL    -- Override; defaults to standard OpenAI endpoint
    MODEL_NAME      -- Override; defaults to gpt-4o-mini
    GEMINI_API_KEY  -- Alternative key (local dev)
    HF_TOKEN        -- Alternative key name (hackathon convention)
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Bootstrap — load .env and resolve config
# ---------------------------------------------------------------------------
load_dotenv()

API_BASE_URL: str = os.getenv(
    "API_BASE_URL",
    "https://api.openai.com/v1",
)
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY: str = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("HF_TOKEN")
    or ""
)

# Model fallback chain — if primary model isn't registered in the validator's
# LiteLLM proxy, we try these in order before giving up.
MODEL_FALLBACKS: List[str] = [
    MODEL_NAME,
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]
# Deduplicate while preserving order
_seen: set = set()
_unique_fallbacks: List[str] = []
for _m in MODEL_FALLBACKS:
    if _m not in _seen:
        _seen.add(_m)
        _unique_fallbacks.append(_m)
MODEL_FALLBACKS = _unique_fallbacks

if not API_KEY:
    print("[FATAL] No API key found. Set OPENAI_API_KEY / GEMINI_API_KEY in .env or export HF_TOKEN.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Import environment + models directly (no server needed for local eval)
# ---------------------------------------------------------------------------
from supportsphere.server.supportsphere_environment import SupportSphereEnvironment
from supportsphere.models import SupportSphereAction
from supportsphere.graders import grade_task

# ---------------------------------------------------------------------------
# Structured logging (hackathon spec)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=supportsphere model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Any = None) -> None:
    action_display = (action[:120] + "…") if len(action) > 120 else action
    print(f"[STEP] step={step} action={action_display!r} reward={reward:.4f} done={done} error={error}", flush=True)


def log_end(success: bool, steps: int, score: float, grader_score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={success} steps={steps} env_score={score:.4f} "
        f"grader_score={grader_score:.4f} rewards={[round(r, 4) for r in rewards]}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt engineering — task-aware system + user messages
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """You are an expert customer support agent for Scaler, a leading EdTech platform.

YOUR WORKFLOW (follow this order STRICTLY for every ticket):
Step 1: Use "view_student" to verify the student's identity.
Step 2: Use "reply" to address the student's concern based on policy.
Step 3: If a refund is needed AND the student is refund-eligible, use "issue_refund". If NOT eligible, use "reply" to explain why, mention freezing the account, then "escalate".
Step 4: Use "close_ticket" to mark the issue as resolved. You MUST close the ticket within 5 steps.

CRITICAL RULES:
- NEVER issue a refund for non-paying or ineligible students.
- For non-paying students demanding refunds: refuse politely, mention "freeze" in your message, then ESCALATE.
- For certificate delays: explain nightly batch processing. Reply once, then close.
- For enrollment queries: provide info, then close.
- You MUST close the ticket. Do NOT keep replying after the issue is addressed.
- After step 3, ALWAYS choose close_ticket.

RESPOND WITH ONLY THIS JSON (no markdown fences, no explanation):
{"action_type": "<view_student|reply|issue_refund|escalate|close_ticket|suggest_resource|ask_clarification>", "payload": {"message": "<your message>"}}"""


def build_user_prompt(
    ticket_id: str,
    ticket_summary: str,
    student_info: str,
    kb: str,
    conversation: List[Dict],
    step: int,
) -> str:
    history_text = ""
    if conversation:
        history_text = "\n".join(
            f"  Step {c['step']}: [{c['action_type']}] {c.get('payload', {}).get('message', '')[:80]}"
            for c in conversation[-6:]
        )
    else:
        history_text = "  (no prior actions — start with view_student)"

    # Adaptive instruction based on step count
    step_hint = ""
    if step == 1:
        step_hint = "INSTRUCTION: Start by using view_student to verify the student."
    elif step >= 4:
        step_hint = "INSTRUCTION: You have taken enough steps. Use close_ticket NOW to resolve this ticket."
    elif step >= 3:
        step_hint = "INSTRUCTION: If the issue is addressed, use close_ticket. If not, take one final action then close."

    return f"""TICKET: {ticket_id}
SUMMARY: {ticket_summary}

STUDENT PROFILE:
{student_info}

KNOWLEDGE BASE (follow these rules):
{kb}

CONVERSATION HISTORY:
{history_text}

CURRENT STEP: {step} of 5 max
{step_hint}

Respond with ONLY a JSON object. No markdown. No explanation."""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
VALID_ACTIONS: set[str] = {
    "view_student", "reply", "issue_refund", "escalate",
    "close_ticket", "suggest_resource", "ask_clarification",
}


def parse_llm_response(raw: str) -> tuple[str, Dict[str, Any]]:
    """Extract action_type and payload from LLM response. Robust fallback."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        action_type = parsed.get("action_type", "reply")
        payload = parsed.get("payload", {})
        if action_type in VALID_ACTIONS:
            return action_type, payload if isinstance(payload, dict) else {"message": str(payload)}
    except json.JSONDecodeError:
        pass

    # Fallback: keyword matching
    lower = raw.lower()
    if "view_student" in lower or "verify" in lower:
        return "view_student", {"message": raw}
    if "issue_refund" in lower:
        return "issue_refund", {"message": raw}
    if "escalate" in lower:
        return "escalate", {"message": raw}
    if "close_ticket" in lower or "close" in lower:
        return "close_ticket", {"message": raw}
    if "suggest_resource" in lower:
        return "suggest_resource", {"message": raw}
    if "ask_clarification" in lower:
        return "ask_clarification", {"message": raw}

    return "reply", {"message": raw}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SupportSphereEnvironment()

    total_grader_score: float = 0.0

    for task_name in ["easy", "medium", "hard"]:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_name.upper()}")
        print(f"{'='*60}")
        log_start(task=task_name, model=MODEL_NAME)

        rewards: List[float] = []
        trajectory: List[Dict] = []
        steps_taken: int = 0
        env_score: float = 0.0
        grader_score: float = 0.0
        success: bool = False

        try:
            obs = env.reset(task=task_name)

            for step in range(1, 8):
                if obs.done:
                    break

                prompt = build_user_prompt(
                    ticket_id=obs.current_ticket_id,
                    ticket_summary=obs.ticket_summary,
                    student_info=obs.student_profile_snippet,
                    kb=obs.knowledge_base_snippet or "",
                    conversation=obs.conversation_history,
                    step=step,
                )

                if step > 1:
                    time.sleep(13)

                raw_text = ""
                # Try each model in the fallback chain; on 400/NotFound rotate to next.
                active_model = MODEL_NAME
                for attempt in range(3 * len(MODEL_FALLBACKS)):
                    model_idx = attempt // 3
                    if model_idx >= len(MODEL_FALLBACKS):
                        break
                    active_model = MODEL_FALLBACKS[model_idx]
                    try:
                        completion = client.chat.completions.create(
                            model=active_model,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.0,
                            max_tokens=400,
                        )
                        raw_text = (completion.choices[0].message.content or "").strip()
                        break
                    except Exception as exc:
                        exc_str = str(exc)
                        if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                            # Rate limit: wait and retry same model
                            wait = 15 * ((attempt % 3) + 1)
                            print(f"  [RATE-LIMIT] model={active_model} retrying in {wait}s (attempt {attempt+1})", flush=True)
                            time.sleep(wait)
                        elif any(code in exc_str for code in ["400", "404", "NotFoundError", "not exist", "model_not_found"]):
                            # Model not available — rotate to next fallback immediately
                            print(f"  [WARN] Model '{active_model}' unavailable: {exc}", file=sys.stderr)
                            attempt = (model_idx + 1) * 3 - 1  # jump to next model slot
                        else:
                            print(f"  [WARN] LLM call failed at step {step}: {exc}", file=sys.stderr)
                            break

                action_type, payload = parse_llm_response(raw_text)
                action = SupportSphereAction(action_type=action_type, payload=payload)

                obs = env.step(action)
                reward: float = obs.reward if obs.reward is not None else 0.0
                rewards.append(reward)
                steps_taken = step

                trajectory.append({
                    "step": step,
                    "action_type": action_type,
                    "payload": payload,
                    "reward": reward,
                    "done": obs.done,
                })

                log_step(step=step, action=f"{action_type}: {payload.get('message', '')[:80]}", reward=reward, done=obs.done)

            # --- Scores ---
            env_score = sum(rewards) / max(len(rewards), 1)
            env_score = max(0.0, min(1.0, float(env_score)))

            raw_grader = grade_task(task_name, trajectory)
            grader_score = max(0.0, min(1.0, float(raw_grader)))
            total_grader_score += grader_score
            success = grader_score >= 0.5

        except Exception as e:
            print(f"  [FATAL] Unhandled exception in task '{task_name}': {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

        finally:
            log_end(success=success, steps=steps_taken, score=env_score, grader_score=grader_score, rewards=rewards)
            print(f"  -> {task_name} env_score={env_score:.3f}  grader_score={grader_score:.3f}")

    # --- Summary ---
    avg_grader: float = total_grader_score / 3.0
    print(f"\n{'='*60}")
    print(f"  OVERALL GRADER SCORE: {avg_grader:.3f} / 1.000")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
