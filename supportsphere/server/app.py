"""
SupportSphere FastAPI server.

Exposes /reset, /step, /state, and /health over HTTP.
Session-based: each /reset returns a session_id used in subsequent /step calls.
Sessions older than SESSION_TTL_SECONDS are purged automatically on each /reset.
"""

import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from supportsphere.server.supportsphere_environment import SupportSphereEnvironment
from supportsphere.models import SupportSphereAction

app = FastAPI(title="SupportSphere", version="1.0.0")

# ---------------------------------------------------------------------------
# Session store  {session_id: {"env": SupportSphereEnvironment, "created_at": float}}
# ---------------------------------------------------------------------------
_sessions: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SECONDS = 600  # 10 minutes


def _purge_old_sessions() -> None:
    now = time.time()
    stale = [sid for sid, s in _sessions.items() if now - s["created_at"] > SESSION_TTL_SECONDS]
    for sid in stale:
        try:
            _sessions[sid]["env"].close()
        except Exception:
            pass
        del _sessions[sid]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = None) -> Dict[str, Any]:
    """Create a new episode. Returns session_id + initial observation."""
    _purge_old_sessions()
    req = req or ResetRequest()

    sid = str(uuid.uuid4())
    env = SupportSphereEnvironment()
    obs = env.reset(task=req.task, seed=req.seed)

    _sessions[sid] = {"env": env, "created_at": time.time()}

    return {
        "session_id": sid,
        "observation": obs.dict() if hasattr(obs, "dict") else vars(obs),
        "done": False,
        "reward": 0.0,
    }


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    """Execute one action. Returns observation, reward, done, info."""
    session = _sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    env: SupportSphereEnvironment = session["env"]

    try:
        action = SupportSphereAction(**req.action)
    except Exception as exc:
        # Invalid action format → return zero reward, don't crash
        return {
            "observation": {},
            "reward": 0.0,
            "done": False,
            "info": {"error": f"Invalid action: {exc}"},
        }

    obs = env.step(action)

    obs_dict = obs.dict() if hasattr(obs, "dict") else vars(obs)
    reward = float(obs.reward) if obs.reward is not None else 0.0
    done = bool(obs.done)

    return {
        "observation": obs_dict,
        "reward": reward,
        "done": done,
        "info": {},
    }


@app.get("/state")
def state(session_id: str) -> Dict[str, Any]:
    """Return current episode state."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    env: SupportSphereEnvironment = session["env"]
    # state() is a method in the fixed environment
    s = env.state()
    if s is None:
        return {"error": "No active state. Call /reset first."}

    state_dict = s.dict() if hasattr(s, "dict") else vars(s)
    return state_dict
