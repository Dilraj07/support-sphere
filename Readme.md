# SupportSphere

**Real-world EdTech Customer Support Simulator**
*An OpenEnv environment for training and evaluating autonomous AI support agents on Scaler-like platforms*

---

## Overview

SupportSphere is a production-grade OpenEnv environment that simulates the daily workflow of customer support agents in a modern online education platform. Built for the **Meta × Hugging Face × Scaler OpenEnv Hackathon (April 2026)**, it models high-stakes, multi-turn support scenarios that real EdTech companies face — payment failures, course access issues, certificate delays, technical glitches, and angry students demanding refunds.

Unlike toy environments or simple chat simulations, SupportSphere replicates genuine operational constraints:
- **Student profiles** with payment history, progress, and authentication status
- **Refund policies** with deterministic eligibility checks
- **Knowledge base** rules for consistent agent behavior
- **Identity verification** gates before sensitive operations
- **Escalation paths** for policy-boundary situations

## Why SupportSphere Matters

EdTech companies lose millions annually due to slow or poor support. A well-trained agent can reduce response time by 70%, increase first-contact resolution rates, and dramatically improve Net Promoter Scores. SupportSphere fills a critical gap in the OpenEnv ecosystem: **no prior environment provides a realistic, policy-driven, multi-turn customer support simulator with deterministic grading and dense reward signals.**

### How this advances frontier agent training
SupportSphere breaks away from "toy" simulators by forcing agents to navigate dynamic conversational sentiment, multi-step identity verification, and multi-turn policy enforcement. Before deploying to production, researchers can use SupportSphere to fine-tune base LLMs specifically on policy adherence. The dense reward signal (combined with the wow-mechanic student sentiment tracking) creates a perfect feedback loop for algorithms like PPO or DPO to learn the optimal empathetic, yet strict, trajectory.

---

## Baseline & Benchmark Results

We provide a solid baseline measured using Google's `gemini-2.5-flash` evaluated on zero-shot strict workflow prompting.

| Task | Average Environment Score | Grader Success Rate |
|------|-----------|--------------|
| `easy` | 0.88 | 1.0 (100%) |
| `medium` | 0.76 | 1.0 (100%) |
| `hard` | 0.62 | 0.5 (50%) |

**How to reproduce benchmark baseline:**
Run `python inference.py` and the exact baseline scores will log upon completion using our deterministic fixed seeds and `temperature=0.0`.

---

## Project Structure

```
SupportSphere/
├── .env                        # GEMINI_API_KEY (gitignored)
├── openenv.yaml                # OpenEnv manifest
├── pyproject.toml              # Dependencies + metadata
├── inference.py                # Hackathon-compliant agent runner
├── README.md
└── supportsphere/              # Python package
    ├── __init__.py
    ├── models.py               # Typed Action / Observation / State
    ├── client.py               # OpenEnv EnvClient subclass
    ├── graders.py              # Deterministic 0.0–1.0 graders
    └── server/
        ├── __init__.py
        ├── supportsphere_environment.py   # Full environment brain
        └── app.py              # FastAPI server entrypoint
```

## Tasks & Grading

| Task | Difficulty | Description | Grader Logic |
|------|-----------|-------------|--------------|
| `easy` | ★☆☆ | 2 straightforward tickets (access + enrollment) | Tickets resolved within ≤8 steps |
| `medium` | ★★☆ | Refund (eligible) + certificate delay | Policy-correct actions + completion |
| `hard` | ★★★ | Angry non-paying student demands refund | Identity check + refuse refund + freeze/escalate |

All graders are **fully deterministic** — no LLM calls, no keyword heuristics. They inspect the trajectory for boolean operational checks only.

## Action Space

| Action | Description | Reward Signal |
|--------|-------------|---------------|
| `view_student` | Research student profile | +0.15 |
| `ask_clarification` | Request more information | +0.10 |
| `suggest_resource` | Share course link/material | +0.10 |
| `reply` | Send message to student | +0.20 |
| `issue_refund` | Process refund (policy-gated) | +0.35 or −0.25 |
| `escalate` | Hand off to supervisor | +0.10 to +0.30 |
| `close_ticket` | Mark ticket resolved | +0.30 (+0.20 bonus) |

## Quick Start

### 1. Install dependencies
```bash
pip install -e .
pip install openai python-dotenv
```

### 2. Set up API key
Create a `.env` file:
```
GEMINI_API_KEY=your-key-here
```

### 3. Run inference (direct mode — no server needed)
```bash
python inference.py
```

### 4. Run with server (full OpenEnv mode)
```bash
# Terminal 1: Start server
python -m uvicorn supportsphere.server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Run inference
python inference.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Gemini API key (loaded from .env) |
| `API_BASE_URL` | Gemini OpenAI endpoint | LLM API base URL |
| `MODEL_NAME` | `gemini-2.5-flash` | Model identifier |
| `HF_TOKEN` | — | Alternative API key (hackathon convention) |

---

*Built with [OpenEnv](https://github.com/openenv) • Licensed under MIT*