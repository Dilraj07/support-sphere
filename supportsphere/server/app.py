"""FastAPI application for the SupportSphere OpenEnv server.

Launch with:
    python -m uvicorn supportsphere.server.app:app --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server import create_fastapi_app

from .supportsphere_environment import SupportSphereEnvironment
from ..models import SupportSphereAction, SupportSphereObservation

app = create_fastapi_app(
    env=SupportSphereEnvironment,
    action_cls=SupportSphereAction,
    observation_cls=SupportSphereObservation,
)

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
