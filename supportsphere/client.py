"""Official OpenEnv client for SupportSphere.

Used by ``inference.py`` (and the hackathon grading harness) to
communicate with the SupportSphere server over HTTP / WebSocket.
"""

from openenv.core.env_client import EnvClient

from .models import SupportSphereAction, SupportSphereObservation, SupportSphereState


class SupportSphereEnv(EnvClient[SupportSphereAction, SupportSphereObservation, SupportSphereState]):
    """Typed client — call ``reset()`` and ``step()`` against the running server."""

    pass
