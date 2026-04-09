FROM --platform=linux/amd64 python:3.11-slim

# Install uv
RUN pip install uv

# HF Spaces requires a non-root user
RUN useradd -m -u 1000 appuser
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

WORKDIR /app

# Copy lockfile and project manifest first for layer caching
COPY --chown=appuser:appuser pyproject.toml uv.lock ./

# Install dependencies from lockfile exactly — no surprises on judge machine
RUN uv sync --frozen

# Copy the rest of the source
COPY --chown=appuser:appuser . .

# HF Spaces targets 7860 natively
EXPOSE 7860

# Serve the OpenEnv environment natively
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
