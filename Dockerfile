FROM --platform=linux/amd64 python:3.11-slim

# HF Spaces requires a non-root user
RUN useradd -m -u 1000 appuser

# Switch to the user before installing dependencies
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

WORKDIR /app

# Update toolchain
RUN pip install --no-cache-dir --upgrade pip

COPY --chown=appuser:appuser pyproject.toml /app/
COPY --chown=appuser:appuser README.md /app/
COPY --chown=appuser:appuser supportsphere /app/supportsphere/

RUN pip install --no-cache-dir .
RUN pip install --no-cache-dir openai python-dotenv

# Copy the rest
COPY --chown=appuser:appuser . /app/

# HF Spaces targets 7860 natively
EXPOSE 7860

# Serve the OpenEnv environment natively
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
