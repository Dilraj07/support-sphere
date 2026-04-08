FROM --platform=linux/amd64 python:3.11-slim

# HF Spaces requires a non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Update toolchain, copy project definition and install
RUN pip install --upgrade pip

COPY pyproject.toml /app/
COPY Readme.md /app/
COPY supportsphere /app/supportsphere/

RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir openai python-dotenv

# Copy the rest
COPY . /app/

# HF Spaces targets 7860 natively
EXPOSE 7860

USER appuser

# Serve the OpenEnv environment natively
CMD ["uvicorn", "supportsphere.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
