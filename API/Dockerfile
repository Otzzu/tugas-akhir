# Vulnerability-detection API image: Python + Joern (JDK) + torch/PyG + FastAPI.
# CPU build by default. For GPU, swap the base to an nvidia/cuda + cudnn image and
# install the matching torch cu1xx wheels (see scripts/setup_cloud.sh for versions).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    JOERN_CLI=/opt/joern/joern-cli \
    API_DEVICE=cpu \
    HF_HOME=/app/.hf

# ── system deps: JDK (Joern), build + fetch tools ───────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless curl unzip git build-essential pigz \
    && rm -rf /var/lib/apt/lists/*

# ── Joern (CPG generation) ──────────────────────────────────────────────────
# Pin a release for reproducibility; bump as needed.
ARG JOERN_VERSION=v4.0.230
RUN curl -L "https://github.com/joernio/joern/releases/download/${JOERN_VERSION}/joern-cli.zip" \
        -o /tmp/joern-cli.zip \
    && mkdir -p /opt/joern && unzip -q /tmp/joern-cli.zip -d /opt/joern \
    && rm /tmp/joern-cli.zip \
    && chmod +x /opt/joern/joern-cli/*.sh || true

WORKDIR /app

# ── uv (fast, reproducible installer) ───────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1

# ── Python deps ─────────────────────────────────────────────────────────────
# torch + PyG (CPU wheels), then transformers + the API extra (fastapi, sqlalchemy, ...).
RUN uv pip install --system torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --system torch-geometric==2.7.0 \
    && uv pip install --system torch-scatter torch-sparse \
        -f https://data.pyg.org/whl/torch-2.6.0+cpu.html \
    && uv pip install --system transformers loguru numpy pandas scikit-learn networkx

COPY API/requirements.txt /app/API/requirements.txt
RUN uv pip install --system -r /app/API/requirements.txt

# ── app ─────────────────────────────────────────────────────────────────────
COPY src /app/src
COPY configs /app/configs
COPY API /app/API
COPY checkpoints /app/checkpoints

EXPOSE 8000
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8000"]
