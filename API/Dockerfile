# Vulnerability-detection API image.
#
# The API is a standalone service: it depends on the `gnn_vuln` model code ONLY as a
# published package (gnn-vuln, from PyPI), never as a source tree. The final image holds
# just the installed package + the thin API/ service layer (no src/, no local wheel build).
#
# CPU build by default. For GPU, pass cu124 torch/pyg indexes as build args (see below).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    JOERN_CLI=/opt/joern/joern-cli \
    API_DEVICE=cpu \
    HF_HOME=/app/.hf

# ── system deps: JDK (Joern), build + fetch tools ───────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-21-jre-headless curl unzip git build-essential pigz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1

# torch + PyG sparse extensions need their own wheel indexes; install them first so
# gnn-vuln's transitive torch / torch-geometric deps are already satisfied.
# CPU by default; override for GPU at build time, e.g.:
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 \
#   --build-arg PYG_FIND_LINKS=https://data.pyg.org/whl/torch-2.6.0+cu124.html
ARG TORCH_VERSION=2.6.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG PYG_FIND_LINKS=https://data.pyg.org/whl/torch-2.6.0+cpu.html
RUN uv pip install --system torch==${TORCH_VERSION} --index-url ${TORCH_INDEX} \
    && uv pip install --system torch-geometric==2.7.0 \
    && uv pip install --system torch-scatter torch-sparse -f ${PYG_FIND_LINKS}

# Install gnn-vuln (from PyPI) + the API web deps. torch/pyg above already satisfy the
# heavy transitive deps; the rest (transformers, numpy, ...) resolve from PyPI.
COPY API/requirements.txt /app/API/requirements.txt
RUN uv pip install --system -r /app/API/requirements.txt

# ── Joern (CPG generation). Placed AFTER the heavy pip layers so bumping the
# Joern version never invalidates the torch/deps cache. `curl -f` fails on 404
# instead of silently leaving /opt/joern empty; the final `test -x` asserts the
# CLI actually extracted. ──
ARG JOERN_VERSION=v4.0.526   # match the local Joern used to build datasets
RUN curl -fL "https://github.com/joernio/joern/releases/download/${JOERN_VERSION}/joern-cli.zip" \
        -o /tmp/joern-cli.zip \
    && mkdir -p /opt/joern \
    && unzip -q /tmp/joern-cli.zip -d /opt/joern \
    && rm /tmp/joern-cli.zip \
    && chmod +x /opt/joern/joern-cli/joern* \
    && test -x /opt/joern/joern-cli/joern-parse

# ── app: the thin service layer + configs. The model/inference/training code lives
# in the installed gnn_vuln package; checkpoints + data are bind-mounted at runtime
# (see docker-compose.yml), not baked into the image. ──
COPY configs /app/configs
COPY API /app/API

EXPOSE 8000
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8000"]
