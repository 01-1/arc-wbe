FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1 \
    FLOPSCOPE_GPU=0 \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH="/workspace/.venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev \
    && rm -rf /tmp/uv-cache

COPY estimator_covariance.py local_engine.py ./
COPY scripts ./scripts

ENTRYPOINT ["python", "scripts/fly_object_entrypoint.py"]
