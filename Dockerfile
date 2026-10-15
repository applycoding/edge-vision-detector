# CPU image for the demo. It does not include a YOLO weight file; mount one
# at /app/models (see scripts/download_model.py).
#
# Jetson / L4T is a different base image and a different ONNX Runtime build.
# See deploy/Dockerfile.jetson and the README. This file was not meant to run
# TensorRT.

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs
COPY samples ./samples
COPY scripts ./scripts

ENTRYPOINT ["edge-vision"]
CMD ["run", "--config", "configs/default.yaml"]
