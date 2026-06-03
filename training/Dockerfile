FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY training/requirements.txt /app/training/requirements.txt
RUN pip install --upgrade pip \
    && pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r /app/training/requirements.txt

COPY training /app/training
COPY db /app/db
COPY config /app/config

CMD ["python", "-m", "training.train"]
