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

COPY services/eval_consumer/requirements.txt /app/services/eval_consumer/requirements.txt
RUN pip install --upgrade pip \
    && pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r /app/services/eval_consumer/requirements.txt

COPY services/__init__.py /app/services/__init__.py
COPY services/session.py /app/services/session.py
COPY services/eval_consumer /app/services/eval_consumer
COPY db /app/db
COPY config /app/config

EXPOSE 8002

CMD ["python", "-m", "services.eval_consumer.consumer"]
