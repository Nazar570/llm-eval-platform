FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY services/rag_app/requirements.txt /app/services/rag_app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/services/rag_app/requirements.txt

COPY services/__init__.py /app/services/__init__.py
COPY services/rag_app /app/services/rag_app
COPY config /app/config

EXPOSE 8001

HEALTHCHECK --interval=20s --timeout=10s --retries=5 --start-period=30s \
    CMD curl -sf http://localhost:8001/health || exit 1

CMD ["uvicorn", "services.rag_app.main:app", "--host", "0.0.0.0", "--port", "8001"]
