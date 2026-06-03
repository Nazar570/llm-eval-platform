FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/drift_detector/requirements.txt /app/services/drift_detector/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/services/drift_detector/requirements.txt

COPY services/__init__.py /app/services/__init__.py
COPY services/drift_detector /app/services/drift_detector
COPY db /app/db
COPY config /app/config

EXPOSE 8003

CMD ["python", "-m", "services.drift_detector.detector"]
