FROM apache/airflow:2.9.3-python3.11

ENV PYTHONPATH=/opt/airflow \
    HF_HOME=/opt/airflow/.cache/huggingface \
    TRAINING_PYTHON=/opt/training-venv/bin/python

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY training/requirements.txt /opt/airflow/training/requirements.txt
RUN python -m venv /opt/training-venv \
    && /opt/training-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/training-venv/bin/pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && /opt/training-venv/bin/pip install --no-cache-dir -r /opt/airflow/training/requirements.txt

COPY db /opt/airflow/db
COPY config /opt/airflow/config

USER airflow
