from __future__ import annotations

import logging
import os
import time

from prometheus_client import CollectorRegistry, Counter, Gauge, push_to_gateway

logger = logging.getLogger(__name__)

PUSH_JOB = "batch_eval"
DEFAULT_PUSHGATEWAY_URL = "http://pushgateway:9091"

BATCH_EVAL_REGISTRY = CollectorRegistry()

BATCH_EVAL_SCORED_TRACES = Counter(
    "llm_batch_eval_scored_traces_total",
    "Traces scored by the DistilBERT judge in a batch run",
    ["registered_model"],
    registry=BATCH_EVAL_REGISTRY,
)

BATCH_EVAL_POSITIVE_LABELS = Counter(
    "llm_batch_eval_positive_label_total",
    "Traces scored with the positive judge label in a batch run",
    ["registered_model"],
    registry=BATCH_EVAL_REGISTRY,
)

BATCH_EVAL_LAST_RUN = Gauge(
    "llm_batch_eval_last_run_timestamp_seconds",
    "Unix timestamp of the last completed batch evaluation run",
    ["registered_model"],
    registry=BATCH_EVAL_REGISTRY,
)


def push_batch_eval_metrics(registered_model: str, scored: int, positive: int) -> None:
    gateway_url = os.getenv("PROMETHEUS_PUSHGATEWAY_URL", DEFAULT_PUSHGATEWAY_URL)
    BATCH_EVAL_SCORED_TRACES.labels(registered_model=registered_model).inc(scored)
    BATCH_EVAL_POSITIVE_LABELS.labels(registered_model=registered_model).inc(positive)
    BATCH_EVAL_LAST_RUN.labels(registered_model=registered_model).set(time.time())
    try:
        push_to_gateway(gateway_url, job=PUSH_JOB, registry=BATCH_EVAL_REGISTRY)
    except OSError:
        logger.exception("batch_eval failed to push metrics to %s", gateway_url)
