from __future__ import annotations

import logging

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

LLM_EVAL_SCORE = Gauge(
    "llm_eval_score",
    "Latest evaluation score per metric",
    ["metric"],
)

LLM_EVAL_TRACES_TOTAL = Counter(
    "llm_eval_traces_total",
    "Total traces fully evaluated and persisted",
)

LLM_EVAL_SCORING_DURATION = Histogram(
    "llm_eval_scoring_duration_seconds",
    "Time spent scoring a trace per scorer",
    ["scorer"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

LLM_EVAL_SCORING_FAILURES = Counter(
    "llm_eval_scoring_failures_total",
    "Total scorer failures per scorer",
    ["scorer"],
)


def start_metrics_server(port: int) -> None:
    start_http_server(port)
    logger.info("metrics server listening on port %s", port)
