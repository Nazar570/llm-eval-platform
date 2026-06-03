from __future__ import annotations

import logging

from prometheus_client import Counter, Gauge, start_http_server

logger = logging.getLogger(__name__)

LLM_DRIFT_DETECTED = Counter(
    "llm_drift_detected_total",
    "Total drift events detected per metric",
    ["metric"],
)

LLM_DRIFT_SCORE = Gauge(
    "llm_drift_score",
    "Latest drift score per metric",
    ["metric"],
)


def start_metrics_server(port: int) -> None:
    start_http_server(port)
    logger.info("drift metrics server listening on port %s", port)
