from __future__ import annotations

import asyncio
import logging
import math
import signal
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report
from sqlalchemy import select

from db.models import DriftEvent, Evaluation
from services.drift_detector.config import drift_runtime, settings
from services.drift_detector.metrics import (
    LLM_DRIFT_DETECTED,
    LLM_DRIFT_SCORE,
    start_metrics_server,
)
from services.drift_detector.session import AsyncSessionLocal, dispose_engine

logger = logging.getLogger(__name__)


async def _load_metric_scores(metric: str, limit: int) -> list[float]:
    async with AsyncSessionLocal() as db:
        rows = await db.scalars(
            select(Evaluation.score)
            .where(Evaluation.metric == metric)
            .order_by(Evaluation.scored_at.desc())
            .limit(limit)
        )
        scores = list(rows)
    scores.reverse()
    return [float(value) for value in scores]


def _split_windows(
    scores: list[float], reference_size: int, current_size: int
) -> tuple[list[float], list[float]] | None:
    if len(scores) < reference_size + current_size:
        return None
    reference = scores[-(reference_size + current_size) : -current_size]
    current = scores[-current_size:]
    return reference, current


def _run_drift_report(reference: list[float], current: list[float], stattest: str, threshold: float) -> dict[str, Any]:
    reference_frame = pd.DataFrame({"score": reference})
    current_frame = pd.DataFrame({"score": current})
    report = Report(metrics=[ColumnDriftMetric(column_name="score", stattest=stattest, stattest_threshold=threshold)])
    report.run(reference_data=reference_frame, current_data=current_frame)
    result = report.as_dict()
    metric_result = result["metrics"][0]["result"]
    return {
        "drift_detected": bool(metric_result["drift_detected"]),
        "drift_score": float(metric_result["drift_score"]),
        "stattest": str(metric_result.get("stattest_name", stattest)),
    }


async def _persist_drift_event(
    metric: str,
    outcome: dict[str, Any],
    reference_size: int,
    current_size: int,
) -> None:
    async with AsyncSessionLocal() as db:
        try:
            db.add(
                DriftEvent(
                    metric=metric,
                    drift_score=outcome["drift_score"],
                    stattest=outcome["stattest"],
                    drift_detected=outcome["drift_detected"],
                    reference_window_size=reference_size,
                    current_window_size=current_size,
                    report=outcome,
                )
            )
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception("failed to persist drift event for metric=%s", metric)


async def _fire_retrain_webhook(client: httpx.AsyncClient, drifted_metrics: list[str]) -> None:
    payload = {
        "conf": {
            "drifted_metrics": drifted_metrics,
            "triggered_at": datetime.now(UTC).isoformat(),
        }
    }
    try:
        response = await client.post(
            settings.airflow_webhook_url,
            json=payload,
            auth=(settings.airflow_username, settings.airflow_password),
        )
        response.raise_for_status()
        logger.info("retrain webhook fired for metrics=%s", drifted_metrics)
    except httpx.HTTPError:
        logger.exception("failed to fire retrain webhook for metrics=%s", drifted_metrics)


async def _check_metric(metric: str) -> bool:
    scores = await _load_metric_scores(metric, drift_runtime.reference_window_size + drift_runtime.current_window_size)
    windows = _split_windows(scores, drift_runtime.reference_window_size, drift_runtime.current_window_size)
    if windows is None:
        logger.info("metric=%s has insufficient samples for drift check", metric)
        return False
    reference, current = windows
    outcome = await asyncio.to_thread(
        _run_drift_report,
        reference,
        current,
        drift_runtime.stattest,
        drift_runtime.stattest_threshold,
    )
    await _persist_drift_event(metric, outcome, len(reference), len(current))
    LLM_DRIFT_SCORE.labels(metric=metric).set(outcome["drift_score"])
    if outcome["drift_detected"]:
        LLM_DRIFT_DETECTED.labels(metric=metric).inc()
        logger.info("drift detected on metric=%s score=%.4f", metric, outcome["drift_score"])
    return bool(outcome["drift_detected"])


async def _run_cycle(client: httpx.AsyncClient) -> None:
    monitored = drift_runtime.monitored_metrics
    results = await asyncio.gather(*(_check_metric(metric) for metric in monitored))
    drifted = [metric for metric, is_drifted in zip(monitored, results, strict=True) if is_drifted]
    required = max(1, math.ceil(len(monitored) * drift_runtime.drifted_metric_share))
    if len(drifted) >= required:
        await _fire_retrain_webhook(client, drifted)


async def _loop(stop_event: asyncio.Event) -> None:
    async with httpx.AsyncClient(timeout=settings.webhook_timeout_seconds) as client:
        while not stop_event.is_set():
            try:
                await _run_cycle(client)
            except Exception:
                logger.exception("drift detection cycle failed")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=drift_runtime.check_interval_seconds)
            except TimeoutError:
                continue
    await dispose_engine()
    logger.info("drift detector stopped")


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)


async def main() -> None:
    start_metrics_server(settings.metrics_port)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, stop_event)
    logger.info("drift detector started, interval=%ss", drift_runtime.check_interval_seconds)
    await _loop(stop_event)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
