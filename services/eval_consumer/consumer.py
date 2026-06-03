from __future__ import annotations

import asyncio
import json
import logging
import signal
from typing import Any, Protocol

from aiokafka import AIOKafkaConsumer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Evaluation, Trace
from services.eval_consumer.config import settings
from services.eval_consumer.metrics_exporter import (
    LLM_EVAL_SCORE,
    LLM_EVAL_SCORING_DURATION,
    LLM_EVAL_SCORING_FAILURES,
    LLM_EVAL_TRACES_TOTAL,
    start_metrics_server,
)
from services.eval_consumer.scorers.bert_score import BertScoreScorer
from services.eval_consumer.scorers.llm_judge import LlmJudgeScorer
from services.eval_consumer.scorers.ragas_scorer import RagasScorer
from services.session import AsyncSessionLocal, dispose_engine

logger = logging.getLogger(__name__)


class Scorer(Protocol):
    scorer_name: str

    async def score(self, trace: dict[str, Any]) -> dict[str, float]: ...


async def _run_scorer(scorer: Scorer, trace: dict[str, Any]) -> dict[str, float]:
    with LLM_EVAL_SCORING_DURATION.labels(scorer=scorer.scorer_name).time():
        try:
            return await scorer.score(trace)
        except Exception:
            LLM_EVAL_SCORING_FAILURES.labels(scorer=scorer.scorer_name).inc()
            logger.exception(
                "scorer=%s failed for trace_id=%s",
                scorer.scorer_name,
                trace.get("trace_id"),
            )
            return {}


async def _persist_trace(db: AsyncSession, trace: dict[str, Any]) -> int:
    existing = await db.scalar(select(Trace).where(Trace.trace_id == trace["trace_id"]))
    if existing is not None:
        return existing.id
    row = Trace(
        trace_id=trace["trace_id"],
        question=trace["question"],
        answer=trace["answer"],
        contexts=trace.get("contexts", []),
        ground_truth=trace.get("ground_truth"),
        model_name=trace.get("model_name", "unknown"),
        latency_ms=trace.get("latency_ms"),
    )
    db.add(row)
    await db.flush()
    return row.id


async def _persist_results(
    db: AsyncSession,
    trace_pk: int,
    scorers: list[Scorer],
    results: list[dict[str, float]],
) -> None:
    for scorer, metrics in zip(scorers, results, strict=True):
        for metric, score in metrics.items():
            db.add(
                Evaluation(
                    trace_pk=trace_pk,
                    scorer=scorer.scorer_name,
                    metric=metric,
                    score=float(score),
                    detail={},
                )
            )
            LLM_EVAL_SCORE.labels(metric=metric).set(float(score))


async def _handle_trace(trace: dict[str, Any], scorers: list[Scorer]) -> None:
    results = await asyncio.gather(*(_run_scorer(scorer, trace) for scorer in scorers))
    async with AsyncSessionLocal() as db:
        try:
            trace_pk = await _persist_trace(db, trace)
            await _persist_results(db, trace_pk, scorers, results)
            await db.commit()
            LLM_EVAL_TRACES_TOTAL.inc()
        except Exception:
            await db.rollback()
            logger.exception("failed to persist trace_id=%s", trace.get("trace_id"))


def _build_scorers() -> list[Scorer]:
    return [
        RagasScorer(base_url=settings.ollama_base_url, model=settings.ollama_model),
        LlmJudgeScorer(base_url=settings.ollama_base_url, model=settings.ollama_model),
        BertScoreScorer(),
    ]


async def _consume(stop_event: asyncio.Event) -> None:
    scorers = _build_scorers()
    consumer = AIOKafkaConsumer(
        settings.kafka_trace_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        auto_offset_reset=settings.kafka_auto_offset_reset,
        enable_auto_commit=False,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )
    await consumer.start()
    logger.info("eval consumer started on topic %s", settings.kafka_trace_topic)
    try:
        while not stop_event.is_set():
            batch = await consumer.getmany(timeout_ms=1000, max_records=8)
            for _, messages in batch.items():
                for message in messages:
                    await _handle_trace(message.value, scorers)
            if batch:
                await consumer.commit()
    finally:
        await consumer.stop()
        await dispose_engine()
        logger.info("eval consumer stopped")


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)


async def main() -> None:
    start_metrics_server(settings.metrics_port)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, stop_event)
    await _consume(stop_event)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
