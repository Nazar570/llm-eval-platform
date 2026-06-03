from __future__ import annotations

import logging
from datetime import UTC, datetime

from aiokafka import AIOKafkaProducer
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class QATrace(BaseModel):
    trace_id: str
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    ground_truth: str | None = None
    model_name: str
    latency_ms: float | None = None
    created_at: datetime = Field(default_factory=_utc_now)


class TraceProducer:
    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap_servers,
            value_serializer=lambda value: value.encode("utf-8"),
            key_serializer=lambda key: key.encode("utf-8"),
            enable_idempotence=True,
            acks="all",
        )
        await self._producer.start()
        logger.info("trace producer connected to %s", self._bootstrap_servers)

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None
            logger.info("trace producer stopped")

    async def publish(self, trace: QATrace) -> None:
        if self._producer is None:
            raise RuntimeError("trace producer not started")
        try:
            await self._producer.send_and_wait(
                self._topic,
                key=trace.trace_id,
                value=trace.model_dump_json(),
            )
        except Exception:
            logger.exception("failed to publish trace_id=%s", trace.trace_id)
            raise
