from __future__ import annotations

import asyncio
import logging
from typing import Any

import mlflow
import torch
from mlflow.tracking import MlflowClient
from peft import PeftModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from db.models import Evaluation, Trace
from training.config import settings, training_runtime
from training.metrics import push_batch_eval_metrics
from training.session import AsyncSessionLocal, dispose_engine

logger = logging.getLogger(__name__)

JUDGE_SCORER = "distilbert-judge"
JUDGE_METRIC = "judge_label"


def _latest_model_artifact_uri() -> str | None:
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    versions = client.get_latest_versions(training_runtime.registered_model_name)
    if not versions:
        return None
    latest = max(versions, key=lambda version: int(version.version))
    return latest.source


def _load_judge(artifact_uri: str) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    local_path = mlflow.artifacts.download_artifacts(artifact_uri)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        training_runtime.base_model,
        num_labels=training_runtime.num_labels,
    )
    model = PeftModel.from_pretrained(base_model, local_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    return model, tokenizer


async def _fetch_unscored_traces(db: AsyncSession) -> list[Trace]:
    scored_subquery = select(Evaluation.trace_pk).where(Evaluation.scorer == JUDGE_SCORER).scalar_subquery()
    statement = (
        select(Trace)
        .where(Trace.id.not_in(scored_subquery))
        .order_by(Trace.created_at.desc())
        .limit(training_runtime.batch_eval_size)
    )
    rows = (await db.execute(statement)).scalars().all()
    return list(rows)


def _score_batch(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    traces: list[Trace],
    device: torch.device,
) -> list[dict[str, Any]]:
    texts = [f"{trace.question}\n\n{trace.answer}" for trace in traces]
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=training_runtime.max_length,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)
    positive_probs = probabilities[:, 1].cpu().tolist()
    predicted_labels = torch.argmax(probabilities, dim=-1).cpu().tolist()
    confidences = torch.max(probabilities, dim=-1).values.cpu().tolist()
    results: list[dict[str, Any]] = []
    for trace, positive_prob, predicted_label, confidence in zip(
        traces,
        positive_probs,
        predicted_labels,
        confidences,
        strict=True,
    ):
        results.append(
            {
                "trace_pk": trace.id,
                "score": float(positive_prob),
                "predicted_label": int(predicted_label),
                "confidence": float(confidence),
            }
        )
    return results


async def _persist_scores(db: AsyncSession, scores: list[dict[str, Any]]) -> None:
    for entry in scores:
        db.add(
            Evaluation(
                trace_pk=entry["trace_pk"],
                scorer=JUDGE_SCORER,
                metric=JUDGE_METRIC,
                score=entry["score"],
                detail={
                    "predicted_label": entry["predicted_label"],
                    "confidence": entry["confidence"],
                    "positive_probability": entry["score"],
                    "uncertainty": abs(0.5 - entry["score"]),
                },
            )
        )
    await db.commit()


async def batch_eval() -> int:
    artifact_uri = _latest_model_artifact_uri()
    if artifact_uri is None:
        logger.info("batch_eval found no registered model %s, skipping", training_runtime.registered_model_name)
        return 0

    model, tokenizer = _load_judge(artifact_uri)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    async with AsyncSessionLocal() as db:
        traces = await _fetch_unscored_traces(db)
        if not traces:
            logger.info("batch_eval found no unscored traces")
            return 0
        scores = _score_batch(model, tokenizer, traces, device)
        await _persist_scores(db, scores)

    positive = sum(1 for entry in scores if entry["predicted_label"] == 1)
    push_batch_eval_metrics(training_runtime.registered_model_name, len(scores), positive)
    logger.info("batch_eval scored %d traces with %s", len(scores), JUDGE_SCORER)
    return len(scores)


async def _main() -> None:
    try:
        await batch_eval()
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(_main())
