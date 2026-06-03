from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select

from db.models import Evaluation, Trace
from training.config import settings, training_runtime
from training.session import AsyncSessionLocal, dispose_engine

logger = logging.getLogger(__name__)


def build_example(
    question: str,
    answer: str,
    label: int,
    source: str,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "text": f"{question}\n\n{answer}",
        "label": int(label),
        "source": source,
        "detail": detail or {},
    }


async def fetch_label_studio_annotations() -> list[dict[str, Any]]:
    headers = {"Authorization": f"Token {settings.label_studio_token}"}
    url = f"{settings.label_studio_url}/api/projects/{settings.label_studio_project_id}/export"
    params = {"exportType": "JSON"}

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        payload = response.json()

    examples: list[dict[str, Any]] = []
    for task in payload:
        data = task.get("data") or {}
        annotations = task.get("annotations") or []
        question = data.get("question")
        answer = data.get("answer")

        if not question or not answer or not annotations:
            continue

        result = annotations[0].get("result") or []
        if not result:
            continue

        choices = result[0].get("value", {}).get("choices", [])
        if not choices:
            continue

        label = 1 if choices[0].lower() in training_runtime.positive_choice_labels else 0
        examples.append(
            build_example(
                question=question,
                answer=answer,
                label=label,
                source="labelstudio",
                detail={"choice": choices[0]},
            )
        )

    return examples


def _vote_label(metrics: dict[str, float]) -> tuple[int | None, dict[str, Any]]:
    weak_cfg = training_runtime.weak_labeling
    positive_cfg = weak_cfg.positive.model_dump()
    negative_cfg = weak_cfg.negative.model_dump()
    min_votes = weak_cfg.min_votes

    pos_votes = 0
    neg_votes = 0
    used: dict[str, float] = {}

    for metric, value in metrics.items():
        if value is None:
            continue
        used[metric] = float(value)

        pos_thr = positive_cfg.get(metric)
        neg_thr = negative_cfg.get(metric)

        if pos_thr is not None and value >= pos_thr:
            pos_votes += 1
        elif neg_thr is not None and value <= neg_thr:
            neg_votes += 1

    if pos_votes >= min_votes and pos_votes > neg_votes:
        return 1, {"pos_votes": pos_votes, "neg_votes": neg_votes, "metrics": used}

    if neg_votes >= min_votes and neg_votes > pos_votes:
        return 0, {"pos_votes": pos_votes, "neg_votes": neg_votes, "metrics": used}

    return None, {"pos_votes": pos_votes, "neg_votes": neg_votes, "metrics": used}


async def fetch_stored_evaluations() -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    async with AsyncSessionLocal() as db:
        trace_rows = (
            await db.execute(
                select(Trace.id, Trace.question, Trace.answer)
                .order_by(Trace.created_at.desc())
                .limit(training_runtime.fallback_row_limit)
            )
        ).all()

        if not trace_rows:
            return examples

        trace_map = {
            trace_id: {"question": question, "answer": answer}
            for trace_id, question, answer in trace_rows
            if question and answer
        }

        metric_rows = (
            await db.execute(
                select(Evaluation.trace_pk, Evaluation.metric, Evaluation.score).where(
                    Evaluation.trace_pk.in_(list(trace_map.keys()))
                )
            )
        ).all()

        grouped: dict[int, dict[str, float]] = defaultdict(dict)
        for trace_pk, metric, score in metric_rows:
            if score is None:
                continue
            grouped[int(trace_pk)][str(metric)] = float(score)

        for trace_pk, metrics in grouped.items():
            base = trace_map.get(trace_pk)
            if not base:
                continue

            label, detail = _vote_label(metrics)
            if label is None:
                continue

            examples.append(
                build_example(
                    question=base["question"],
                    answer=base["answer"],
                    label=label,
                    source="weak-supervision",
                    detail=detail,
                )
            )

    return examples


def deduplicate_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}

    for example in examples:
        key = example["text"]
        if key not in seen:
            seen[key] = example
            continue

        if seen[key].get("source") != "labelstudio" and example.get("source") == "labelstudio":
            seen[key] = example

    return list(seen.values())


def write_dataset(examples: list[dict[str, Any]]) -> Path:
    output_path = Path(training_runtime.dataset_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    return output_path


async def build_dataset() -> int:
    results = await asyncio.gather(
        fetch_label_studio_annotations(),
        fetch_stored_evaluations(),
        return_exceptions=True,
    )
    annotations_result, weak_result = results

    label_studio_examples: list[dict[str, Any]] = []
    weak_examples: list[dict[str, Any]] = []

    if isinstance(annotations_result, list):
        label_studio_examples = annotations_result
    else:
        logger.exception("dataset_builder failed to fetch Label Studio annotations", exc_info=annotations_result)

    if isinstance(weak_result, list):
        weak_examples = weak_result
    else:
        logger.exception("dataset_builder failed to fetch stored evaluations", exc_info=weak_result)

    examples = deduplicate_examples(label_studio_examples + weak_examples)
    if not examples:
        logger.info("dataset_builder found no usable examples, skipping dataset write")
        return 0

    output_path = write_dataset(examples)
    logger.info(
        "dataset_builder wrote %d examples to %s (labelstudio=%d weak=%d)",
        len(examples),
        output_path,
        len(label_studio_examples),
        len(weak_examples),
    )
    return len(examples)


async def main() -> None:
    try:
        await build_dataset()
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
