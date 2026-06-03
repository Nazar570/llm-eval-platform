from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import DriftEvent, Evaluation, ModelVersion
from training.session import AsyncSessionLocal, dispose_engine

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 7
DEFAULT_OUTPUT_PATH = "data/eval_report.md"


def _window_start() -> datetime:
    days = int(os.getenv("EVAL_REPORT_WINDOW_DAYS", str(DEFAULT_WINDOW_DAYS)))
    return datetime.now(UTC) - timedelta(days=days)


async def _evaluation_rows(db: AsyncSession, since: datetime) -> list[tuple[str, str, int, float]]:
    statement = (
        select(
            Evaluation.scorer,
            Evaluation.metric,
            func.count(Evaluation.id),
            func.avg(Evaluation.score),
        )
        .where(Evaluation.scored_at >= since)
        .group_by(Evaluation.scorer, Evaluation.metric)
        .order_by(Evaluation.scorer, Evaluation.metric)
    )
    rows = (await db.execute(statement)).all()
    return [(scorer, metric, int(count), float(avg or 0.0)) for scorer, metric, count, avg in rows]


async def _drift_rows(db: AsyncSession, since: datetime) -> list[tuple[str, int, float]]:
    statement = (
        select(
            DriftEvent.metric,
            func.count(DriftEvent.id),
            func.max(DriftEvent.drift_score),
        )
        .where(DriftEvent.detected_at >= since, DriftEvent.drift_detected.is_(True))
        .group_by(DriftEvent.metric)
        .order_by(DriftEvent.metric)
    )
    rows = (await db.execute(statement)).all()
    return [(metric, int(count), float(score or 0.0)) for metric, count, score in rows]


async def _model_rows(db: AsyncSession) -> list[ModelVersion]:
    statement = select(ModelVersion).order_by(ModelVersion.registered_at.desc()).limit(10)
    rows = (await db.execute(statement)).scalars().all()
    return list(rows)


def _render_markdown(
    since: datetime,
    evaluations: list[tuple[str, str, int, float]],
    drifts: list[tuple[str, int, float]],
    models: list[ModelVersion],
) -> str:
    lines: list[str] = []
    lines.append("# LLM Evaluation Report")
    lines.append("")
    lines.append(f"Window start (UTC): {since.isoformat()}")
    lines.append("")
    lines.append("## Evaluation Scores")
    lines.append("")
    if evaluations:
        lines.append("| Scorer | Metric | Count | Avg Score |")
        lines.append("| --- | --- | ---: | ---: |")
        for scorer, metric, count, avg in evaluations:
            lines.append(f"| {scorer} | {metric} | {count} | {avg:.4f} |")
    else:
        lines.append("No evaluations recorded in the window.")
    lines.append("")
    lines.append("## Drift Events")
    lines.append("")
    if drifts:
        lines.append("| Metric | Detected Count | Max Drift Score |")
        lines.append("| --- | ---: | ---: |")
        for metric, count, score in drifts:
            lines.append(f"| {metric} | {count} | {score:.4f} |")
    else:
        lines.append("No drift detected in the window.")
    lines.append("")
    lines.append("## Registered Model Versions")
    lines.append("")
    if models:
        lines.append("| Model | Version | Stage | Accuracy | F1 | Registered At |")
        lines.append("| --- | --- | --- | ---: | ---: | --- |")
        for model in models:
            accuracy = f"{model.eval_accuracy:.4f}" if model.eval_accuracy is not None else "-"
            f1 = f"{model.eval_f1:.4f}" if model.eval_f1 is not None else "-"
            lines.append(
                f"| {model.registered_model_name} | {model.mlflow_version} | {model.stage} "
                f"| {accuracy} | {f1} | {model.registered_at.isoformat()} |"
            )
    else:
        lines.append("No registered model versions found.")
    lines.append("")
    return "\n".join(lines)


async def build_report() -> str:
    since = _window_start()
    async with AsyncSessionLocal() as db:
        evaluations = await _evaluation_rows(db, since)
        drifts = await _drift_rows(db, since)
        models = await _model_rows(db)
    markdown = _render_markdown(since, evaluations, drifts, models)
    output_path = Path(os.getenv("EVAL_REPORT_OUTPUT", DEFAULT_OUTPUT_PATH))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    logger.info("eval_report wrote report to %s", output_path)
    return markdown


async def _main() -> None:
    try:
        markdown = await build_report()
        print(markdown)
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(_main())
