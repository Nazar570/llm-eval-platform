from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from db.models import ModelVersion
from training.config import settings, training_runtime
from training.session import AsyncSessionLocal, dispose_engine

logger = logging.getLogger(__name__)

ARTIFACT_PATH = "model"


def _read_run_metadata() -> dict[str, Any]:
    metadata_path = Path(training_runtime.model_output_dir) / "run_metadata.json"
    if not metadata_path.exists():
        logger.warning("run metadata %s missing", metadata_path)
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _passes_quality_gate(metadata: dict[str, Any]) -> bool:
    return float(metadata.get("eval_f1", 0.0)) >= training_runtime.min_eval_f1


def _log_and_register(metadata: dict[str, Any]) -> dict[str, Any]:
    run_id = metadata["run_id"]
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts(training_runtime.model_output_dir, artifact_path=ARTIFACT_PATH)

    model_uri = f"runs:/{run_id}/{ARTIFACT_PATH}"
    registered = mlflow.register_model(model_uri, training_runtime.registered_model_name)

    stage = "None"
    if _passes_quality_gate(metadata):
        client.transition_model_version_stage(
            name=training_runtime.registered_model_name,
            version=registered.version,
            stage="Staging",
        )
        stage = "Staging"

    return {
        "mlflow_version": str(registered.version),
        "source": registered.source,
        "stage": stage,
    }


async def _persist_model_version(metadata: dict[str, Any], registration: dict[str, Any]) -> None:
    async with AsyncSessionLocal() as db:
        db.add(
            ModelVersion(
                registered_model_name=training_runtime.registered_model_name,
                mlflow_version=registration["mlflow_version"],
                mlflow_run_id=metadata["run_id"],
                base_model=training_runtime.base_model,
                eval_accuracy=float(metadata["eval_accuracy"]) if metadata.get("eval_accuracy") is not None else None,
                eval_f1=float(metadata["eval_f1"]) if metadata.get("eval_f1") is not None else None,
                training_rows=int(metadata["training_rows"]) if metadata.get("training_rows") is not None else None,
                metrics={
                    "eval_accuracy": metadata.get("eval_accuracy"),
                    "eval_f1": metadata.get("eval_f1"),
                    "holdout_rows": metadata.get("holdout_rows"),
                    "source": registration["source"],
                },
                stage=registration["stage"],
            )
        )
        await db.commit()


async def register() -> dict[str, Any]:
    metadata = _read_run_metadata()
    if not metadata.get("run_id"):
        logger.info("register found no run metadata, skipping registration")
        return {}

    registration = _log_and_register(metadata)
    await _persist_model_version(metadata, registration)
    logger.info(
        "register completed model=%s version=%s stage=%s",
        training_runtime.registered_model_name,
        registration["mlflow_version"],
        registration["stage"],
    )
    return registration


async def _main() -> None:
    try:
        await register()
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(_main())
