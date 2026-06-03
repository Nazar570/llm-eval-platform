from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.config import settings, training_runtime

logger = logging.getLogger(__name__)


def _load_examples(dataset_path: str) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        logger.warning("dataset file %s missing", dataset_path)
        return []
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            examples.append(json.loads(stripped))
    return examples


def _read_run_metadata() -> dict[str, Any]:
    metadata_path = Path(training_runtime.model_output_dir) / "run_metadata.json"
    if not metadata_path.exists():
        logger.warning("run metadata %s missing", metadata_path)
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _build_holdout_split(examples: list[dict[str, Any]]) -> Dataset:
    texts = [example["text"] for example in examples]
    labels = [int(example["label"]) for example in examples]
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    split = dataset.train_test_split(
        test_size=training_runtime.eval_split,
        seed=training_runtime.seed,
    )
    return split["test"]


def _load_trained_model() -> AutoModelForSequenceClassification:
    base_model = AutoModelForSequenceClassification.from_pretrained(
        training_runtime.base_model,
        num_labels=training_runtime.num_labels,
    )
    model = PeftModel.from_pretrained(base_model, training_runtime.model_output_dir)
    model.eval()
    return model


def _predict(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
) -> np.ndarray:
    predictions: list[int] = []
    batch_size = training_runtime.batch_size
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=training_runtime.max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return np.array(predictions)


def _write_run_metadata(payload: dict[str, Any]) -> None:
    metadata_path = Path(training_runtime.model_output_dir) / "run_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate() -> dict[str, Any]:
    examples = _load_examples(training_runtime.dataset_output_path)
    if not examples:
        logger.info("evaluate found no examples, skipping post-train evaluation")
        return {}

    metadata = _read_run_metadata()
    holdout = _build_holdout_split(examples)
    references = np.array(holdout["label"])

    tokenizer = AutoTokenizer.from_pretrained(training_runtime.model_output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_trained_model().to(device)

    predictions = _predict(model, tokenizer, list(holdout["text"]), device)
    accuracy = float(accuracy_score(references, predictions))
    f1 = float(f1_score(references, predictions, average="weighted", zero_division=0))

    metadata["eval_accuracy"] = accuracy
    metadata["eval_f1"] = f1
    metadata["holdout_rows"] = int(len(references))
    _write_run_metadata(metadata)

    run_id = metadata.get("run_id")
    if run_id:
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(
                    {
                        "holdout_accuracy": accuracy,
                        "holdout_f1": f1,
                        "holdout_rows": float(len(references)),
                    }
                )
        except Exception:
            logger.exception("evaluate failed to log holdout metrics to mlflow run %s", run_id)

    logger.info("evaluate completed accuracy=%.4f f1=%.4f rows=%d", accuracy, f1, len(references))
    return {"eval_accuracy": accuracy, "eval_f1": f1, "holdout_rows": int(len(references))}


def _main() -> None:
    evaluate()


if __name__ == "__main__":
    _main()
