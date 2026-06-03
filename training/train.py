from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

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


def _build_splits(examples: list[dict[str, Any]]) -> tuple[Dataset, Dataset]:
    texts = [example["text"] for example in examples]
    labels = [int(example["label"]) for example in examples]

    stratify = None
    if len(set(labels)) > 1:
        counts = Counter(labels)
        if min(counts.values()) >= 2:
            stratify = labels

    x_train, x_eval, y_train, y_eval = train_test_split(
        texts,
        labels,
        test_size=training_runtime.eval_split,
        random_state=training_runtime.seed,
        stratify=stratify,
    )

    train_ds = Dataset.from_dict({"text": x_train, "label": y_train})
    eval_ds = Dataset.from_dict({"text": x_eval, "label": y_eval})
    return train_ds, eval_ds


def _persist_holdout(dataset: Dataset) -> Path:
    holdout_path = Path(training_runtime.model_output_dir) / "holdout.jsonl"
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    with holdout_path.open("w", encoding="utf-8") as handle:
        for text, label in zip(dataset["text"], dataset["label"], strict=True):
            handle.write(json.dumps({"text": text, "label": int(label)}, ensure_ascii=False) + "\n")
    return holdout_path


def _tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def _encode(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=training_runtime.max_length)

    return dataset.map(_encode, batched=True, remove_columns=["text"])


def _compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    logits = prediction.predictions
    references = prediction.label_ids
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(references, predictions)),
        "f1": float(f1_score(references, predictions, average="weighted", zero_division=0)),
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _build_peft_model() -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(
        training_runtime.base_model,
        num_labels=training_runtime.num_labels,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=training_runtime.lora.r,
        lora_alpha=training_runtime.lora.alpha,
        lora_dropout=training_runtime.lora.dropout,
        target_modules=training_runtime.lora.target_modules,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def _build_training_arguments() -> TrainingArguments:
    return TrainingArguments(
        output_dir=training_runtime.model_output_dir,
        per_device_train_batch_size=training_runtime.batch_size,
        per_device_eval_batch_size=training_runtime.batch_size,
        gradient_accumulation_steps=training_runtime.gradient_accumulation_steps,
        learning_rate=training_runtime.learning_rate,
        num_train_epochs=training_runtime.num_epochs,
        fp16=training_runtime.fp16 and torch.cuda.is_available(),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=training_runtime.seed,
        report_to=[],
    )


def train() -> dict[str, Any]:
    examples = _load_examples(training_runtime.dataset_output_path)
    if not examples:
        logger.info("train found no examples, skipping training run")
        return {}

    tokenizer = AutoTokenizer.from_pretrained(training_runtime.base_model)
    train_dataset, eval_dataset = _build_splits(examples)
    holdout_path = _persist_holdout(eval_dataset)

    tokenized_train = _tokenize_dataset(train_dataset, tokenizer)
    tokenized_eval = _tokenize_dataset(eval_dataset, tokenizer)

    labels = np.array(tokenized_train["label"])
    class_counts = np.bincount(labels, minlength=training_runtime.num_labels)
    class_weights = torch.tensor(
        [len(labels) / max(count, 1) for count in class_counts],
        dtype=torch.float32,
    )

    model = _build_peft_model()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=_build_training_arguments(),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(training_runtime.mlflow_experiment_name)

    with mlflow.start_run() as active_run:
        trainer.train()
        metrics = trainer.evaluate()

        mlflow.log_params(
            {
                "base_model": training_runtime.base_model,
                "lora_r": training_runtime.lora.r,
                "lora_alpha": training_runtime.lora.alpha,
                "lora_dropout": training_runtime.lora.dropout,
                "learning_rate": training_runtime.learning_rate,
                "batch_size": training_runtime.batch_size,
                "gradient_accumulation_steps": training_runtime.gradient_accumulation_steps,
                "num_epochs": training_runtime.num_epochs,
                "training_rows": len(tokenized_train),
                "eval_rows": len(tokenized_eval),
            }
        )
        mlflow.log_metrics(
            {
                "eval_accuracy": float(metrics.get("eval_accuracy", 0.0)),
                "eval_f1": float(metrics.get("eval_f1", 0.0)),
                "eval_loss": float(metrics.get("eval_loss", 0.0)),
            }
        )

        output_dir = Path(training_runtime.model_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        payload = {
            "run_id": active_run.info.run_id,
            "eval_accuracy": float(metrics.get("eval_accuracy", 0.0)),
            "eval_f1": float(metrics.get("eval_f1", 0.0)),
            "training_rows": len(tokenized_train),
            "holdout_rows": len(tokenized_eval),
            "model_output_dir": str(output_dir),
            "holdout_path": str(holdout_path),
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("train completed run_id=%s", payload["run_id"])
        return payload


def main() -> None:
    train()


if __name__ == "__main__":
    main()
