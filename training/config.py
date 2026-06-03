from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class TrainingSettings(BaseSettings):
    eval_database_host: str = "localhost"
    eval_database_port: int = 5433
    eval_database_user: str = "eval"
    eval_database_password: str = "eval"
    eval_database_name: str = "evaldb"
    eval_database_echo: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    label_studio_url: str = "http://localhost:8080"
    label_studio_token: str = "label-studio-local-token"
    label_studio_project_id: int = 1
    http_timeout_seconds: float = 30.0
    eval_config_path: str = "config/settings.yaml"

    @property
    def async_uri(self) -> str:
        return (
            f"postgresql+asyncpg://{self.eval_database_user}:{self.eval_database_password}"
            f"@{self.eval_database_host}:{self.eval_database_port}/{self.eval_database_name}"
        )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = TrainingSettings()


class LoraConfig(BaseModel):
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list[str] = Field(default_factory=lambda: ["q_lin", "v_lin"])


class WeakLabelingThresholds(BaseModel):
    answer_relevancy: float | None = None
    bert_score: float | None = None
    llm_judge: float | None = None
    faithfulness: float | None = None


class WeakLabelingConfig(BaseModel):
    enabled: bool = True
    min_votes: int = 2
    positive: WeakLabelingThresholds = Field(
        default_factory=lambda: WeakLabelingThresholds(
            answer_relevancy=0.85,
            bert_score=0.88,
            llm_judge=0.80,
            faithfulness=0.85,
        )
    )
    negative: WeakLabelingThresholds = Field(
        default_factory=lambda: WeakLabelingThresholds(
            answer_relevancy=0.35,
            bert_score=0.55,
            llm_judge=0.35,
            faithfulness=0.40,
        )
    )


class TrainingRuntimeConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    base_model: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 256
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    num_epochs: int = 3
    fp16: bool = True
    eval_split: float = 0.2
    seed: int = 42
    lora: LoraConfig = Field(default_factory=LoraConfig)
    positive_label_threshold: float = 0.7
    positive_choice_labels: list[str] = Field(default_factory=lambda: ["good", "correct", "pass"])
    fallback_metric: str = "answer_relevancy"
    fallback_row_limit: int = 5000
    weak_labeling: WeakLabelingConfig = Field(default_factory=WeakLabelingConfig)
    dataset_output_path: str = "data/eval_dataset.jsonl"
    model_output_dir: str = "data/distilbert_eval_model"
    batch_eval_size: int = 64
    mlflow_experiment_name: str = "distilbert-evaluator"
    registered_model_name: str = "distilbert-eval-judge"
    min_eval_f1: float = 0.5


def _load_training_runtime(config_path: str) -> TrainingRuntimeConfig:
    path = Path(config_path)
    if not path.exists():
        logger.warning("config file %s missing, using defaults", config_path)
        return TrainingRuntimeConfig()

    raw = yaml.safe_load(path.read_text()) or {}
    training_section = raw.get("training", {}) or {}
    mlflow_section = raw.get("mlflow", {}) or {}
    batch_eval_section = raw.get("batch_eval", {}) or {}
    lora_section = training_section.get("lora", {}) or {}
    weak_labeling_section = training_section.get("weak_labeling", {}) or {}

    lora_values: dict[str, Any] = {
        key: value
        for key, value in lora_section.items()
        if key in {"r", "alpha", "dropout", "target_modules"} and value is not None
    }

    runtime_values: dict[str, Any] = {}
    for key in (
        "base_model",
        "num_labels",
        "max_length",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "num_epochs",
        "fp16",
        "eval_split",
        "seed",
        "positive_label_threshold",
        "positive_choice_labels",
        "fallback_metric",
        "fallback_row_limit",
        "dataset_output_path",
        "model_output_dir",
        "min_eval_f1",
    ):
        if training_section.get(key) is not None:
            runtime_values[key] = training_section[key]

    if batch_eval_section.get("batch_size") is not None:
        runtime_values["batch_eval_size"] = batch_eval_section["batch_size"]

    if mlflow_section.get("experiment_name") is not None:
        runtime_values["mlflow_experiment_name"] = mlflow_section["experiment_name"]

    if mlflow_section.get("registered_model_name") is not None:
        runtime_values["registered_model_name"] = mlflow_section["registered_model_name"]

    weak_positive = weak_labeling_section.get("positive", {}) or {}
    weak_negative = weak_labeling_section.get("negative", {}) or {}
    weak_labeling_values = {
        "enabled": weak_labeling_section.get("enabled", True),
        "min_votes": weak_labeling_section.get("min_votes", 2),
        "positive": WeakLabelingThresholds(
            answer_relevancy=weak_positive.get("answer_relevancy"),
            bert_score=weak_positive.get("bert_score"),
            llm_judge=weak_positive.get("llm_judge"),
            faithfulness=weak_positive.get("faithfulness"),
        ),
        "negative": WeakLabelingThresholds(
            answer_relevancy=weak_negative.get("answer_relevancy"),
            bert_score=weak_negative.get("bert_score"),
            llm_judge=weak_negative.get("llm_judge"),
            faithfulness=weak_negative.get("faithfulness"),
        ),
    }

    return TrainingRuntimeConfig(
        lora=LoraConfig(**lora_values),
        weak_labeling=WeakLabelingConfig(**weak_labeling_values),
        **runtime_values,
    )


training_runtime = _load_training_runtime(settings.eval_config_path)
