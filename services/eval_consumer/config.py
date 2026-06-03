from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ConsumerSettings(BaseSettings):
    kafka_bootstrap_servers: str = "kafka:29092"
    kafka_trace_topic: str = "llm-traces"
    kafka_consumer_group: str = "eval-consumer-group"
    kafka_auto_offset_reset: str = "earliest"
    kafka_max_poll_records: int = 8
    eval_database_host: str = "postgres"
    eval_database_port: int = 5432
    eval_database_user: str = "eval"
    eval_database_password: str = "eval"
    eval_database_name: str = "evaldb"
    eval_database_echo: bool = False
    redis_host: str = "redis"
    redis_port: int = 6379
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = 120.0
    metrics_port: int = 8002
    config_path: str = "config/settings.yaml"

    @property
    def async_uri(self) -> str:
        return (
            f"postgresql+asyncpg://{self.eval_database_user}:{self.eval_database_password}"
            f"@{self.eval_database_host}:{self.eval_database_port}/{self.eval_database_name}"
        )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = ConsumerSettings()


class RagasConfig(BaseModel):
    metrics: list[str] = ["faithfulness", "answer_relevancy", "context_recall"]


class BertScoreConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str = "microsoft/deberta-base-mnli"
    lang: str = "en"
    rescale_with_baseline: bool = True


class LlmJudgeConfig(BaseModel):
    scale_min: int = 1
    scale_max: int = 5
    prompt_version: str = "v1"


class ScorerConfig(BaseModel):
    ragas: RagasConfig = Field(default_factory=RagasConfig)
    bert_score: BertScoreConfig = Field(default_factory=BertScoreConfig)
    llm_judge: LlmJudgeConfig = Field(default_factory=LlmJudgeConfig)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    temperature: float = 0.0
    num_ctx: int = 4096


def _load_scorer_config(config_path: str) -> ScorerConfig:
    path = Path(config_path)
    if not path.exists():
        logger.warning("config file %s missing, using defaults", config_path)
        return ScorerConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    scorers_section = raw.get("scorers", {}) or {}
    ollama_section = raw.get("ollama", {}) or {}
    rag_section = raw.get("rag", {}) or {}

    ragas_values: dict[str, Any] = {}
    if "ragas" in scorers_section and scorers_section["ragas"]:
        ragas_values = {k: v for k, v in scorers_section["ragas"].items() if v is not None}

    bert_values: dict[str, Any] = {}
    if "bert_score" in scorers_section and scorers_section["bert_score"]:
        bert_values = {k: v for k, v in scorers_section["bert_score"].items() if v is not None}

    judge_values: dict[str, Any] = {}
    if "llm_judge" in scorers_section and scorers_section["llm_judge"]:
        judge_values = {k: v for k, v in scorers_section["llm_judge"].items() if v is not None}

    top_values: dict[str, Any] = {}
    for key in ("temperature", "num_ctx"):
        if key in ollama_section and ollama_section[key] is not None:
            top_values[key] = ollama_section[key]
    if rag_section.get("embedding_model") is not None:
        top_values["embedding_model"] = rag_section["embedding_model"]

    return ScorerConfig(
        ragas=RagasConfig(**ragas_values),
        bert_score=BertScoreConfig(**bert_values),
        llm_judge=LlmJudgeConfig(**judge_values),
        **top_values,
    )


scorer_config = _load_scorer_config(settings.config_path)
