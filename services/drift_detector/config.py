from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class DriftSettings(BaseSettings):
    eval_database_host: str = "postgres"
    eval_database_port: int = 5432
    eval_database_user: str = "eval"
    eval_database_password: str = "eval"
    eval_database_name: str = "evaldb"
    eval_database_echo: bool = False
    airflow_webhook_url: str = "http://airflow-webserver:8080/api/v1/dags/retrain_trigger/dagRuns"
    airflow_username: str = "admin"
    airflow_password: str = "admin"
    webhook_timeout_seconds: float = 30.0
    metrics_port: int = 8003
    config_path: str = "config/settings.yaml"

    @property
    def async_uri(self) -> str:
        return (
            f"postgresql+asyncpg://{self.eval_database_user}:{self.eval_database_password}"
            f"@{self.eval_database_host}:{self.eval_database_port}/{self.eval_database_name}"
        )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = DriftSettings()


class DriftRuntimeConfig(BaseModel):
    reference_window_size: int = 200
    current_window_size: int = 100
    check_interval_seconds: int = 300
    stattest: str = "ks"
    stattest_threshold: float = 0.05
    drifted_metric_share: float = 0.5
    monitored_metrics: list[str] = Field(default_factory=list)


def _load_drift_runtime(config_path: str) -> DriftRuntimeConfig:
    path = Path(config_path)
    if not path.exists():
        logger.warning("config file %s missing, using defaults", config_path)
        return DriftRuntimeConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    drift_section = raw.get("drift", {}) or {}
    thresholds_section = raw.get("thresholds", {}) or {}

    values: dict[str, Any] = {}
    for key in (
        "reference_window_size",
        "current_window_size",
        "check_interval_seconds",
        "stattest",
        "stattest_threshold",
        "drifted_metric_share",
    ):
        if drift_section.get(key) is not None:
            values[key] = drift_section[key]

    monitored_metrics = [str(metric) for metric in thresholds_section]
    return DriftRuntimeConfig(monitored_metrics=monitored_metrics, **values)


drift_runtime = _load_drift_runtime(settings.config_path)
