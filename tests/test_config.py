from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from training.config import TrainingRuntimeConfig, TrainingSettings, _load_training_runtime


def test_load_training_runtime_reads_valid_settings(tmp_settings_path: Path) -> None:
    runtime = _load_training_runtime(str(tmp_settings_path))
    assert runtime.base_model == "distilbert-base-uncased"
    assert runtime.max_length == 128
    assert runtime.batch_size == 4
    assert runtime.num_epochs == 1
    assert runtime.fp16 is False
    assert runtime.min_eval_f1 == 0.5
    assert runtime.lora.r == 4
    assert runtime.lora.target_modules == ["q_lin", "v_lin"]
    assert runtime.batch_eval_size == 16
    assert runtime.mlflow_experiment_name == "test-experiment"
    assert runtime.registered_model_name == "test-eval-judge"


def test_load_training_runtime_missing_file_uses_defaults(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    runtime = _load_training_runtime(str(missing))
    assert runtime == TrainingRuntimeConfig()
    assert runtime.base_model == "distilbert-base-uncased"
    assert runtime.batch_eval_size == 64


def test_none_and_missing_keys_do_not_override_defaults(
    write_settings: Callable[[dict[str, object]], Path],
) -> None:
    payload: dict[str, object] = {
        "training": {"base_model": None, "max_length": 320},
        "mlflow": {},
        "batch_eval": {},
    }
    path = write_settings(payload)
    defaults = TrainingRuntimeConfig()
    runtime = _load_training_runtime(str(path))
    assert runtime.base_model == defaults.base_model
    assert runtime.max_length == 320
    assert runtime.batch_eval_size == defaults.batch_eval_size
    assert runtime.registered_model_name == defaults.registered_model_name


def test_training_settings_reads_eval_config_path_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_CONFIG_PATH", "/opt/airflow/config/settings.yaml")
    settings = TrainingSettings()
    assert settings.eval_config_path == "/opt/airflow/config/settings.yaml"


def test_async_uri_composed_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_DATABASE_USER", "tester")
    monkeypatch.setenv("EVAL_DATABASE_PASSWORD", "secret")
    monkeypatch.setenv("EVAL_DATABASE_HOST", "db-host")
    monkeypatch.setenv("EVAL_DATABASE_PORT", "6543")
    monkeypatch.setenv("EVAL_DATABASE_NAME", "customdb")
    settings = TrainingSettings()
    assert settings.async_uri == "postgresql+asyncpg://tester:secret@db-host:6543/customdb"
