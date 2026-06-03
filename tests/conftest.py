from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

_MINIMAL_SETTINGS: dict[str, object] = {
    "training": {
        "base_model": "distilbert-base-uncased",
        "max_length": 128,
        "batch_size": 4,
        "num_epochs": 1,
        "fp16": False,
        "min_eval_f1": 0.5,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.05, "target_modules": ["q_lin", "v_lin"]},
    },
    "mlflow": {
        "experiment_name": "test-experiment",
        "registered_model_name": "test-eval-judge",
    },
    "batch_eval": {"schedule_interval_hours": 12, "batch_size": 16},
}


@pytest.fixture
def settings_payload() -> dict[str, object]:
    return _MINIMAL_SETTINGS.copy()


@pytest.fixture
def write_settings(tmp_path: Path) -> Callable[[dict[str, object]], Path]:
    def _write(payload: dict[str, object]) -> Path:
        path = tmp_path / "settings.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def tmp_settings_path(
    write_settings: Callable[[dict[str, object]], Path],
    settings_payload: dict[str, object],
) -> Path:
    return write_settings(settings_payload)
