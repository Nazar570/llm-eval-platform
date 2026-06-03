from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from training import evaluate as evaluate_module
from training import register_model as register_module
from training import train as train_module


def test_load_examples_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "a", "label": 1}\n\n{"text": "b", "label": 0}\n', encoding="utf-8")
    examples = train_module._load_examples(str(path))
    assert [e["text"] for e in examples] == ["a", "b"]


def test_load_examples_missing_file_returns_empty(tmp_path: Path) -> None:
    assert train_module._load_examples(str(tmp_path / "missing.jsonl")) == []


def test_build_splits_partitions_examples() -> None:
    examples = [{"text": f"t{i}", "label": i % 2} for i in range(10)]
    train_ds, eval_ds = train_module._build_splits(examples)
    assert len(train_ds) + len(eval_ds) == 10
    assert len(eval_ds) == 2


def test_compute_metrics_returns_accuracy_and_f1() -> None:
    logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    prediction = SimpleNamespace(predictions=logits, label_ids=np.array([1, 0, 0]))
    metrics = train_module._compute_metrics(prediction)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert "f1" in metrics


def test_persist_holdout_writes_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from datasets import Dataset

    monkeypatch.setattr(train_module.training_runtime, "model_output_dir", str(tmp_path / "model"))
    dataset = Dataset.from_dict({"text": ["a", "b"], "label": [1, 0]})
    holdout_path = train_module._persist_holdout(dataset)
    rows = [json.loads(line) for line in holdout_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"text": "a", "label": 1}, {"text": "b", "label": 0}]


def test_read_run_metadata_missing_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(evaluate_module.training_runtime, "model_output_dir", str(tmp_path))
    assert evaluate_module._read_run_metadata() == {}


def test_build_holdout_split_is_deterministic() -> None:
    examples = [{"text": f"t{i}", "label": i % 2} for i in range(10)]
    holdout = evaluate_module._build_holdout_split(examples)
    assert len(holdout) == 2


class _Encoded(dict):
    def to(self, _device: Any) -> _Encoded:
        return self


class _FakeTokenizer:
    def __call__(self, batch: list[str], **_kwargs: Any) -> _Encoded:
        return _Encoded({"input_ids": [[0] for _ in batch]})


class _FakeModel:
    def __call__(self, input_ids: list[list[int]], **_kwargs: Any) -> SimpleNamespace:
        rows = [[0.1, 0.9] if i % 2 == 0 else [0.9, 0.1] for i in range(len(input_ids))]
        return SimpleNamespace(logits=torch.tensor(rows))


def test_predict_returns_argmax_per_text() -> None:
    predictions = evaluate_module._predict(_FakeModel(), _FakeTokenizer(), ["a", "b", "c"], torch.device("cpu"))
    assert predictions.tolist() == [1, 0, 1]


def test_passes_quality_gate_uses_min_f1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(register_module.training_runtime, "min_eval_f1", 0.5)
    assert register_module._passes_quality_gate({"eval_f1": 0.6}) is True
    assert register_module._passes_quality_gate({"eval_f1": 0.4}) is False
    assert register_module._passes_quality_gate({}) is False
