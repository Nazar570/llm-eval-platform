from __future__ import annotations

import importlib.util
from typing import Any

import pytest

_HEAVY_DEPS = ("torch", "sqlalchemy", "mlflow", "peft", "transformers")
_MISSING = [name for name in _HEAVY_DEPS if importlib.util.find_spec(name) is None]
requires_runtime = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"batch_eval runtime deps not installed: {', '.join(_MISSING)}",
)

if not _MISSING:
    from training import batch_eval as batch_eval_module
    from training.batch_eval import batch_eval


def _count_positive(scores: list[dict[str, Any]]) -> int:
    return sum(1 for entry in scores if entry["predicted_label"] == 1)


def test_count_positive_logic() -> None:
    scores = [
        {"predicted_label": 1},
        {"predicted_label": 0},
        {"predicted_label": 1},
        {"predicted_label": 0},
        {"predicted_label": 1},
    ]
    assert _count_positive(scores) == 3
    assert _count_positive([]) == 0
    assert _count_positive([{"predicted_label": 0}, {"predicted_label": 0}]) == 0


class _FakeModel:
    def to(self, _device: Any) -> _FakeModel:
        return self


class _FakeAsyncSession:
    async def __aenter__(self) -> _FakeAsyncSession:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    pushed: dict[str, Any] = {}

    monkeypatch.setattr(batch_eval_module, "_load_judge", lambda uri: (_FakeModel(), object()))
    monkeypatch.setattr(batch_eval_module, "AsyncSessionLocal", _FakeAsyncSession)

    def _fake_push(registered_model: str, scored: int, positive: int) -> None:
        pushed["registered_model"] = registered_model
        pushed["scored"] = scored
        pushed["positive"] = positive

    monkeypatch.setattr(batch_eval_module, "push_batch_eval_metrics", _fake_push)
    return pushed


@requires_runtime
async def test_batch_eval_no_model_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    pushed = _patch_common(monkeypatch)
    monkeypatch.setattr(batch_eval_module, "_latest_model_artifact_uri", lambda: None)

    result = await batch_eval()

    assert result == 0
    assert pushed == {}


@requires_runtime
async def test_batch_eval_no_traces_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    pushed = _patch_common(monkeypatch)
    monkeypatch.setattr(batch_eval_module, "_latest_model_artifact_uri", lambda: "runs:/abc/model")

    async def _empty(_db: Any) -> list[Any]:
        return []

    monkeypatch.setattr(batch_eval_module, "_fetch_unscored_traces", _empty)

    result = await batch_eval()

    assert result == 0
    assert pushed == {}


@requires_runtime
async def test_batch_eval_scores_and_pushes(monkeypatch: pytest.MonkeyPatch) -> None:
    pushed = _patch_common(monkeypatch)
    monkeypatch.setattr(batch_eval_module, "_latest_model_artifact_uri", lambda: "runs:/abc/model")

    fake_traces = [object(), object(), object()]

    async def _traces(_db: Any) -> list[Any]:
        return fake_traces

    fake_scores = [
        {"trace_pk": 1, "score": 0.9, "predicted_label": 1, "confidence": 0.9},
        {"trace_pk": 2, "score": 0.2, "predicted_label": 0, "confidence": 0.8},
        {"trace_pk": 3, "score": 0.7, "predicted_label": 1, "confidence": 0.7},
    ]

    async def _persist(_db: Any, _scores: list[dict[str, Any]]) -> None:
        return None

    monkeypatch.setattr(batch_eval_module, "_fetch_unscored_traces", _traces)
    monkeypatch.setattr(batch_eval_module, "_score_batch", lambda model, tok, traces, device: fake_scores)
    monkeypatch.setattr(batch_eval_module, "_persist_scores", _persist)

    result = await batch_eval()

    assert result == 3
    assert pushed["registered_model"] == batch_eval_module.training_runtime.registered_model_name
    assert pushed["scored"] == 3
    assert pushed["positive"] == 2
