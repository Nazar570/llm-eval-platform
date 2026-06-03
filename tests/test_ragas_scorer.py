from __future__ import annotations

from typing import Any

import pytest

from services.eval_consumer.scorers import ragas_scorer as ragas_module
from services.eval_consumer.scorers.ragas_scorer import RagasScorer


def _bare_scorer(metrics: list[str]) -> RagasScorer:
    scorer = object.__new__(RagasScorer)
    scorer._configured_metrics = metrics
    scorer._llm = None
    scorer._embeddings = None
    return scorer


def test_select_metrics_drops_ground_truth_metrics_when_missing() -> None:
    scorer = _bare_scorer(["faithfulness", "answer_relevancy", "context_recall"])
    assert scorer._select_metrics(has_ground_truth=False) == ["faithfulness", "answer_relevancy"]


def test_select_metrics_keeps_all_with_ground_truth() -> None:
    scorer = _bare_scorer(["faithfulness", "answer_relevancy", "context_recall"])
    assert scorer._select_metrics(has_ground_truth=True) == [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
    ]


def test_build_dataset_normalizes_contexts() -> None:
    scorer = _bare_scorer(["faithfulness"])
    dataset = scorer.build_dataset(
        {"question": "q", "answer": "a", "contexts": ["doc", "  ", "other"]},
        ground_truth="gt",
    )
    row = dataset[0]
    assert row["question"] == "q"
    assert row["contexts"] == ["doc", "other"]
    assert row["ground_truth"] == "gt"


class _FakeRow:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def to_dict(self) -> dict[str, Any]:
        return self._data


class _FakeIloc:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, _index: int) -> _FakeRow:
        return _FakeRow(self._data)


class _FakeFrame:
    def __init__(self, data: dict[str, Any]) -> None:
        self.iloc = _FakeIloc(data)


class _FakeResult:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def to_pandas(self) -> _FakeFrame:
        return _FakeFrame(self._data)


def test_evaluate_cleans_none_and_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = _bare_scorer(["faithfulness", "answer_relevancy"])
    row = {"faithfulness": 0.8, "answer_relevancy": float("nan")}
    monkeypatch.setattr(ragas_module, "evaluate", lambda **_kwargs: _FakeResult(row))

    result = scorer._evaluate({"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "gt"})

    assert result == {"faithfulness": 0.8}


def test_evaluate_returns_empty_when_no_metric_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = _bare_scorer(["context_recall"])

    def _should_not_run(**_kwargs: Any) -> Any:
        raise AssertionError("evaluate must not be called when no metrics selected")

    monkeypatch.setattr(ragas_module, "evaluate", _should_not_run)

    assert scorer._evaluate({"question": "q", "answer": "a", "contexts": ["c"]}) == {}
