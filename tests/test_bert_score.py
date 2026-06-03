from __future__ import annotations

from typing import Any

import pytest

from services.eval_consumer.scorers.bert_score import BertScoreScorer


@pytest.fixture
def scorer() -> BertScoreScorer:
    return BertScoreScorer()


async def test_score_skips_without_ground_truth(scorer: BertScoreScorer) -> None:
    result = await scorer.score({"answer": "an answer", "ground_truth": None})
    assert result == {}


async def test_score_skips_with_blank_ground_truth(scorer: BertScoreScorer) -> None:
    result = await scorer.score({"answer": "an answer", "ground_truth": "   "})
    assert result == {}


async def test_score_skips_with_blank_answer(scorer: BertScoreScorer) -> None:
    result = await scorer.score({"answer": "  ", "ground_truth": "the reference"})
    assert result == {}


async def test_score_delegates_to_evaluate(monkeypatch: pytest.MonkeyPatch, scorer: BertScoreScorer) -> None:
    captured: dict[str, Any] = {}

    def _fake_evaluate(candidate: str, reference: str) -> dict[str, float]:
        captured["candidate"] = candidate
        captured["reference"] = reference
        return {"bert_score": 0.91, "bert_score_precision": 0.9, "bert_score_recall": 0.92}

    monkeypatch.setattr(scorer, "_evaluate", _fake_evaluate)

    result = await scorer.score({"answer": "  Berlin is the capital. ", "ground_truth": " Berlin "})

    assert result["bert_score"] == 0.91
    assert captured == {"candidate": "Berlin is the capital.", "reference": " Berlin "}
