from __future__ import annotations

from typing import Any

import pytest

from services.eval_consumer.scorers import llm_judge as llm_judge_module
from services.eval_consumer.scorers.llm_judge import LlmJudgeScorer


@pytest.fixture
def scorer() -> LlmJudgeScorer:
    return LlmJudgeScorer(base_url="http://ollama:11434", model="llama3.2:3b")


def test_normalize_maps_scale_to_unit_interval(scorer: LlmJudgeScorer) -> None:
    assert scorer._normalize(scorer._scale_min) == 0.0
    assert scorer._normalize(scorer._scale_max) == 1.0
    midpoint = (scorer._scale_min + scorer._scale_max) / 2
    assert scorer._normalize(midpoint) == pytest.approx(0.5)


def test_normalize_clamps_out_of_range(scorer: LlmJudgeScorer) -> None:
    assert scorer._normalize(scorer._scale_max + 10) == 1.0
    assert scorer._normalize(scorer._scale_min - 10) == 0.0


def test_parse_rating_reads_json(scorer: LlmJudgeScorer) -> None:
    assert scorer._parse_rating('{"rating": 4, "reason": "ok"}') == 4.0


def test_parse_rating_falls_back_to_first_number(scorer: LlmJudgeScorer) -> None:
    assert scorer._parse_rating("the rating is 3 out of 5") == 3.0


def test_parse_rating_returns_none_without_number(scorer: LlmJudgeScorer) -> None:
    assert scorer._parse_rating("no score here") is None


def test_build_prompt_includes_context_and_answer(scorer: LlmJudgeScorer) -> None:
    prompt = scorer._build_prompt(
        {"question": "What is RAG?", "answer": "Retrieval augmented generation.", "contexts": ["doc a", "doc b"]}
    )
    assert "What is RAG?" in prompt
    assert "Retrieval augmented generation." in prompt
    assert "[0] doc a" in prompt
    assert "[1] doc b" in prompt


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def post(self, *_args: Any, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(self._payload)


async def test_score_returns_normalized_rating(monkeypatch: pytest.MonkeyPatch, scorer: LlmJudgeScorer) -> None:
    payload = {"response": '{"rating": 5, "reason": "great"}'}
    monkeypatch.setattr(llm_judge_module.httpx, "AsyncClient", lambda *a, **k: _FakeClient(payload))

    result = await scorer.score({"trace_id": "t1", "question": "q", "answer": "a", "contexts": []})

    assert result == {"llm_judge": 1.0}


async def test_score_returns_empty_when_unparsable(monkeypatch: pytest.MonkeyPatch, scorer: LlmJudgeScorer) -> None:
    payload = {"response": "totally unparsable"}
    monkeypatch.setattr(llm_judge_module.httpx, "AsyncClient", lambda *a, **k: _FakeClient(payload))

    result = await scorer.score({"trace_id": "t1", "question": "q", "answer": "a", "contexts": []})

    assert result == {}
