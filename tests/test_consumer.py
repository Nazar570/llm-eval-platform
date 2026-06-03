from __future__ import annotations

from typing import Any, cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Evaluation, Trace
from services.eval_consumer import consumer as consumer_module
from services.eval_consumer.consumer import Scorer, _persist_results, _persist_trace, _run_scorer


class _FakeSession:
    def __init__(self, existing: Any = None) -> None:
        self.added: list[Any] = []
        self.committed = False
        self.rolled_back = False
        self._existing = existing
        self._next_id = 1

    async def scalar(self, _statement: Any) -> Any:
        return self._existing

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        for obj in self.added:
            if isinstance(obj, Trace) and obj.id is None:
                obj.id = self._next_id
                self._next_id += 1

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class _FakeScorer:
    def __init__(self, name: str, result: dict[str, float] | None = None, raises: bool = False) -> None:
        self.scorer_name = name
        self._result = result or {}
        self._raises = raises

    async def score(self, _trace: dict[str, Any]) -> dict[str, float]:
        if self._raises:
            raise RuntimeError("scorer boom")
        return self._result


async def test_run_scorer_returns_result_on_success() -> None:
    scorer = _FakeScorer("llm_judge", {"llm_judge": 0.8})
    assert await _run_scorer(scorer, {"trace_id": "t"}) == {"llm_judge": 0.8}


async def test_run_scorer_swallows_exception() -> None:
    scorer = _FakeScorer("ragas", raises=True)
    assert await _run_scorer(scorer, {"trace_id": "t"}) == {}


async def test_persist_trace_returns_existing_id() -> None:
    existing = Trace(id=42, trace_id="t", question="q", answer="a", model_name="m")
    db = _FakeSession(existing=existing)

    trace_pk = await _persist_trace(cast(AsyncSession, db), {"trace_id": "t", "question": "q", "answer": "a"})

    assert trace_pk == 42
    assert db.added == []


async def test_persist_trace_inserts_new_trace() -> None:
    db = _FakeSession(existing=None)
    trace = {
        "trace_id": "new",
        "question": "q",
        "answer": "a",
        "contexts": ["c1"],
        "model_name": "llama",
        "latency_ms": 12.5,
    }

    trace_pk = await _persist_trace(cast(AsyncSession, db), trace)

    assert trace_pk == 1
    assert len(db.added) == 1
    stored = db.added[0]
    assert isinstance(stored, Trace)
    assert stored.trace_id == "new"
    assert stored.contexts == ["c1"]


async def test_persist_results_writes_one_row_per_metric() -> None:
    db = _FakeSession()
    scorers = [_FakeScorer("ragas"), _FakeScorer("llm_judge")]
    results = [{"faithfulness": 0.9, "answer_relevancy": 0.7}, {"llm_judge": 0.6}]

    await _persist_results(cast(AsyncSession, db), trace_pk=5, scorers=cast("list[Scorer]", scorers), results=results)

    evaluations = [row for row in db.added if isinstance(row, Evaluation)]
    assert len(evaluations) == 3
    pairs = {(row.scorer, row.metric, row.score) for row in evaluations}
    assert ("ragas", "faithfulness", 0.9) in pairs
    assert ("llm_judge", "llm_judge", 0.6) in pairs
    assert all(row.trace_pk == 5 for row in evaluations)


async def test_handle_trace_persists_and_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(existing=None)
    monkeypatch.setattr(consumer_module, "AsyncSessionLocal", lambda: session)

    scorers = [_FakeScorer("ragas", {"faithfulness": 0.9}), _FakeScorer("llm_judge", {"llm_judge": 0.8})]
    trace = {"trace_id": "abc", "question": "q", "answer": "a", "model_name": "m"}

    await consumer_module._handle_trace(trace, cast("list[Scorer]", scorers))

    assert session.committed is True
    assert any(isinstance(row, Trace) for row in session.added)
    evaluations = [row for row in session.added if isinstance(row, Evaluation)]
    assert {row.metric for row in evaluations} == {"faithfulness", "llm_judge"}
