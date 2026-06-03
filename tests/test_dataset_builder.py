from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from training import dataset_builder as builder


def test_build_example_shapes_text_and_label() -> None:
    example = builder.build_example("What is X?", "X is Y.", label=1, source="labelstudio")
    assert example["text"] == "What is X?\n\nX is Y."
    assert example["label"] == 1
    assert example["source"] == "labelstudio"
    assert example["detail"] == {}


def test_vote_label_positive_when_enough_positive_votes() -> None:
    label, detail = builder._vote_label({"answer_relevancy": 0.9, "faithfulness": 0.9})
    assert label == 1
    assert detail["pos_votes"] == 2


def test_vote_label_negative_when_enough_negative_votes() -> None:
    label, detail = builder._vote_label({"answer_relevancy": 0.2, "faithfulness": 0.3})
    assert label == 0
    assert detail["neg_votes"] == 2


def test_vote_label_abstains_below_min_votes() -> None:
    label, _ = builder._vote_label({"answer_relevancy": 0.9})
    assert label is None


def test_deduplicate_prefers_label_studio_source() -> None:
    weak = builder.build_example("q", "a", label=0, source="weak-supervision")
    human = builder.build_example("q", "a", label=1, source="labelstudio")
    deduped = builder.deduplicate_examples([weak, human])
    assert len(deduped) == 1
    assert deduped[0]["source"] == "labelstudio"


def test_write_dataset_emits_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "nested" / "eval_dataset.jsonl"
    monkeypatch.setattr(builder.training_runtime, "dataset_output_path", str(out))

    examples = [builder.build_example("q1", "a1", 1, "labelstudio"), builder.build_example("q2", "a2", 0, "weak")]
    written = builder.write_dataset(examples)

    lines = written.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "q1\n\na1"


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def get(self, *_args: Any, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(self._payload)


async def test_fetch_label_studio_annotations_maps_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [
        {
            "data": {"question": "What is RAG?", "answer": "Retrieval augmented generation."},
            "annotations": [{"result": [{"value": {"choices": ["good"]}}]}],
        },
        {
            "data": {"question": "What is X?", "answer": "Wrong."},
            "annotations": [{"result": [{"value": {"choices": ["bad"]}}]}],
        },
        {"data": {"question": "no answer"}, "annotations": []},
    ]
    monkeypatch.setattr(builder.httpx, "AsyncClient", lambda *a, **k: _FakeClient(payload))

    examples = await builder.fetch_label_studio_annotations()

    assert len(examples) == 2
    assert examples[0]["label"] == 1
    assert examples[1]["label"] == 0


async def test_build_dataset_merges_and_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "eval_dataset.jsonl"
    monkeypatch.setattr(builder.training_runtime, "dataset_output_path", str(out))

    human = [builder.build_example("shared", "answer", 1, "labelstudio")]
    weak = [
        builder.build_example("shared", "answer", 0, "weak-supervision"),
        builder.build_example("only-weak", "answer", 0, "weak-supervision"),
    ]

    async def _fake_human() -> list[dict[str, Any]]:
        return human

    async def _fake_weak() -> list[dict[str, Any]]:
        return weak

    monkeypatch.setattr(builder, "fetch_label_studio_annotations", _fake_human)
    monkeypatch.setattr(builder, "fetch_stored_evaluations", _fake_weak)

    count = await builder.build_dataset()

    assert count == 2
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    shared = next(row for row in rows if row["text"] == "shared\n\nanswer")
    assert shared["source"] == "labelstudio"
