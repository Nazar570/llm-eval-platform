from __future__ import annotations

from typing import Any

import pytest

from training import metrics
from training.metrics import (
    BATCH_EVAL_REGISTRY,
    DEFAULT_PUSHGATEWAY_URL,
    PUSH_JOB,
    push_batch_eval_metrics,
)


def _sample(name: str, model: str) -> float | None:
    return BATCH_EVAL_REGISTRY.get_sample_value(name, {"registered_model": model})


def _value(name: str, model: str) -> float:
    return _sample(name, model) or 0.0


def test_push_increments_counters_and_sets_gauge(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_push(gateway: str, job: str, registry: Any) -> None:
        captured["gateway"] = gateway
        captured["job"] = job
        captured["registry"] = registry

    monkeypatch.setattr(metrics, "push_to_gateway", _fake_push)
    model = "model-counters"
    before_scored = _value("llm_batch_eval_scored_traces_total", model)
    before_positive = _value("llm_batch_eval_positive_label_total", model)

    push_batch_eval_metrics(model, scored=10, positive=4)

    assert _value("llm_batch_eval_scored_traces_total", model) - before_scored == 10
    assert _value("llm_batch_eval_positive_label_total", model) - before_positive == 4
    gauge = _sample("llm_batch_eval_last_run_timestamp_seconds", model)
    assert gauge is not None
    assert gauge > 0
    assert captured["job"] == PUSH_JOB
    assert captured["registry"] is BATCH_EVAL_REGISTRY


def test_push_uses_env_gateway_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        metrics,
        "push_to_gateway",
        lambda gateway, job, registry: captured.update(gateway=gateway),
    )
    monkeypatch.setenv("PROMETHEUS_PUSHGATEWAY_URL", "http://custom-gateway:9091")

    push_batch_eval_metrics("model-env", scored=1, positive=0)

    assert captured["gateway"] == "http://custom-gateway:9091"


def test_push_defaults_gateway_url_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        metrics,
        "push_to_gateway",
        lambda gateway, job, registry: captured.update(gateway=gateway),
    )
    monkeypatch.delenv("PROMETHEUS_PUSHGATEWAY_URL", raising=False)

    push_batch_eval_metrics("model-default", scored=1, positive=0)

    assert captured["gateway"] == DEFAULT_PUSHGATEWAY_URL


def test_push_swallows_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(gateway: str, job: str, registry: Any) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr(metrics, "push_to_gateway", _raise)
    model = "model-error"
    before_scored = _value("llm_batch_eval_scored_traces_total", model)

    push_batch_eval_metrics(model, scored=3, positive=2)

    assert _value("llm_batch_eval_scored_traces_total", model) - before_scored == 3
