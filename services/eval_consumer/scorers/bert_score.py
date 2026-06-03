from __future__ import annotations

import asyncio
import logging
from typing import Any

import torch
from bert_score import BERTScorer

from services.eval_consumer.config import scorer_config

logger = logging.getLogger(__name__)


class BertScoreScorer:
    scorer_name = "bert_score"

    def __init__(self) -> None:
        self._model_type = scorer_config.bert_score.model_type
        self._lang = scorer_config.bert_score.lang
        self._rescale_with_baseline = scorer_config.bert_score.rescale_with_baseline
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._scorer: BERTScorer | None = None

    def _get_scorer(self) -> BERTScorer:
        if self._scorer is None:
            self._scorer = BERTScorer(
                model_type=self._model_type,
                lang=self._lang,
                rescale_with_baseline=self._rescale_with_baseline,
                device=self._device,
            )
        return self._scorer

    def _evaluate(self, candidate: str, reference: str) -> dict[str, float]:
        scorer = self._get_scorer()
        precision, recall, f1 = scorer.score([candidate], [reference])
        return {
            "bert_score_precision": float(precision.mean().item()),
            "bert_score_recall": float(recall.mean().item()),
            "bert_score": float(f1.mean().item()),
        }

    async def score(self, trace: dict[str, Any]) -> dict[str, float]:
        reference = trace.get("ground_truth")
        if not reference or not str(reference).strip():
            return {}
        candidate = str(trace.get("answer", "")).strip()
        if not candidate:
            return {}
        return await asyncio.to_thread(self._evaluate, candidate, str(reference))
