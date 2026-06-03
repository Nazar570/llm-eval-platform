from __future__ import annotations

import asyncio
import logging
from typing import Any

from datasets import Dataset, Features, Sequence, Value
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_recall, faithfulness

from services.eval_consumer.config import scorer_config

logger = logging.getLogger(__name__)

_METRIC_REGISTRY = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_recall": context_recall,
}

_GROUND_TRUTH_REQUIRED = {"context_recall"}


class RagasScorer:
    scorer_name = "ragas"

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url
        self._model = model
        self._llm = LangchainLLMWrapper(
            ChatOllama(
                base_url=base_url,
                model=model,
                temperature=scorer_config.temperature,
            )
        )
        self._embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=scorer_config.embedding_model))
        self._configured_metrics = [name for name in scorer_config.ragas.metrics if name in _METRIC_REGISTRY]

    def _select_metrics(self, has_ground_truth: bool) -> list[str]:
        if has_ground_truth:
            return list(self._configured_metrics)
        return [name for name in self._configured_metrics if name not in _GROUND_TRUTH_REQUIRED]

    def build_dataset(self, trace: dict[str, Any], ground_truth: str) -> Dataset:
        contexts = trace.get("contexts") or []
        normalized_contexts = [str(chunk) for chunk in contexts if str(chunk).strip()]

        payload = {
            "question": [str(trace.get("question", ""))],
            "answer": [str(trace.get("answer", ""))],
            "contexts": [normalized_contexts],
            "ground_truth": [str(ground_truth)],
        }

        features = Features(
            {
                "question": Value("string"),
                "answer": Value("string"),
                "contexts": Sequence(Value("string")),
                "ground_truth": Value("string"),
            }
        )

        return Dataset.from_dict(payload, features=features)

    def _evaluate(self, trace: dict[str, Any]) -> dict[str, float]:
        raw_ground_truth = trace.get("ground_truth")
        has_ground_truth = bool(raw_ground_truth and str(raw_ground_truth).strip())
        selected = self._select_metrics(has_ground_truth)
        if not selected:
            return {}
        metrics = [_METRIC_REGISTRY[name] for name in selected]
        dataset = self.build_dataset(trace, str(raw_ground_truth or ""))
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._llm,
            embeddings=self._embeddings,
            raise_exceptions=False,
        )
        row = result.to_pandas().iloc[0].to_dict()
        cleaned: dict[str, float] = {}
        for name in selected:
            value = row.get(name)
            if value is None:
                continue
            numeric = float(value)
            if numeric != numeric:
                continue
            cleaned[name] = numeric
        return cleaned

    async def score(self, trace: dict[str, Any]) -> dict[str, float]:
        return await asyncio.to_thread(self._evaluate, trace)
