from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from services.eval_consumer.config import scorer_config, settings

logger = logging.getLogger(__name__)

_SCORE_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


class LlmJudgeScorer:
    scorer_name = "llm_judge"

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url
        self._model = model
        self._scale_min = scorer_config.llm_judge.scale_min
        self._scale_max = scorer_config.llm_judge.scale_max
        self._timeout = settings.ollama_timeout_seconds

    def _build_prompt(self, trace: dict[str, Any]) -> str:
        contexts = trace.get("contexts") or []
        joined = "\n\n".join(f"[{index}] {chunk}" for index, chunk in enumerate(contexts))
        return (
            "You are a strict evaluator of question answering quality. "
            "Rate how well the answer responds to the question and is grounded "
            f"in the context, on an integer scale from {self._scale_min} to {self._scale_max}, "
            f"where {self._scale_min} is worst and {self._scale_max} is best. "
            'Respond ONLY with JSON of the form {"rating": <int>, "reason": "<short>"}.\n\n'
            f"Context:\n{joined}\n\n"
            f"Question: {trace.get('question', '')}\n"
            f"Answer: {trace.get('answer', '')}\n"
        )

    def _normalize(self, rating: float) -> float:
        span = self._scale_max - self._scale_min
        if span <= 0:
            return 0.0
        clamped = max(self._scale_min, min(self._scale_max, rating))
        return (clamped - self._scale_min) / span

    def _parse_rating(self, content: str) -> float | None:
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and "rating" in payload:
                return float(payload["rating"])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        match = _SCORE_PATTERN.search(content)
        if match is not None:
            return float(match.group())
        return None

    async def score(self, trace: dict[str, Any]) -> dict[str, float]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": self._build_prompt(trace),
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": scorer_config.temperature},
                },
            )
            response.raise_for_status()
            content = str(response.json().get("response", "")).strip()
        rating = self._parse_rating(content)
        if rating is None:
            logger.warning("llm_judge could not parse rating for trace_id=%s", trace.get("trace_id"))
            return {}
        return {"llm_judge": self._normalize(rating)}
