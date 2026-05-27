from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.config import OLLAMA_BASE_URL, GENERATION_MODEL
from src.retrieval.dense import RetrievalResult
from src.generation.citations import CitationResult
from src.generation.prompts import CONFIDENCE_PROMPT


@dataclass
class ConfidenceScore:
    retrieval_confidence: float
    citation_coverage: float
    answer_completeness: float
    composite: float


class ConfidenceScorer:
    def __init__(self, model: str = GENERATION_MODEL):
        self.model = model

    def score(
        self,
        question: str,
        answer: str,
        context_chunks: list[RetrievalResult],
        citation_results: list[CitationResult],
    ) -> ConfidenceScore:
        retrieval = self._retrieval_confidence(context_chunks)
        citation = self._citation_coverage(citation_results)
        completeness = self._answer_completeness(question, answer)

        composite = (0.4 * retrieval) + (0.35 * citation) + (0.25 * completeness)

        return ConfidenceScore(
            retrieval_confidence=round(retrieval, 3),
            citation_coverage=round(citation, 3),
            answer_completeness=round(completeness, 3),
            composite=round(composite, 3),
        )

    def _retrieval_confidence(self, chunks: list[RetrievalResult]) -> float:
        if not chunks:
            return 0.0
        scores = [c.score for c in chunks]
        # Normalize: reranker scores are 0-10, RRF scores are small floats
        max_score = max(scores)
        if max_score > 1.0:
            normalized = [s / 10.0 for s in scores]
        else:
            normalized = scores
        return sum(normalized) / len(normalized)

    def _citation_coverage(self, citations: list[CitationResult]) -> float:
        if not citations:
            return 0.0
        supported = sum(1 for c in citations if c.supported)
        return supported / len(citations)

    def _answer_completeness(self, question: str, answer: str) -> float:
        prompt = CONFIDENCE_PROMPT.format(question=question, answer=answer)

        try:
            resp = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 5},
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            text = resp.json()["response"].strip()
            score = float("".join(c for c in text if c.isdigit() or c == ".")[:4])
            return min(max(score / 10.0, 0.0), 1.0)
        except (httpx.HTTPError, ValueError, KeyError):
            return 0.0
