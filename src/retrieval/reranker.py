from __future__ import annotations

import httpx

from src.config import OLLAMA_BASE_URL, GENERATION_MODEL, RERANK_TOP_K
from src.retrieval.dense import RetrievalResult

RERANK_PROMPT = """Rate the relevance of this text chunk to the question on a scale of 0-10.
Only respond with a single integer number, nothing else.

Question: {question}

Text chunk:
{chunk}

Relevance score (0-10):"""


class Reranker:
    def __init__(
        self,
        model: str = GENERATION_MODEL,
        top_k: int = RERANK_TOP_K,
    ):
        self.model = model
        self.top_k = top_k

    def rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        """Score each candidate's relevance via LLM-as-judge, keep top_k."""
        scored = []
        for candidate in candidates:
            relevance = self._score_relevance(query, candidate.content)
            scored.append((candidate, relevance))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                chunk_id=c.chunk_id,
                content=c.content,
                score=relevance_score,
                metadata={**c.metadata, "rerank_score": relevance_score},
            )
            for c, relevance_score in scored[: self.top_k]
        ]

    def _score_relevance(self, question: str, chunk: str) -> float:
        prompt = RERANK_PROMPT.format(question=question, chunk=chunk[:1000])

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
            return min(max(score, 0.0), 10.0)
        except (httpx.HTTPError, ValueError, KeyError):
            return 0.0
