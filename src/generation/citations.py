from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from src.config import OLLAMA_BASE_URL, GENERATION_MODEL
from src.retrieval.dense import RetrievalResult
from src.generation.prompts import CITATION_VERIFY_PROMPT


@dataclass
class CitationResult:
    citation_id: int
    claim: str
    source_chunk_id: str
    supported: bool


class CitationVerifier:
    def __init__(self, model: str = GENERATION_MODEL):
        self.model = model

    def verify(
        self, answer: str, context_chunks: list[RetrievalResult]
    ) -> list[CitationResult]:
        claims = self._extract_claims_with_citations(answer)
        results = []

        for citation_id, claim in claims:
            if citation_id < 1 or citation_id > len(context_chunks):
                results.append(CitationResult(
                    citation_id=citation_id,
                    claim=claim,
                    source_chunk_id="INVALID",
                    supported=False,
                ))
                continue

            chunk = context_chunks[citation_id - 1]
            supported = self._check_support(claim, chunk.content)
            results.append(CitationResult(
                citation_id=citation_id,
                claim=claim,
                source_chunk_id=chunk.chunk_id,
                supported=supported,
            ))

        return results

    def _extract_claims_with_citations(self, answer: str) -> list[tuple[int, str]]:
        """Extract (citation_number, sentence_containing_citation) pairs."""
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        claims = []

        for sentence in sentences:
            refs = re.findall(r"\[(\d+)\]", sentence)
            clean_sentence = re.sub(r"\s*\[\d+\]", "", sentence).strip()
            if not clean_sentence:
                continue
            for ref in refs:
                claims.append((int(ref), clean_sentence))

        return claims

    def _check_support(self, claim: str, source_text: str) -> bool:
        prompt = CITATION_VERIFY_PROMPT.format(
            claim=claim,
            source=source_text[:1500],
        )

        try:
            resp = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 10},
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            verdict = resp.json()["response"].strip().upper()
            return "SUPPORTED" in verdict
        except (httpx.HTTPError, KeyError):
            return False
