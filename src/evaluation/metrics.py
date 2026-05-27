from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.config import OLLAMA_BASE_URL, GENERATION_MODEL
from src.evaluation.dataset import EvalCase
from src.generation.pipeline import GenerationResult


CORRECTNESS_PROMPT = """Compare the generated answer against the gold (reference) answer.
Rate correctness on a scale of 0-10. Only respond with a single integer.

Question: {question}

Gold answer: {gold_answer}

Generated answer: {generated_answer}

Correctness score (0-10):"""

FAITHFULNESS_PROMPT = """Determine if the claim is fully supported by the provided context.
Respond with only "FAITHFUL" or "UNFAITHFUL".

Claim: {claim}

Context: {context}

Verdict:"""


@dataclass
class EvalResult:
    question_id: str
    question_type: str
    answer_correctness: float
    faithfulness: float
    retrieval_relevance: float
    citation_accuracy: float
    retrieval_recall: float


class EvalMetrics:
    def __init__(self, model: str = GENERATION_MODEL):
        self.model = model

    def evaluate(self, case: EvalCase, result: GenerationResult) -> EvalResult:
        correctness = self._answer_correctness(case.question, case.gold_answer, result.answer)
        faithfulness = self._faithfulness(result.answer, result.context_chunks)
        retrieval_rel = self._retrieval_relevance(result.context_chunks, case.expected_doc_ids)
        citation_acc = self._citation_accuracy(result.citations)
        retrieval_recall = self._retrieval_recall(result.context_chunks, case.expected_doc_ids)

        return EvalResult(
            question_id=case.question_id,
            question_type=case.question_type,
            answer_correctness=correctness,
            faithfulness=faithfulness,
            retrieval_relevance=retrieval_rel,
            citation_accuracy=citation_acc,
            retrieval_recall=retrieval_recall,
        )

    def _answer_correctness(self, question: str, gold: str, generated: str) -> float:
        prompt = CORRECTNESS_PROMPT.format(
            question=question,
            gold_answer=gold,
            generated_answer=generated,
        )
        return self._llm_score(prompt)

    def _faithfulness(self, answer: str, chunks: list) -> float:
        if not chunks:
            return 0.0

        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        if not sentences:
            return 0.0

        context = "\n".join(c.content[:500] for c in chunks[:5])
        faithful_count = 0

        for sentence in sentences:
            clean = re.sub(r"\[\d+\]", "", sentence).strip()
            if not clean or len(clean) < 10:
                faithful_count += 1
                continue

            prompt = FAITHFULNESS_PROMPT.format(claim=clean, context=context)
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
                if "FAITHFUL" in verdict and "UNFAITHFUL" not in verdict:
                    faithful_count += 1
            except (httpx.HTTPError, KeyError):
                pass

        return faithful_count / len(sentences)

    def _retrieval_relevance(self, chunks: list, expected_doc_ids: list[str]) -> float:
        """Precision: what fraction of retrieved chunks come from expected docs."""
        if not chunks:
            return 0.0
        relevant = sum(
            1 for c in chunks
            if c.metadata.get("doc_id") in expected_doc_ids
        )
        return relevant / len(chunks)

    def _retrieval_recall(self, chunks: list, expected_doc_ids: list[str]) -> float:
        """Recall: what fraction of expected docs appear in retrieved chunks."""
        if not expected_doc_ids:
            return 0.0
        retrieved_doc_ids = {c.metadata.get("doc_id") for c in chunks}
        found = sum(1 for doc_id in expected_doc_ids if doc_id in retrieved_doc_ids)
        return found / len(expected_doc_ids)

    def _citation_accuracy(self, citations: list) -> float:
        if not citations:
            return 0.0
        supported = sum(1 for c in citations if c.supported)
        return supported / len(citations)

    def _llm_score(self, prompt: str) -> float:
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
