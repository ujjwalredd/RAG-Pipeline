from __future__ import annotations

from dataclasses import dataclass, field

from src.retrieval.dense import RetrievalResult
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import GroundedGenerator
from src.generation.citations import CitationVerifier, CitationResult
from src.generation.confidence import ConfidenceScorer, ConfidenceScore

LOW_CONFIDENCE_THRESHOLD = 0.3


@dataclass
class GenerationResult:
    question: str
    answer: str
    context_chunks: list[RetrievalResult]
    citations: list[CitationResult]
    confidence: ConfidenceScore
    is_confident: bool = True
    fallback_message: str = ""


class GenerationPipeline:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
        verify_citations: bool = True,
    ):
        self.retriever = retriever or HybridRetriever()
        self.generator = GroundedGenerator()
        self.verifier = CitationVerifier()
        self.scorer = ConfidenceScorer()
        self.confidence_threshold = confidence_threshold
        self.verify_citations = verify_citations

    def ask(self, question: str) -> GenerationResult:
        context_chunks = self.retriever.retrieve(question)

        retrieval_scores = [c.score for c in context_chunks]
        avg_retrieval = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0

        if avg_retrieval < self.confidence_threshold and len(context_chunks) > 0:
            return self._low_confidence_response(question, context_chunks)

        if not context_chunks:
            return self._no_results_response(question)

        answer = self.generator.generate(question, context_chunks)

        if self.verify_citations:
            citations = self.verifier.verify(answer, context_chunks)
        else:
            citations = []

        confidence = self.scorer.score(question, answer, context_chunks, citations)

        if confidence.composite < self.confidence_threshold:
            return self._low_confidence_response(question, context_chunks, answer, citations)

        return GenerationResult(
            question=question,
            answer=answer,
            context_chunks=context_chunks,
            citations=citations,
            confidence=confidence,
            is_confident=True,
        )

    def _low_confidence_response(
        self,
        question: str,
        chunks: list[RetrievalResult],
        partial_answer: str = "",
        citations: list[CitationResult] | None = None,
    ) -> GenerationResult:
        citations = citations or []

        sources_found = []
        for i, chunk in enumerate(chunks[:3], 1):
            title = chunk.metadata.get("title", "Unknown")
            source_type = chunk.metadata.get("source_type", "unknown")
            sources_found.append(f"  [{i}] {title} ({source_type}) — relevance: {chunk.score:.3f}")

        fallback = (
            f"I could not find enough reliable information to fully answer this question.\n\n"
            f"What I found:\n" + "\n".join(sources_found) + "\n\n"
            f"These documents may be partially relevant but confidence is low. "
            f"Consider checking the source documents directly."
        )

        if partial_answer:
            fallback = (
                f"Partial answer (low confidence):\n{partial_answer}\n\n"
                f"---\n{fallback}"
            )

        confidence = self.scorer.score(question, partial_answer or "", chunks, citations)

        return GenerationResult(
            question=question,
            answer=fallback,
            context_chunks=chunks,
            citations=citations,
            confidence=confidence,
            is_confident=False,
            fallback_message=fallback,
        )

    def _no_results_response(self, question: str) -> GenerationResult:
        empty_confidence = ConfidenceScore(
            retrieval_confidence=0.0,
            citation_coverage=0.0,
            answer_completeness=0.0,
            composite=0.0,
        )
        return GenerationResult(
            question=question,
            answer="No relevant documents found in the knowledge base for this question.",
            context_chunks=[],
            citations=[],
            confidence=empty_confidence,
            is_confident=False,
            fallback_message="No relevant documents found.",
        )
