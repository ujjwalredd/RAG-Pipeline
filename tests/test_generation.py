from src.generation.citations import CitationVerifier, CitationResult
from src.generation.confidence import ConfidenceScorer, ConfidenceScore
from src.generation.pipeline import GenerationPipeline, GenerationResult
from src.retrieval.dense import RetrievalResult


def _make_result(chunk_id: str, content: str = "", score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, content=content, score=score,
        metadata={"title": "Test", "source_type": "txt"},
    )


class TestCitationExtraction:
    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_extract_single_citation(self):
        answer = "The limit is 10 MiB [1]."
        claims = self.verifier._extract_claims_with_citations(answer)
        assert len(claims) == 1
        assert claims[0] == (1, "The limit is 10 MiB.")

    def test_extract_multiple_citations(self):
        answer = "Fact A [1]. Fact B [2]. Fact C [1]."
        claims = self.verifier._extract_claims_with_citations(answer)
        assert len(claims) == 3

    def test_extract_multi_ref_sentence(self):
        answer = "Both sources agree on this [1] [2]."
        claims = self.verifier._extract_claims_with_citations(answer)
        assert len(claims) == 2
        ids = [c[0] for c in claims]
        assert 1 in ids and 2 in ids

    def test_no_citations(self):
        answer = "This has no citations at all."
        claims = self.verifier._extract_claims_with_citations(answer)
        assert len(claims) == 0

    def test_invalid_citation_id(self):
        chunks = [_make_result("c1", "some content")]
        results = self.verifier.verify(
            "Claim [5].", chunks
        )
        # Ollama not running so _check_support returns False,
        # but citation_id=5 with only 1 chunk → INVALID
        invalid = [r for r in results if r.source_chunk_id == "INVALID"]
        assert len(invalid) == 1


class TestConfidenceScoring:
    def setup_method(self):
        self.scorer = ConfidenceScorer()

    def test_retrieval_confidence_high(self):
        chunks = [_make_result("c1", score=0.9), _make_result("c2", score=0.85)]
        conf = self.scorer._retrieval_confidence(chunks)
        assert conf > 0.8

    def test_retrieval_confidence_empty(self):
        assert self.scorer._retrieval_confidence([]) == 0.0

    def test_citation_coverage_all_supported(self):
        citations = [
            CitationResult(1, "claim a", "c1", supported=True),
            CitationResult(2, "claim b", "c2", supported=True),
        ]
        assert self.scorer._citation_coverage(citations) == 1.0

    def test_citation_coverage_mixed(self):
        citations = [
            CitationResult(1, "claim a", "c1", supported=True),
            CitationResult(2, "claim b", "c2", supported=False),
        ]
        assert self.scorer._citation_coverage(citations) == 0.5

    def test_citation_coverage_empty(self):
        assert self.scorer._citation_coverage([]) == 0.0

    def test_composite_score_structure(self):
        chunks = [_make_result("c1", score=0.8)]
        citations = [CitationResult(1, "claim", "c1", supported=True)]
        score = self.scorer.score("question?", "answer [1].", chunks, citations)

        assert isinstance(score, ConfidenceScore)
        assert 0.0 <= score.composite <= 1.0
        assert 0.0 <= score.retrieval_confidence <= 1.0
        assert 0.0 <= score.citation_coverage <= 1.0


class TestNoResultsResponse:
    def test_no_results(self):
        pipeline = GenerationPipeline.__new__(GenerationPipeline)
        result = pipeline._no_results_response("What is X?")

        assert not result.is_confident
        assert result.confidence.composite == 0.0
        assert "No relevant documents" in result.answer
