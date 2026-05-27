import json
import tempfile
from pathlib import Path

from src.evaluation.dataset import EvalDataset, EvalCase
from src.evaluation.metrics import EvalMetrics, EvalResult
from src.evaluation.run_eval import _compute_summary
from src.retrieval.dense import RetrievalResult
from src.generation.citations import CitationResult
from src.generation.pipeline import GenerationResult
from src.generation.confidence import ConfidenceScore


def _make_case(qid: str = "q1", qtype: str = "basic") -> EvalCase:
    return EvalCase(
        question_id=qid,
        question="What is X?",
        question_type=qtype,
        source_types=["confluence"],
        expected_doc_ids=["doc1"],
        gold_answer="X is Y.",
        answer_facts=["X is Y"],
    )


def _make_gen_result(
    answer: str = "X is Y [1].",
    chunk_doc_ids: list[str] | None = None,
    citation_supported: list[bool] | None = None,
) -> GenerationResult:
    chunk_doc_ids = chunk_doc_ids or ["doc1"]
    citation_supported = citation_supported or [True]

    chunks = [
        RetrievalResult(
            chunk_id=f"c{i}",
            content=f"Content about doc {did}",
            score=0.8,
            metadata={"doc_id": did, "title": "Test", "source_type": "txt"},
        )
        for i, did in enumerate(chunk_doc_ids)
    ]

    citations = [
        CitationResult(
            citation_id=i + 1,
            claim=f"Claim {i+1}",
            source_chunk_id=f"c{i}",
            supported=sup,
        )
        for i, sup in enumerate(citation_supported)
    ]

    return GenerationResult(
        question="What is X?",
        answer=answer,
        context_chunks=chunks,
        citations=citations,
        confidence=ConfidenceScore(0.8, 1.0, 0.7, 0.8),
        is_confident=True,
    )


class TestEvalDataset:
    def test_load_enterprise_bench(self):
        ds = EvalDataset.from_enterprise_bench(max_cases=10)
        assert len(ds) == 10
        assert all(isinstance(c, EvalCase) for c in ds)
        assert ds.cases[0].question_id

    def test_filter_by_type(self):
        ds = EvalDataset.from_enterprise_bench(max_cases=100)
        basic = ds.filter_by_type("basic")
        assert all(c.question_type == "basic" for c in basic)

    def test_save_load_jsonl(self):
        cases = [_make_case("q1"), _make_case("q2")]
        ds = EvalDataset(cases)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            ds.save_jsonl(f.name)
            loaded = EvalDataset.from_jsonl(f.name)

        assert len(loaded) == 2
        assert loaded.cases[0].question_id == "q1"


class TestRetrievalMetrics:
    def setup_method(self):
        self.metrics = EvalMetrics()

    def test_retrieval_relevance_perfect(self):
        chunks = [
            RetrievalResult("c1", "text", 0.9, {"doc_id": "doc1"}),
            RetrievalResult("c2", "text", 0.8, {"doc_id": "doc1"}),
        ]
        rel = self.metrics._retrieval_relevance(chunks, ["doc1"])
        assert rel == 1.0

    def test_retrieval_relevance_partial(self):
        chunks = [
            RetrievalResult("c1", "text", 0.9, {"doc_id": "doc1"}),
            RetrievalResult("c2", "text", 0.8, {"doc_id": "doc_other"}),
        ]
        rel = self.metrics._retrieval_relevance(chunks, ["doc1"])
        assert rel == 0.5

    def test_retrieval_relevance_empty(self):
        assert self.metrics._retrieval_relevance([], ["doc1"]) == 0.0

    def test_retrieval_recall_perfect(self):
        chunks = [
            RetrievalResult("c1", "text", 0.9, {"doc_id": "doc1"}),
            RetrievalResult("c2", "text", 0.8, {"doc_id": "doc2"}),
        ]
        recall = self.metrics._retrieval_recall(chunks, ["doc1", "doc2"])
        assert recall == 1.0

    def test_retrieval_recall_partial(self):
        chunks = [RetrievalResult("c1", "text", 0.9, {"doc_id": "doc1"})]
        recall = self.metrics._retrieval_recall(chunks, ["doc1", "doc2"])
        assert recall == 0.5

    def test_retrieval_recall_empty_expected(self):
        chunks = [RetrievalResult("c1", "text", 0.9, {"doc_id": "doc1"})]
        assert self.metrics._retrieval_recall(chunks, []) == 0.0

    def test_citation_accuracy_all_supported(self):
        citations = [
            CitationResult(1, "claim", "c1", True),
            CitationResult(2, "claim", "c2", True),
        ]
        assert self.metrics._citation_accuracy(citations) == 1.0

    def test_citation_accuracy_mixed(self):
        citations = [
            CitationResult(1, "claim", "c1", True),
            CitationResult(2, "claim", "c2", False),
        ]
        assert self.metrics._citation_accuracy(citations) == 0.5

    def test_citation_accuracy_empty(self):
        assert self.metrics._citation_accuracy([]) == 0.0


class TestComputeSummary:
    def test_summary_structure(self):
        results = [
            EvalResult("q1", "basic", 0.8, 0.9, 0.5, 1.0, 0.7),
            EvalResult("q2", "basic", 0.6, 0.7, 0.3, 0.5, 0.5),
            EvalResult("q3", "multi_hop", 0.9, 0.8, 0.6, 1.0, 0.8),
        ]
        summary = _compute_summary(results, 10.0, [])

        assert summary["total"] == 3
        assert summary["errors"] == 0
        assert "overall" in summary
        assert "by_question_type" in summary
        assert "basic" in summary["by_question_type"]
        assert "multi_hop" in summary["by_question_type"]
        assert summary["by_question_type"]["basic"]["count"] == 2

    def test_summary_empty(self):
        summary = _compute_summary([], 1.0, [{"error": "fail"}])
        assert summary["total"] == 0
        assert summary["errors"] == 1

    def test_summary_averages(self):
        results = [
            EvalResult("q1", "basic", 0.8, 1.0, 0.6, 1.0, 1.0),
            EvalResult("q2", "basic", 0.4, 0.6, 0.4, 0.5, 0.5),
        ]
        summary = _compute_summary(results, 5.0, [])
        assert summary["overall"]["avg_correctness"] == 0.6
        assert summary["overall"]["avg_faithfulness"] == 0.8
