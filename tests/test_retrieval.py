from src.retrieval.dense import RetrievalResult
from src.retrieval.fusion import reciprocal_rank_fusion


def _make_result(chunk_id: str, content: str = "", score: float = 0.0) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, content=content, score=score, metadata={})


class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        dense = [_make_result("a", score=0.9), _make_result("b", score=0.8)]
        sparse = [_make_result("b", score=5.0), _make_result("c", score=3.0)]

        fused = reciprocal_rank_fusion(dense, sparse, top_k=10)

        # "b" appears in both lists, should rank highest
        assert fused[0].chunk_id == "b"
        assert len(fused) == 3

    def test_respects_top_k(self):
        dense = [_make_result(f"d{i}") for i in range(10)]
        sparse = [_make_result(f"s{i}") for i in range(10)]

        fused = reciprocal_rank_fusion(dense, sparse, top_k=5)
        assert len(fused) == 5

    def test_empty_inputs(self):
        fused = reciprocal_rank_fusion([], [], top_k=5)
        assert fused == []

    def test_single_list_only(self):
        dense = [_make_result("a"), _make_result("b")]
        fused = reciprocal_rank_fusion(dense, [], top_k=5)
        assert len(fused) == 2

    def test_weight_affects_ranking(self):
        # dense has "a" first, sparse has "b" first
        dense = [_make_result("a"), _make_result("b")]
        sparse = [_make_result("b"), _make_result("a")]

        # Heavy dense weight — "a" should win
        fused_dense_heavy = reciprocal_rank_fusion(
            dense, sparse, dense_weight=0.9, sparse_weight=0.1, top_k=5
        )
        assert fused_dense_heavy[0].chunk_id == "a"

        # Heavy sparse weight — "b" should win
        fused_sparse_heavy = reciprocal_rank_fusion(
            dense, sparse, dense_weight=0.1, sparse_weight=0.9, top_k=5
        )
        assert fused_sparse_heavy[0].chunk_id == "b"

    def test_scores_are_positive(self):
        dense = [_make_result("a")]
        sparse = [_make_result("b")]
        fused = reciprocal_rank_fusion(dense, sparse, top_k=5)
        assert all(r.score > 0 for r in fused)

    def test_metadata_preserved(self):
        dense = [RetrievalResult(chunk_id="a", content="text", score=0.9, metadata={"source": "vec"})]
        sparse = []
        fused = reciprocal_rank_fusion(dense, sparse, top_k=5)
        assert fused[0].metadata["source"] == "vec"
        assert fused[0].metadata["retrieval_type"] == "hybrid_rrf"

    def test_content_preserved(self):
        dense = [_make_result("a", content="hello world")]
        sparse = [_make_result("a", content="should not overwrite")]
        fused = reciprocal_rank_fusion(dense, sparse, top_k=5)
        assert fused[0].content == "hello world"

    def test_duplicate_boosted(self):
        """Chunk appearing in both lists should score higher than either alone."""
        dense = [_make_result("shared"), _make_result("dense_only")]
        sparse = [_make_result("shared"), _make_result("sparse_only")]

        fused = reciprocal_rank_fusion(
            dense, sparse, dense_weight=0.5, sparse_weight=0.5, top_k=10
        )

        shared_score = next(r.score for r in fused if r.chunk_id == "shared")
        other_scores = [r.score for r in fused if r.chunk_id != "shared"]
        assert all(shared_score > s for s in other_scores)
