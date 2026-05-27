from __future__ import annotations

from src.config import RRF_DENSE_WEIGHT, RRF_SPARSE_WEIGHT, FUSION_TOP_K
from src.retrieval.dense import RetrievalResult


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    dense_weight: float = RRF_DENSE_WEIGHT,
    sparse_weight: float = RRF_SPARSE_WEIGHT,
    top_k: int = FUSION_TOP_K,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """
    Combine dense and sparse results using Reciprocal Rank Fusion.
    RRF score = sum(weight / (k + rank)) across all lists where the doc appears.
    """
    scores: dict[str, float] = {}
    content_map: dict[str, str] = {}
    metadata_map: dict[str, dict] = {}

    for rank, result in enumerate(dense_results):
        rrf_score = dense_weight / (rrf_k + rank + 1)
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + rrf_score
        content_map[result.chunk_id] = result.content
        metadata_map[result.chunk_id] = result.metadata

    for rank, result in enumerate(sparse_results):
        rrf_score = sparse_weight / (rrf_k + rank + 1)
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + rrf_score
        if result.chunk_id not in content_map:
            content_map[result.chunk_id] = result.content
            metadata_map[result.chunk_id] = result.metadata

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

    return [
        RetrievalResult(
            chunk_id=cid,
            content=content_map[cid],
            score=scores[cid],
            metadata={**metadata_map[cid], "retrieval_type": "hybrid_rrf"},
        )
        for cid in sorted_ids
    ]
