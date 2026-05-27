from __future__ import annotations

from src.config import SPARSE_TOP_K
from src.indexing.bm25_index import BM25Index
from src.retrieval.dense import RetrievalResult


class SparseRetriever:
    def __init__(self, bm25_index: BM25Index | None = None, top_k: int = SPARSE_TOP_K):
        self.bm25_index = bm25_index or BM25Index()
        self.top_k = top_k
        self._chunk_content_cache: dict[str, str] = {}

    def set_chunk_content(self, chunk_map: dict[str, str]) -> None:
        """Cache chunk_id -> content for result building."""
        self._chunk_content_cache = chunk_map

    def retrieve(self, query: str) -> list[RetrievalResult]:
        ranked = self.bm25_index.query(query, top_k=self.top_k)

        results = []
        for chunk_id, score in ranked:
            content = self._chunk_content_cache.get(chunk_id, "")
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=content,
                score=score,
                metadata={"retrieval_type": "sparse"},
            ))

        return results
