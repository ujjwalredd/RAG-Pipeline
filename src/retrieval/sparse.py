from __future__ import annotations

from src.config import SPARSE_TOP_K
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import VectorStore
from src.retrieval.dense import RetrievalResult


class SparseRetriever:
    def __init__(
        self,
        bm25_index: BM25Index | None = None,
        vector_store: VectorStore | None = None,
        top_k: int = SPARSE_TOP_K,
    ):
        self.bm25_index = bm25_index or BM25Index()
        self.vector_store = vector_store
        self.top_k = top_k
        self._chunk_content_cache: dict[str, str] = {}

    def set_chunk_content(self, chunk_map: dict[str, str]) -> None:
        self._chunk_content_cache = chunk_map

    def retrieve(self, query: str) -> list[RetrievalResult]:
        ranked = self.bm25_index.query(query, top_k=self.top_k)

        results = []
        for chunk_id, score in ranked:
            content = self._chunk_content_cache.get(chunk_id, "")
            metadata = {"retrieval_type": "sparse"}

            # Fetch content from ChromaDB if not cached
            if not content and self.vector_store:
                try:
                    got = self.vector_store.collection.get(
                        ids=[chunk_id], include=["documents", "metadatas"]
                    )
                    if got["documents"]:
                        content = got["documents"][0]
                    if got["metadatas"]:
                        metadata.update(got["metadatas"][0])
                except Exception:
                    pass

            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=content,
                score=score,
                metadata=metadata,
            ))

        return results
