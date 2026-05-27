from __future__ import annotations

from dataclasses import dataclass

from src.config import DENSE_TOP_K
from src.indexing.embeddings import embed_single
from src.indexing.vector_store import VectorStore


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    score: float
    metadata: dict


class DenseRetriever:
    def __init__(self, vector_store: VectorStore | None = None, top_k: int = DENSE_TOP_K):
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievalResult]:
        query_embedding = embed_single(query)
        raw = self.vector_store.query(query_embedding, top_k=self.top_k)

        results = []
        ids = raw["ids"][0]
        docs = raw["documents"][0]
        distances = raw["distances"][0]
        metadatas = raw["metadatas"][0]

        for chunk_id, content, distance, meta in zip(ids, docs, distances, metadatas):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            score = 1.0 - (distance / 2.0)
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=content,
                score=score,
                metadata=meta,
            ))

        return results
