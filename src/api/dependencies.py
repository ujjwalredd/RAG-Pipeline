from __future__ import annotations

from functools import lru_cache

from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.generation.pipeline import GenerationPipeline


class AppState:
    """Shared application state for API dependencies."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()
        self._pipeline: GenerationPipeline | None = None

    @property
    def pipeline(self) -> GenerationPipeline:
        if self._pipeline is None:
            self._load_bm25()
            retriever = HybridRetriever(
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
            )
            self._pipeline = GenerationPipeline(retriever=retriever)
        return self._pipeline

    def _load_bm25(self) -> None:
        try:
            self.bm25_index.load()
        except FileNotFoundError:
            pass

    def reload(self) -> None:
        """Force reload after new ingestion."""
        self._pipeline = None


@lru_cache(maxsize=1)
def get_app_state() -> AppState:
    return AppState()
