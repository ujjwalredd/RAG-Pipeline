from __future__ import annotations

import numpy as np

from src.config import CHUNK_SIZE, SEMANTIC_SIMILARITY_THRESHOLD
from src.models import Chunk, Document
from src.chunking.base import BaseChunker


class SemanticChunker(BaseChunker):
    """Split on topic boundaries using embedding similarity between sentences."""

    strategy_name = "semantic"

    def __init__(
        self,
        embed_fn=None,
        similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
        max_chunk_size: int = CHUNK_SIZE,
    ):
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [self._make_chunk(document, text.strip(), 0)]

        if self.embed_fn is None:
            from src.indexing.embeddings import embed_texts
            self.embed_fn = embed_texts

        embeddings = self.embed_fn(sentences)
        groups = self._group_by_similarity(sentences, embeddings)

        chunks = []
        for idx, group in enumerate(groups):
            content = " ".join(group).strip()
            if content:
                chunks.append(self._make_chunk(document, content, idx))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        import re
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if s.strip()]

    def _group_by_similarity(
        self, sentences: list[str], embeddings: list[list[float]]
    ) -> list[list[str]]:
        embs = np.array(embeddings)
        groups: list[list[str]] = [[sentences[0]]]
        current_size = len(sentences[0])

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embs[i - 1], embs[i])
            new_size = current_size + len(sentences[i]) + 1

            if sim >= self.similarity_threshold and new_size <= self.max_chunk_size:
                groups[-1].append(sentences[i])
                current_size = new_size
            else:
                groups.append([sentences[i]])
                current_size = len(sentences[i])

        return groups

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
