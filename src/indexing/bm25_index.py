from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import BM25_INDEX_PATH
from src.models import Chunk


class BM25Index:
    def __init__(self, index_path: str = BM25_INDEX_PATH):
        self.index_path = Path(index_path)
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.corpus: list[list[str]] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.corpus = [self._tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def query(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokens = self._tokenize(text)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.chunk_ids[i], float(scores[i])) for i in top_indices]

    def save(self) -> None:
        data = {
            "chunk_ids": self.chunk_ids,
            "corpus": self.corpus,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self.chunk_ids = data["chunk_ids"]
        self.corpus = data["corpus"]
        self.bm25 = BM25Okapi(self.corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()
