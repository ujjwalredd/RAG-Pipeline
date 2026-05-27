from __future__ import annotations

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.models import Chunk, Document
from src.chunking.base import BaseChunker


class FixedSizeChunker(BaseChunker):
    strategy_name = "fixed_size"

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(self._make_chunk(document, chunk_text, idx))
                idx += 1

            start += self.chunk_size - self.overlap
            if self.chunk_size - self.overlap <= 0:
                break

        return chunks
