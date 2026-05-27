from __future__ import annotations

import re

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.models import Chunk, Document
from src.chunking.base import BaseChunker

HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class RecursiveChunker(BaseChunker):
    """Split by section headers first, then by size within sections."""

    strategy_name = "recursive"

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        sections = self._split_by_headers(text)
        chunks = []
        idx = 0

        for heading, section_text in sections:
            sub_chunks = self._split_by_size(section_text)
            for sub in sub_chunks:
                if sub.strip():
                    chunks.append(self._make_chunk(document, sub.strip(), idx, heading))
                    idx += 1

        return chunks

    def _split_by_headers(self, text: str) -> list[tuple[str, str]]:
        matches = list(HEADER_PATTERN.finditer(text))
        if not matches:
            return [("", text)]

        sections = []
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(("", preamble))

        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append((heading, section_text))

        return sections

    def _split_by_size(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        parts = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                break_at = text.rfind("\n", start, end)
                if break_at > start:
                    end = break_at + 1
            parts.append(text[start:end])
            start = max(start + 1, end - self.overlap)

        return parts
