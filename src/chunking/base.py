from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import Chunk, Document


class BaseChunker(ABC):
    strategy_name: str = "base"

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        ...

    def _make_chunk(
        self,
        document: Document,
        content: str,
        chunk_index: int,
        section_heading: str = "",
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{document.doc_id}_{self.strategy_name}_{chunk_index}",
            doc_id=document.doc_id,
            content=content,
            chunk_index=chunk_index,
            source_type=document.source_type,
            title=document.title,
            section_heading=section_heading,
            chunking_strategy=self.strategy_name,
            metadata=document.metadata.copy(),
        )
