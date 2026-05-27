from __future__ import annotations

import chromadb

from src.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.models import Chunk


class VectorStore:
    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        if not chunks:
            return

        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "source_type": c.source_type,
                    "title": c.title,
                    "section_heading": c.section_heading,
                    "chunking_strategy": c.chunking_strategy,
                    "char_count": c.char_count,
                }
                for c in chunks
            ],
        )

    def query(
        self, embedding: list[float], top_k: int = 10
    ) -> dict:
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def get_all_embeddings(self) -> tuple[list[str], list[list[float]]]:
        result = self.collection.get(include=["embeddings"])
        return result["ids"], result["embeddings"]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
