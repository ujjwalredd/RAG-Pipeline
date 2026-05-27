from __future__ import annotations

from src.models import Document, Chunk
from src.chunking.factory import get_chunker, ChunkingStrategy
from src.indexing.embeddings import embed_texts
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index
from src.indexing.deduplication import deduplicate_chunks


class IndexingPipeline:
    def __init__(
        self,
        strategy: ChunkingStrategy | str = ChunkingStrategy.RECURSIVE,
        batch_size: int = 50,
        **chunker_kwargs,
    ):
        self.chunker = get_chunker(strategy, **chunker_kwargs)
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()
        self.batch_size = batch_size
        self.all_chunks: list[Chunk] = []

    def run(self, documents: list[Document]) -> dict:
        """Full pipeline: chunk → embed → dedup → index."""
        print(f"Processing {len(documents)} documents...")

        # Chunk all documents
        chunks = []
        for doc in documents:
            chunks.extend(self.chunker.chunk(doc))
        print(f"Generated {len(chunks)} chunks using '{self.chunker.strategy_name}' strategy")

        # Embed in batches
        all_embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_texts = [c.content for c in batch]
            batch_embeddings = embed_texts(batch_texts)
            all_embeddings.extend(batch_embeddings)
            print(f"  Embedded batch {i // self.batch_size + 1}/{(len(chunks) - 1) // self.batch_size + 1}")

        # Deduplicate
        chunks, all_embeddings = deduplicate_chunks(chunks, all_embeddings)

        # Store in vector DB
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_embs = all_embeddings[i : i + self.batch_size]
            self.vector_store.add_chunks(batch_chunks, batch_embs)

        # Build BM25 index
        self.bm25_index.build(chunks)
        self.bm25_index.save()

        self.all_chunks = chunks

        stats = {
            "documents": len(documents),
            "chunks_generated": len(chunks),
            "chunks_indexed": self.vector_store.count(),
            "strategy": self.chunker.strategy_name,
        }
        print(f"Indexing complete: {stats}")
        return stats
