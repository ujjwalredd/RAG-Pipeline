from __future__ import annotations

from enum import Enum

from src.chunking.base import BaseChunker
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.semantic import SemanticChunker


class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


def get_chunker(strategy: ChunkingStrategy | str, **kwargs) -> BaseChunker:
    strategy = ChunkingStrategy(strategy)
    match strategy:
        case ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunker(**kwargs)
        case ChunkingStrategy.RECURSIVE:
            return RecursiveChunker(**kwargs)
        case ChunkingStrategy.SEMANTIC:
            return SemanticChunker(**kwargs)
