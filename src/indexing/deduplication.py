from __future__ import annotations

import numpy as np

from src.config import DEDUP_SIMILARITY_THRESHOLD
from src.models import Chunk


def deduplicate_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> tuple[list[Chunk], list[list[float]]]:
    """Remove near-duplicate chunks based on cosine similarity."""
    if not chunks:
        return [], []

    embs = np.array(embeddings)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embs / norms

    keep_mask = [True] * len(chunks)

    for i in range(len(chunks)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(chunks)):
            if not keep_mask[j]:
                continue
            sim = float(np.dot(normalized[i], normalized[j]))
            if sim > threshold:
                keep_mask[j] = False

    kept_chunks = [c for c, k in zip(chunks, keep_mask) if k]
    kept_embeddings = [e for e, k in zip(embeddings, keep_mask) if k]

    removed = len(chunks) - len(kept_chunks)
    if removed > 0:
        print(f"Dedup: removed {removed} near-duplicate chunks ({len(kept_chunks)} remaining)")

    return kept_chunks, kept_embeddings
