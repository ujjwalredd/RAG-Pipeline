from __future__ import annotations

import httpx

from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL

_client = httpx.Client(timeout=120.0)


def embed_texts(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Embed a batch of texts via Ollama API. Supports both new and legacy endpoints."""
    embeddings = []
    for text in texts:
        embedding = _embed_one(text, model)
        embeddings.append(embedding)
    return embeddings


def _embed_one(text: str, model: str) -> list[float]:
    # Try new endpoint first (/api/embed), fall back to legacy (/api/embeddings)
    try:
        resp = _client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": model, "input": text},
        )
        if resp.status_code == 200:
            data = resp.json()
            if "embeddings" in data:
                return data["embeddings"][0]
            if "embedding" in data:
                return data["embedding"]
    except httpx.HTTPError:
        pass

    # Legacy endpoint
    resp = _client.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": model, "prompt": text},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]


def embed_single(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    return _embed_one(text, model)
