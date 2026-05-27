from __future__ import annotations

import httpx

from src.config import OLLAMA_BASE_URL, GENERATION_MODEL
from src.retrieval.dense import RetrievalResult
from src.generation.prompts import SYSTEM_PROMPT, CONTEXT_TEMPLATE


class GroundedGenerator:
    def __init__(self, model: str = GENERATION_MODEL):
        self.model = model

    def generate(self, question: str, context_chunks: list[RetrievalResult]) -> str:
        context_blocks = self._format_context(context_chunks)
        prompt = CONTEXT_TEMPLATE.format(
            context_blocks=context_blocks,
            question=question,
        )

        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": self.model,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1024},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def _format_context(self, chunks: list[RetrievalResult]) -> str:
        blocks = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.metadata.get("title", "Unknown")
            source = chunk.metadata.get("source_type", "unknown")
            blocks.append(f"[{i}] (Source: {title} | Type: {source})\n{chunk.content}")
        return "\n\n---\n\n".join(blocks)
