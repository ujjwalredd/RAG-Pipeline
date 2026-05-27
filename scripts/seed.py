"""Seed script: pull Ollama models and index a sample of the dataset."""

import subprocess
import sys
import time

import httpx

OLLAMA_BASE_URL = "http://localhost:11434"
MODELS = ["llama3", "nomic-embed-text"]
SEED_DOCS = 200


def wait_for_ollama(max_retries: int = 30) -> None:
    print("Waiting for Ollama...")
    for i in range(max_retries):
        try:
            resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                print("Ollama is ready.")
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    print("WARNING: Ollama not reachable, continuing anyway.")


def pull_models() -> None:
    for model in MODELS:
        print(f"Pulling {model}...")
        try:
            resp = httpx.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": model, "stream": False},
                timeout=600.0,
            )
            resp.raise_for_status()
            print(f"  {model} ready.")
        except httpx.HTTPError as e:
            print(f"  WARNING: Could not pull {model}: {e}")


def seed_index() -> None:
    from src.ingestion.enterprise_loader import load_enterprise_dataset
    from src.indexing.pipeline import IndexingPipeline

    print(f"Loading {SEED_DOCS} documents...")
    docs = load_enterprise_dataset(max_docs=SEED_DOCS)

    print("Indexing with recursive chunking...")
    pipeline = IndexingPipeline(strategy="recursive")
    stats = pipeline.run(docs)
    print(f"Seed complete: {stats}")


if __name__ == "__main__":
    wait_for_ollama()
    pull_models()
    seed_index()
