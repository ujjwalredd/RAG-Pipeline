# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) pipeline with hybrid search over internal documentation. Ingests company docs, indexes with dense vector + sparse keyword search, retrieves relevant context, and generates grounded answers with inline source citations.

## Tech Stack

- **Python 3.11+**
- **Local LLM** via Ollama (e.g., Llama 3, Mistral) — used for generation, reranking, and citation verification
- **Local Embeddings** via Ollama or sentence-transformers (e.g., `nomic-embed-text`, `all-MiniLM-L6-v2`)
- **ChromaDB** — vector store (file-based)
- **rank_bm25** — sparse/keyword search
- **FastAPI** — async API layer
- **Streamlit** — query dashboard
- **Docker Compose** — containerization (API + ChromaDB + frontend)

## Architecture

```
src/
  ingestion/       # Document loading (md, txt, html, pdf), normalization, metadata extraction
  chunking/        # Three strategies: fixed-size overlap, recursive by headers, semantic boundary
  indexing/        # Embedding generation, ChromaDB storage, BM25 index, deduplication (cosine >0.95)
  retrieval/       # Dense (vector), sparse (BM25), Reciprocal Rank Fusion, cross-encoder reranker
  generation/      # Grounded prompt construction, LLM calls, citation parsing
  evaluation/      # Citation verification (LLM-as-judge), confidence scoring, faithfulness checks
  api/             # FastAPI routes: POST /v1/ask, GET /v1/documents, POST /v1/ingest
  dashboard/       # Streamlit UI with citation display, chunk viewer, confidence breakdown
```

Key design decisions:
- Both vector and BM25 indexes built over the same chunks and must stay in sync
- Hybrid retrieval uses configurable RRF weighting (default 0.7 dense / 0.3 sparse)
- Reranker narrows top-20 fusion results to top-5 before generation
- Citation verification is a post-generation step: each [n] claim-pair checked by LLM-as-judge
- Confidence score is composite: retrieval relevance + citation coverage + answer completeness
- Below-threshold retrieval confidence triggers structured "I don't know" response instead of hallucination

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run Ollama (must be running for LLM and embeddings)
ollama serve
ollama pull llama3          # generation model
ollama pull nomic-embed-text # embedding model

# Run API server
uvicorn src.api.main:app --reload --port 8000

# Run dashboard
streamlit run src/dashboard/app.py

# Run tests
pytest
pytest tests/test_retrieval.py -v          # single module
pytest tests/test_retrieval.py::test_rrf -v # single test

# Run evaluation suite
python -m src.evaluation.run_eval

# Docker
docker-compose up --build

# Linting
ruff check src/ tests/
ruff format src/ tests/
```

## Local LLM Configuration

All LLM calls go through Ollama's API (`http://localhost:11434`). Model names are configured in `src/config.py` — no OpenAI or external API keys required. When adding new LLM calls, use the shared Ollama client rather than creating new connections.

## Dataset

EnterpriseRAG-Bench (MIT) in `data/EnterpriseRAG-Bench/`. 511K docs (Confluence, GitHub, Slack, Gmail, Jira, etc.), 500 golden Q&A pairs with expected doc IDs and answer facts. Parquet format loaded via `src/ingestion/enterprise_loader.py`.

## API Endpoints

- `GET /health` — system status, indexed chunk count, Ollama availability
- `POST /v1/ask` — question answering with retrieval mode, reranker toggle, weight config
- `GET /v1/documents` — list indexed documents with metadata
- `POST /v1/ingest` — ingest from dataset with chunking strategy selection

## Key Modules

- `src/api/dependencies.py` — singleton `AppState` holds vector store, BM25 index, generation pipeline
- `src/generation/pipeline.py` — `GenerationPipeline.ask()` orchestrates retrieve → generate → verify → score
- `src/retrieval/hybrid.py` — `HybridRetriever` with `retrieve()`, `retrieve_dense_only()`, `retrieve_sparse_only()`
- `src/evaluation/compare.py` — `ChunkingComparison.run()` benchmarks all 3 strategies side-by-side
- `scripts/seed.py` — pulls Ollama models and indexes 200 sample docs for quick start
