from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    use_reranker: bool = True
    dense_weight: float = Field(0.7, ge=0.0, le=1.0)
    sparse_weight: float = Field(0.3, ge=0.0, le=1.0)
    retrieval_mode: str = Field("hybrid", pattern="^(hybrid|dense|sparse)$")


class CitationResponse(BaseModel):
    citation_id: int
    claim: str
    source_chunk_id: str
    supported: bool


class ChunkResponse(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: dict


class ConfidenceResponse(BaseModel):
    retrieval_confidence: float
    citation_coverage: float
    answer_completeness: float
    composite: float


class AskResponse(BaseModel):
    question: str
    answer: str
    is_confident: bool
    confidence: ConfidenceResponse
    citations: list[CitationResponse]
    context_chunks: list[ChunkResponse]


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    source_type: str
    char_count: int


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentInfo]


class IngestRequest(BaseModel):
    max_docs: int | None = Field(None, ge=1, le=100000)
    source_types: list[str] | None = None
    chunking_strategy: str = Field("recursive", pattern="^(fixed_size|recursive|semantic)$")


class IngestResponse(BaseModel):
    documents: int
    chunks_generated: int
    chunks_indexed: int
    strategy: str


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    ollama_available: bool
