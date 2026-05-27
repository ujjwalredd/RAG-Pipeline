from src.models import Document
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.factory import get_chunker, ChunkingStrategy


def _make_doc(content: str) -> Document:
    return Document(doc_id="test_doc", content=content, source_type="txt", title="Test")


def test_fixed_size_basic():
    doc = _make_doc("a" * 1000)
    chunker = FixedSizeChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    assert all(c.chunking_strategy == "fixed_size" for c in chunks)
    assert all(c.char_count <= 200 for c in chunks)


def test_fixed_size_small_doc():
    doc = _make_doc("short text")
    chunker = FixedSizeChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1


def test_fixed_size_empty():
    doc = _make_doc("")
    chunker = FixedSizeChunker()
    assert chunker.chunk(doc) == []


def test_recursive_with_headers():
    content = "# Intro\n\nSome intro text.\n\n## Details\n\nDetailed content here.\n\n## Conclusion\n\nFinal thoughts."
    doc = _make_doc(content)
    chunker = RecursiveChunker(chunk_size=500)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 3
    assert any(c.section_heading == "Intro" for c in chunks)
    assert any(c.section_heading == "Details" for c in chunks)
    assert all(c.chunking_strategy == "recursive" for c in chunks)


def test_recursive_no_headers():
    doc = _make_doc("Just plain text with no headers at all. " * 20)
    chunker = RecursiveChunker(chunk_size=100)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1


def test_factory():
    chunker = get_chunker(ChunkingStrategy.FIXED_SIZE, chunk_size=100)
    assert isinstance(chunker, FixedSizeChunker)

    chunker = get_chunker("recursive")
    assert isinstance(chunker, RecursiveChunker)


def test_chunk_ids_unique():
    doc = _make_doc("a" * 1000)
    chunker = FixedSizeChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_metadata_propagated():
    doc = Document(
        doc_id="d1",
        content="Some content " * 50,
        source_type="confluence",
        title="My Doc",
        metadata={"dataset": "test"},
    )
    chunker = FixedSizeChunker(chunk_size=100)
    chunks = chunker.chunk(doc)

    for c in chunks:
        assert c.doc_id == "d1"
        assert c.source_type == "confluence"
        assert c.title == "My Doc"
        assert c.metadata.get("dataset") == "test"
