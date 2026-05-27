import tempfile
from pathlib import Path

from src.ingestion.loader import load_file, load_directory, _normalize_text
from src.ingestion.enterprise_loader import load_enterprise_dataset


def test_normalize_text():
    raw = "  hello   world  \n\n\n\nfoo  \r\n  bar  "
    result = _normalize_text(raw)
    assert "\r\n" not in result
    assert "\n\n\n" not in result
    assert "  " not in result


def test_load_txt_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Hello world.\nThis is a test document.")
        f.flush()
        doc = load_file(f.name)

    assert doc.content == "Hello world.\nThis is a test document."
    assert doc.source_type == "txt"


def test_load_md_file():
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("# Title\n\nSome content here.\n\n## Section\n\nMore content.")
        f.flush()
        doc = load_file(f.name)

    assert "# Title" in doc.content
    assert doc.source_type == "md"


def test_load_html_file():
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as f:
        f.write("<html><body><h1>Test</h1><p>Content</p><script>evil()</script></body></html>")
        f.flush()
        doc = load_file(f.name)

    assert "evil" not in doc.content
    assert "Content" in doc.content


def test_load_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "a.txt").write_text("doc a")
        (Path(tmpdir) / "b.md").write_text("# doc b")
        (Path(tmpdir) / "c.json").write_text("{}")  # should be skipped

        docs = load_directory(tmpdir)
        assert len(docs) == 2


def test_enterprise_loader():
    docs = load_enterprise_dataset(max_docs=5)
    assert len(docs) == 5
    assert all(d.doc_id for d in docs)
    assert all(d.content for d in docs)
    assert all(d.source_type for d in docs)
