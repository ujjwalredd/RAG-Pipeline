from __future__ import annotations

import hashlib
import re
from pathlib import Path

from bs4 import BeautifulSoup

from src.models import Document

SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".htm", ".pdf"}


def load_file(path: str | Path) -> Document:
    path = Path(path)
    if path.suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    doc_id = hashlib.sha256(str(path).encode()).hexdigest()[:32]
    source_type = path.suffix.lstrip(".")
    title = path.stem

    if path.suffix == ".pdf":
        content = _load_pdf(path)
    elif path.suffix in (".html", ".htm"):
        content = _load_html(path)
    else:
        content = path.read_text(encoding="utf-8")

    content = _normalize_text(content)

    return Document(
        doc_id=doc_id,
        content=content,
        source_type=source_type,
        title=title,
        metadata={"file_path": str(path)},
    )


def load_directory(dir_path: str | Path) -> list[Document]:
    dir_path = Path(dir_path)
    docs = []
    for path in sorted(dir_path.rglob("*")):
        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
            docs.append(load_file(path))
    return docs


def _load_pdf(path: Path) -> str:
    import fitz

    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def _load_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()
