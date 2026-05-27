from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATASET_DIR
from src.models import Document


def load_enterprise_dataset(
    dataset_dir: str | None = None,
    max_docs: int | None = None,
    source_types: list[str] | None = None,
) -> list[Document]:
    """Load documents from EnterpriseRAG-Bench parquet files."""
    base = Path(dataset_dir or DATASET_DIR)
    docs_path = base / "documents" / "test.parquet"

    df = pd.read_parquet(docs_path)

    if source_types:
        df = df[df["source_type"].isin(source_types)]

    if max_docs:
        df = df.head(max_docs)

    documents = []
    for _, row in df.iterrows():
        documents.append(
            Document(
                doc_id=row["doc_id"],
                content=row["content"],
                source_type=row["source_type"],
                title=row.get("title", ""),
                metadata={"dataset": "EnterpriseRAG-Bench"},
            )
        )

    return documents


def load_questions(dataset_dir: str | None = None) -> pd.DataFrame:
    """Load golden Q&A pairs for evaluation."""
    base = Path(dataset_dir or DATASET_DIR)
    return pd.read_parquet(base / "questions" / "test.parquet")
