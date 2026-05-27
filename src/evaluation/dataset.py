from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import DATASET_DIR


@dataclass
class EvalCase:
    question_id: str
    question: str
    question_type: str
    source_types: list[str]
    expected_doc_ids: list[str]
    gold_answer: str
    answer_facts: list[str]


class EvalDataset:
    def __init__(self, cases: list[EvalCase] | None = None):
        self.cases = cases or []

    @classmethod
    def from_enterprise_bench(
        cls,
        dataset_dir: str | None = None,
        max_cases: int | None = None,
        question_types: list[str] | None = None,
    ) -> EvalDataset:
        """Load eval cases from EnterpriseRAG-Bench questions."""
        base = Path(dataset_dir or DATASET_DIR)
        df = pd.read_parquet(base / "questions" / "test.parquet")

        if question_types:
            df = df[df["question_type"].isin(question_types)]

        if max_cases:
            df = df.head(max_cases)

        cases = []
        for _, row in df.iterrows():
            cases.append(EvalCase(
                question_id=row["question_id"],
                question=row["question"],
                question_type=row["question_type"],
                source_types=row["source_types"],
                expected_doc_ids=row["expected_doc_ids"],
                gold_answer=row["gold_answer"],
                answer_facts=row["answer_facts"],
            ))

        return cls(cases)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> EvalDataset:
        """Load from custom JSONL file."""
        cases = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                cases.append(EvalCase(**data))
        return cls(cases)

    def save_jsonl(self, path: str | Path) -> None:
        with open(path, "w") as f:
            for case in self.cases:
                data = {
                    "question_id": case.question_id,
                    "question": case.question,
                    "question_type": case.question_type,
                    "source_types": case.source_types,
                    "expected_doc_ids": case.expected_doc_ids,
                    "gold_answer": case.gold_answer,
                    "answer_facts": case.answer_facts,
                }
                f.write(json.dumps(data) + "\n")

    def filter_by_type(self, question_type: str) -> EvalDataset:
        return EvalDataset([c for c in self.cases if c.question_type == question_type])

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)
