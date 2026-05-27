from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from src.config import PROJECT_ROOT
from src.chunking.factory import ChunkingStrategy
from src.evaluation.dataset import EvalDataset
from src.evaluation.metrics import EvalMetrics, EvalResult
from src.evaluation.run_eval import _compute_summary, _print_summary
from src.generation.pipeline import GenerationPipeline
from src.indexing.pipeline import IndexingPipeline
from src.models import Document


class ChunkingComparison:
    """Run eval suite across all chunking strategies, produce comparison report."""

    def __init__(
        self,
        documents: list[Document],
        dataset: EvalDataset | None = None,
        max_eval_cases: int = 50,
    ):
        self.documents = documents
        self.dataset = dataset or EvalDataset.from_enterprise_bench(max_cases=max_eval_cases)
        self.strategies = [
            ChunkingStrategy.FIXED_SIZE,
            ChunkingStrategy.RECURSIVE,
            ChunkingStrategy.SEMANTIC,
        ]

    def run(self, output_dir: str | None = None) -> dict:
        output_path = Path(output_dir or str(PROJECT_ROOT / "comparison_results"))
        output_path.mkdir(exist_ok=True)

        all_summaries = {}

        for strategy in self.strategies:
            print(f"\n{'='*60}")
            print(f"  STRATEGY: {strategy.value}")
            print(f"{'='*60}")

            indexer = IndexingPipeline(strategy=strategy)
            indexer.run(self.documents)

            pipeline = GenerationPipeline(
                retriever=None,  # will create from current indexes
            )

            metrics = EvalMetrics()
            results: list[EvalResult] = []
            start = time.time()

            for i, case in enumerate(self.dataset):
                try:
                    gen_result = pipeline.ask(case.question)
                    eval_result = metrics.evaluate(case, gen_result)
                    results.append(eval_result)
                except Exception as e:
                    print(f"  [{i+1}] {case.question_id} ERROR: {e}")

            elapsed = time.time() - start
            summary = _compute_summary(results, elapsed, [])
            all_summaries[strategy.value] = summary

            _print_summary(summary)

            # Save per-strategy results
            strategy_dir = output_path / strategy.value
            strategy_dir.mkdir(exist_ok=True)
            with open(strategy_dir / "results.jsonl", "w") as f:
                for r in results:
                    f.write(json.dumps(asdict(r)) + "\n")

        # Save comparison report
        report = self._build_report(all_summaries)
        with open(output_path / "comparison.json", "w") as f:
            json.dump(report, f, indent=2)

        self._print_report(report)
        return report

    def _build_report(self, summaries: dict) -> dict:
        report = {"strategies": {}}

        metric_keys = [
            "avg_correctness",
            "avg_faithfulness",
            "avg_retrieval_relevance",
            "avg_retrieval_recall",
            "avg_citation_accuracy",
        ]

        for strategy, summary in summaries.items():
            overall = summary.get("overall", {})
            report["strategies"][strategy] = overall

        # Find winner per metric
        winners = {}
        for metric in metric_keys:
            best_strategy = max(
                summaries.keys(),
                key=lambda s: summaries[s].get("overall", {}).get(metric, 0),
            )
            best_value = summaries[best_strategy].get("overall", {}).get(metric, 0)
            winners[metric] = {"strategy": best_strategy, "value": best_value}

        report["winners"] = winners
        return report

    def _print_report(self, report: dict) -> None:
        print(f"\n{'='*70}")
        print("  CHUNKING STRATEGY COMPARISON REPORT")
        print(f"{'='*70}")

        header = f"  {'Metric':<25}"
        strategies = list(report["strategies"].keys())
        for s in strategies:
            header += f" {s:>15}"
        print(header)
        print("  " + "-" * (25 + 16 * len(strategies)))

        metrics = [
            ("Correctness", "avg_correctness"),
            ("Faithfulness", "avg_faithfulness"),
            ("Retrieval Relevance", "avg_retrieval_relevance"),
            ("Retrieval Recall", "avg_retrieval_recall"),
            ("Citation Accuracy", "avg_citation_accuracy"),
        ]

        for label, key in metrics:
            row = f"  {label:<25}"
            winner = report["winners"][key]["strategy"]
            for s in strategies:
                val = report["strategies"][s].get(key, 0)
                marker = " *" if s == winner else "  "
                row += f" {val:>12.1%}{marker}"
            print(row)

        print(f"\n  * = best for that metric")
        print(f"{'='*70}")
