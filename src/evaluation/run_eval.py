from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from src.config import PROJECT_ROOT
from src.evaluation.dataset import EvalDataset
from src.evaluation.metrics import EvalMetrics, EvalResult
from src.generation.pipeline import GenerationPipeline


def run_evaluation(
    dataset: EvalDataset | None = None,
    pipeline: GenerationPipeline | None = None,
    max_cases: int | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run full evaluation suite and save results."""
    if dataset is None:
        dataset = EvalDataset.from_enterprise_bench(max_cases=max_cases or 50)
    if pipeline is None:
        pipeline = GenerationPipeline()

    output_path = Path(output_dir or str(PROJECT_ROOT / "eval_results"))
    output_path.mkdir(exist_ok=True)

    metrics = EvalMetrics()
    results: list[EvalResult] = []
    errors: list[dict] = []

    print(f"Running evaluation on {len(dataset)} cases...")
    start = time.time()

    for i, case in enumerate(dataset):
        try:
            gen_result = pipeline.ask(case.question)
            eval_result = metrics.evaluate(case, gen_result)
            results.append(eval_result)
            print(
                f"  [{i+1}/{len(dataset)}] {case.question_id} "
                f"correctness={eval_result.answer_correctness:.2f} "
                f"faithfulness={eval_result.faithfulness:.2f} "
                f"retrieval_recall={eval_result.retrieval_recall:.2f}"
            )
        except Exception as e:
            errors.append({"question_id": case.question_id, "error": str(e)})
            print(f"  [{i+1}/{len(dataset)}] {case.question_id} ERROR: {e}")

    elapsed = time.time() - start

    summary = _compute_summary(results, elapsed, errors)
    _save_results(results, summary, errors, output_path)

    print(f"\nEvaluation complete in {elapsed:.1f}s")
    print(f"Results saved to {output_path}")
    _print_summary(summary)

    return summary


def _compute_summary(
    results: list[EvalResult], elapsed: float, errors: list[dict]
) -> dict:
    if not results:
        return {"total": 0, "errors": len(errors), "elapsed_seconds": elapsed}

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    by_type: dict[str, list[EvalResult]] = {}
    for r in results:
        by_type.setdefault(r.question_type, []).append(r)

    type_summaries = {}
    for qtype, type_results in by_type.items():
        type_summaries[qtype] = {
            "count": len(type_results),
            "avg_correctness": round(avg([r.answer_correctness for r in type_results]), 3),
            "avg_faithfulness": round(avg([r.faithfulness for r in type_results]), 3),
            "avg_retrieval_relevance": round(avg([r.retrieval_relevance for r in type_results]), 3),
            "avg_retrieval_recall": round(avg([r.retrieval_recall for r in type_results]), 3),
            "avg_citation_accuracy": round(avg([r.citation_accuracy for r in type_results]), 3),
        }

    return {
        "total": len(results),
        "errors": len(errors),
        "elapsed_seconds": round(elapsed, 1),
        "overall": {
            "avg_correctness": round(avg([r.answer_correctness for r in results]), 3),
            "avg_faithfulness": round(avg([r.faithfulness for r in results]), 3),
            "avg_retrieval_relevance": round(avg([r.retrieval_relevance for r in results]), 3),
            "avg_retrieval_recall": round(avg([r.retrieval_recall for r in results]), 3),
            "avg_citation_accuracy": round(avg([r.citation_accuracy for r in results]), 3),
        },
        "by_question_type": type_summaries,
    }


def _save_results(
    results: list[EvalResult],
    summary: dict,
    errors: list[dict],
    output_path: Path,
) -> None:
    with open(output_path / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if errors:
        with open(output_path / "errors.json", "w") as f:
            json.dump(errors, f, indent=2)


def _print_summary(summary: dict) -> None:
    overall = summary.get("overall", {})
    print(f"\n{'='*50}")
    print(f"  EVALUATION SUMMARY ({summary['total']} cases, {summary['errors']} errors)")
    print(f"{'='*50}")
    print(f"  Answer Correctness:    {overall.get('avg_correctness', 0):.1%}")
    print(f"  Faithfulness:          {overall.get('avg_faithfulness', 0):.1%}")
    print(f"  Retrieval Relevance:   {overall.get('avg_retrieval_relevance', 0):.1%}")
    print(f"  Retrieval Recall:      {overall.get('avg_retrieval_recall', 0):.1%}")
    print(f"  Citation Accuracy:     {overall.get('avg_citation_accuracy', 0):.1%}")
    print(f"{'='*50}")

    by_type = summary.get("by_question_type", {})
    if by_type:
        print("\n  By Question Type:")
        for qtype, stats in by_type.items():
            print(f"    {qtype} (n={stats['count']}): "
                  f"correct={stats['avg_correctness']:.1%} "
                  f"faithful={stats['avg_faithfulness']:.1%} "
                  f"recall={stats['avg_retrieval_recall']:.1%}")


if __name__ == "__main__":
    run_evaluation()
