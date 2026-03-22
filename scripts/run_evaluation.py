"""CLI to run RAG and baseline evaluations, then generate the benchmark report.

Usage:
    uv run python -X utf8 scripts/run_evaluation.py              # Run both RAG + baseline + report
    uv run python -X utf8 scripts/run_evaluation.py --rag-only   # Run RAG evaluation only
    uv run python -X utf8 scripts/run_evaluation.py --base-only  # Run baseline evaluation only
    uv run python -X utf8 scripts/run_evaluation.py --report     # Generate report from latest results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.evaluation.golden_set import load_golden_set
from src.evaluation.ragas_eval import evaluate_batch, summarize
from src.evaluation.report import generate_full_report


def _print_summary(label: str, results: list) -> None:
    summary = summarize(results)
    logger.info(f"\n{'=' * 50}")
    logger.info(f"{label} Summary ({summary.total_questions} questions)")
    logger.info(f"  Relevancy:   {summary.avg_relevancy:.3f}")
    logger.info(f"  Faithfulness: {summary.avg_faithfulness:.3f}")
    logger.info(f"  Correctness:  {summary.avg_correctness:.3f}")
    for cat, scores in sorted(summary.by_category.items()):
        logger.info(f"  [{cat}] rel={scores['avg_relevancy']:.2f} "
                     f"faith={scores['avg_faithfulness']:.2f} "
                     f"corr={scores['avg_correctness']:.2f} "
                     f"(n={scores['count']})")
    logger.info(f"{'=' * 50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline")
    parser.add_argument("--rag-only", action="store_true", help="Run RAG evaluation only")
    parser.add_argument("--base-only", action="store_true", help="Run baseline evaluation only")
    parser.add_argument("--report", action="store_true", help="Generate report from existing results")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0 = all)")
    args = parser.parse_args()

    if args.report:
        logger.info("Generating report from existing results...")
        report_path = generate_full_report()
        logger.info(f"Report generated: {report_path}")
        return

    golden = load_golden_set()
    if args.limit > 0:
        golden = golden[: args.limit]
    logger.info(f"Loaded {len(golden)} golden Q&A entries")

    run_rag = not args.base_only
    run_base = not args.rag_only

    if run_rag:
        logger.info("Starting RAG evaluation...")
        rag_results = evaluate_batch(golden, use_rag=True, label="rag")
        _print_summary("RAG", rag_results)

    if run_base:
        logger.info("Starting baseline (LLM-only) evaluation...")
        base_results = evaluate_batch(golden, use_rag=False, label="baseline")
        _print_summary("Baseline", base_results)

    if run_rag and run_base:
        logger.info("Generating comparison report...")
        report_path = generate_full_report()
        logger.info(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
