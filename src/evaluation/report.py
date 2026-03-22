"""Generate comparison report: RAG vs Baseline with tables and plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.config import EVALUATION_DIR
from src.evaluation.ragas_eval import EvalResult, EvalSummary, summarize

RESULTS_DIR = EVALUATION_DIR / "results"
METRICS = ["relevancy", "faithfulness", "correctness"]


def _load_results(path: Path) -> list[EvalResult]:
    """Load evaluation results from a JSON file into EvalResult objects."""
    from src.evaluation.ragas_eval import MetricScore

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for d in data:
        results.append(EvalResult(
            id=d["id"],
            category=d["category"],
            question=d["question"],
            reference_answer=d["reference_answer"],
            generated_answer=d["generated_answer"],
            tool_calls=d.get("tool_calls", []),
            relevancy=MetricScore(**d["relevancy"]) if d.get("relevancy") else None,
            faithfulness=MetricScore(**d["faithfulness"]) if d.get("faithfulness") else None,
            correctness=MetricScore(**d["correctness"]) if d.get("correctness") else None,
        ))
    return results


def find_latest_results(label: str) -> Path | None:
    """Find the most recent results file for a given label (rag/baseline)."""
    pattern = f"{label}_*.json"
    files = sorted(RESULTS_DIR.glob(pattern))
    return files[-1] if files else None


def plot_overall_comparison(
    rag_summary: EvalSummary,
    baseline_summary: EvalSummary,
    output_path: Path | None = None,
) -> Path:
    """Bar chart comparing overall RAG vs Baseline scores."""
    if output_path is None:
        output_path = RESULTS_DIR / "overall_comparison.png"

    rag_scores = [rag_summary.avg_relevancy, rag_summary.avg_faithfulness, rag_summary.avg_correctness]
    base_scores = [baseline_summary.avg_relevancy, baseline_summary.avg_faithfulness, baseline_summary.avg_correctness]

    x = np.arange(len(METRICS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, rag_scores, width, label="RAG", color="#2196F3")
    bars2 = ax.bar(x + width / 2, base_scores, width, label="Baseline (LLM-only)", color="#FF9800")

    ax.set_ylabel("Score (0–1)")
    ax.set_title("RAG vs Baseline — Overall Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in METRICS])
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Overall comparison plot saved to {output_path}")
    return output_path


def plot_category_comparison(
    rag_summary: EvalSummary,
    baseline_summary: EvalSummary,
    output_path: Path | None = None,
) -> Path:
    """Grouped bar chart comparing RAG vs Baseline by category (correctness)."""
    if output_path is None:
        output_path = RESULTS_DIR / "category_comparison.png"

    categories = sorted(rag_summary.by_category.keys())
    rag_corr = [rag_summary.by_category[c]["avg_correctness"] for c in categories]
    base_corr = [baseline_summary.by_category.get(c, {}).get("avg_correctness", 0.0) for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, rag_corr, width, label="RAG", color="#2196F3")
    bars2 = ax.bar(x + width / 2, base_corr, width, label="Baseline", color="#FF9800")

    ax.set_ylabel("Correctness Score (0–1)")
    ax.set_title("RAG vs Baseline — Correctness by Category")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in categories], rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Category comparison plot saved to {output_path}")
    return output_path


def plot_per_question_delta(
    rag_results: list[EvalResult],
    baseline_results: list[EvalResult],
    output_path: Path | None = None,
) -> Path:
    """Horizontal bar chart showing per-question correctness delta (RAG - Baseline)."""
    if output_path is None:
        output_path = RESULTS_DIR / "per_question_delta.png"

    base_map = {r.id: r for r in baseline_results}
    deltas = []
    labels = []
    for r in rag_results:
        if r.correctness and r.id in base_map and base_map[r.id].correctness:
            delta = r.correctness.score - base_map[r.id].correctness.score
            deltas.append(delta)
            labels.append(r.id)

    # Sort by delta
    pairs = sorted(zip(deltas, labels), key=lambda x: x[0])
    deltas = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]

    colors = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.3)))
    ax.barh(range(len(labels)), deltas, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Correctness Delta (RAG − Baseline)")
    ax.set_title("Per-Question Correctness Improvement")
    ax.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Per-question delta plot saved to {output_path}")
    return output_path


def generate_markdown_report(
    rag_summary: EvalSummary,
    baseline_summary: EvalSummary,
    rag_results: list[EvalResult],
    baseline_results: list[EvalResult],
    output_path: Path | None = None,
) -> Path:
    """Generate a Markdown report with tables and embedded plot references."""
    if output_path is None:
        output_path = RESULTS_DIR / "benchmark_report.md"

    lines = [
        "# RAG vs Baseline Benchmark Report",
        "",
        "## Overall Metrics",
        "",
        "| Metric | RAG | Baseline | Delta |",
        "|--------|-----|----------|-------|",
    ]

    for metric in METRICS:
        rag_val = getattr(rag_summary, f"avg_{metric}")
        base_val = getattr(baseline_summary, f"avg_{metric}")
        delta = rag_val - base_val
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {metric.title()} | {rag_val:.3f} | {base_val:.3f} | {sign}{delta:.3f} |")

    lines.extend([
        "",
        "![Overall Comparison](overall_comparison.png)",
        "",
        "## By Category (Correctness)",
        "",
        "| Category | RAG | Baseline | Delta |",
        "|----------|-----|----------|-------|",
    ])

    categories = sorted(rag_summary.by_category.keys())
    for cat in categories:
        rag_val = rag_summary.by_category[cat]["avg_correctness"]
        base_val = baseline_summary.by_category.get(cat, {}).get("avg_correctness", 0.0)
        delta = rag_val - base_val
        sign = "+" if delta >= 0 else ""
        label = cat.replace("_", " ").title()
        lines.append(f"| {label} | {rag_val:.3f} | {base_val:.3f} | {sign}{delta:.3f} |")

    lines.extend([
        "",
        "![Category Comparison](category_comparison.png)",
        "",
        "![Per-Question Delta](per_question_delta.png)",
        "",
        "## Detailed Results",
        "",
    ])

    # Per-question table
    base_map = {r.id: r for r in baseline_results}
    lines.append("| ID | Category | RAG Correct | Base Correct | Tools Used |")
    lines.append("|----|----------|-------------|--------------|------------|")
    for r in rag_results:
        rag_c = f"{r.correctness.score:.1f}" if r.correctness else "N/A"
        br = base_map.get(r.id)
        base_c = f"{br.correctness.score:.1f}" if br and br.correctness else "N/A"
        tools = ", ".join(tc["name"] for tc in r.tool_calls) if r.tool_calls else "—"
        lines.append(f"| {r.id} | {r.category} | {rag_c} | {base_c} | {tools} |")

    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown report saved to {output_path}")
    return output_path


def generate_full_report(
    rag_path: Path | None = None,
    baseline_path: Path | None = None,
) -> Path:
    """Load results, compute summaries, generate all plots and the report."""
    if rag_path is None:
        rag_path = find_latest_results("rag")
    if baseline_path is None:
        baseline_path = find_latest_results("baseline")

    if not rag_path or not baseline_path:
        raise FileNotFoundError(
            f"Missing results files. RAG: {rag_path}, Baseline: {baseline_path}"
        )

    rag_results = _load_results(rag_path)
    baseline_results = _load_results(baseline_path)

    rag_summary = summarize(rag_results)
    baseline_summary = summarize(baseline_results)

    # Generate plots
    plot_overall_comparison(rag_summary, baseline_summary)
    plot_category_comparison(rag_summary, baseline_summary)
    plot_per_question_delta(rag_results, baseline_results)

    # Generate markdown report
    report_path = generate_markdown_report(
        rag_summary, baseline_summary, rag_results, baseline_results,
    )

    return report_path
