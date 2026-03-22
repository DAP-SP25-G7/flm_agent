"""Generate publication-quality plots for the research paper."""

import re
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
COLOR_PRIMARY = PALETTE[0]
COLOR_ACCENT = PALETTE[1]
COLOR_GOOD = PALETTE[2]
COLOR_BAD = PALETTE[3]

RCPARAMS: dict = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
}

FIGSIZE_SINGLE = (7, 5)
FIGSIZE_WIDE = (14, 5)
FIGSIZE_TALL = (7, 7)
FIGSIZE_TABLE = (8, 2)


def apply_style() -> None:
    plt.rcParams.update(RCPARAMS)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update(RCPARAMS)


def latex_safe(s: str) -> str:
    return re.sub(r"([_&%$#{}])", r"\\\1", str(s))


def add_bar_labels(ax, bars, fmt="{:.3f}", fontsize=8, rotation=0):
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )


RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation" / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"


def load_results(label: str) -> list[dict]:
    candidates = sorted(RESULTS_DIR.glob(f"{label}_*.json"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No {label} results found in {RESULTS_DIR}")
    with open(candidates[0], "r", encoding="utf-8") as f:
        return json.load(f)


def compute_summary(results: list[dict]) -> dict:
    metrics = {"relevancy": [], "faithfulness": [], "correctness": []}
    categories: dict[str, list[float]] = {}

    for r in results:
        metrics["relevancy"].append(r["relevancy"]["score"])
        metrics["faithfulness"].append(r["faithfulness"]["score"])
        metrics["correctness"].append(r["correctness"]["score"])

        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["correctness"]["score"])

    overall = {k: np.mean(v) for k, v in metrics.items()}
    by_category = {k: np.mean(v) for k, v in sorted(categories.items())}
    return {"overall": overall, "by_category": by_category}


def plot_overall_comparison(rag_summary, base_summary):
    """Fig 1: Overall metric comparison (RAG vs Baseline)."""
    apply_style()
    metrics = ["Relevancy", "Faithfulness", "Correctness"]
    rag_vals = [rag_summary["overall"]["relevancy"],
                rag_summary["overall"]["faithfulness"],
                rag_summary["overall"]["correctness"]]
    base_vals = [base_summary["overall"]["relevancy"],
                 base_summary["overall"]["faithfulness"],
                 base_summary["overall"]["correctness"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars1 = ax.bar(x - width / 2, rag_vals, width, label="RAG", color=COLOR_PRIMARY)
    bars2 = ax.bar(x + width / 2, base_vals, width, label="Baseline (LLM-only)", color=COLOR_ACCENT)

    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc="upper right")
    ax.set_title("Overall Evaluation Metrics: RAG vs Baseline")

    fig.savefig(OUTPUT_DIR / "overall_comparison.pdf")
    plt.close(fig)
    print("Saved overall_comparison.pdf")


def plot_category_correctness(rag_summary, base_summary):
    """Fig 2: Per-category correctness comparison."""
    apply_style()
    categories = list(rag_summary["by_category"].keys())
    labels = [c.replace("_", " ").title() for c in categories]
    rag_vals = [rag_summary["by_category"][c] for c in categories]
    base_vals = [base_summary["by_category"][c] for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars1 = ax.bar(x - width / 2, rag_vals, width, label="RAG", color=COLOR_PRIMARY)
    bars2 = ax.bar(x + width / 2, base_vals, width, label="Baseline", color=COLOR_ACCENT)

    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)

    ax.set_ylabel("Correctness Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_title("Correctness by Question Category")

    fig.savefig(OUTPUT_DIR / "category_correctness.pdf")
    plt.close(fig)
    print("Saved category_correctness.pdf")


def plot_faithfulness_comparison(rag_summary, base_summary):
    """Fig 3: Faithfulness and correctness side by side."""
    apply_style()
    labels = ["Faithfulness", "Correctness"]
    rag_vals = [rag_summary["overall"]["faithfulness"],
                rag_summary["overall"]["correctness"]]
    base_vals = [base_summary["overall"]["faithfulness"],
                 base_summary["overall"]["correctness"]]
    deltas = [r - b for r, b in zip(rag_vals, base_vals)]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars1 = ax.bar(x - width, rag_vals, width, label="RAG", color=COLOR_PRIMARY)
    bars2 = ax.bar(x, base_vals, width, label="Baseline", color=COLOR_ACCENT)
    bars3 = ax.bar(x + width, deltas, width, label="Improvement ($\\Delta$)", color=COLOR_GOOD)

    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)
    add_bar_labels(ax, bars3)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.set_title("RAG Impact on Faithfulness and Correctness")

    fig.savefig(OUTPUT_DIR / "faithfulness_impact.pdf")
    plt.close(fig)
    print("Saved faithfulness_impact.pdf")


def plot_per_question_delta(rag_results, base_results):
    """Fig 4: Per-question correctness delta (horizontal bar)."""
    apply_style()
    base_map = {r["id"]: r["correctness"]["score"] for r in base_results}

    ids = []
    deltas = []
    colors = []
    for r in rag_results:
        qid = r["id"]
        rag_score = r["correctness"]["score"]
        base_score = base_map.get(qid, 0.0)
        delta = rag_score - base_score
        ids.append(qid)
        deltas.append(delta)
        colors.append(COLOR_GOOD if delta >= 0 else COLOR_BAD)

    fig, ax = plt.subplots(figsize=(7, 12))
    y_pos = np.arange(len(ids))
    ax.barh(y_pos, deltas, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([latex_safe(i) for i in ids], fontsize=7)
    ax.set_xlabel("Correctness Delta (RAG $-$ Baseline)")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_title("Per-Question Correctness Improvement")
    ax.invert_yaxis()

    fig.savefig(OUTPUT_DIR / "per_question_delta.pdf")
    plt.close(fig)
    print("Saved per_question_delta.pdf")


def plot_tool_usage(rag_results):
    """Fig 5: Tool usage frequency across all questions."""
    apply_style()
    tool_counts: dict[str, int] = {}
    for r in rag_results:
        for tc in r.get("tool_calls", []):
            name = tc["name"]
            tool_counts[name] = tool_counts.get(name, 0) + 1

    tools = sorted(tool_counts.keys())
    counts = [tool_counts[t] for t in tools]
    labels = [latex_safe(t) for t in tools]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars = ax.bar(labels, counts, color=[PALETTE[i % len(PALETTE)] for i in range(len(tools))])
    add_bar_labels(ax, bars, fmt="{:.0f}")
    ax.set_ylabel("Number of Invocations")
    ax.set_title("Tool Usage Distribution Across Evaluation Set")

    fig.savefig(OUTPUT_DIR / "tool_usage.pdf")
    plt.close(fig)
    print("Saved tool_usage.pdf")


def plot_score_distribution(rag_results, base_results):
    """Fig 6: Distribution of correctness scores."""
    apply_style()
    rag_scores = [r["correctness"]["score"] for r in rag_results]
    base_scores = [r["correctness"]["score"] for r in base_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)

    bins = [-0.1, 0.15, 0.35, 0.65, 0.85, 1.1]
    bin_labels = ["0.0", "0.25", "0.5", "0.75", "1.0"]

    counts_rag, _ = np.histogram(rag_scores, bins=bins)
    counts_base, _ = np.histogram(base_scores, bins=bins)

    x = np.arange(len(bin_labels))
    bars1 = ax1.bar(x, counts_rag, color=COLOR_PRIMARY)
    add_bar_labels(ax1, bars1, fmt="{:.0f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels)
    ax1.set_xlabel("Correctness Score")
    ax1.set_ylabel("Number of Questions")
    ax1.set_title("RAG System")

    bars2 = ax2.bar(x, counts_base, color=COLOR_ACCENT)
    add_bar_labels(ax2, bars2, fmt="{:.0f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels)
    ax2.set_xlabel("Correctness Score")
    ax2.set_title("Baseline (LLM-only)")

    fig.suptitle("Distribution of Correctness Scores", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "score_distribution.pdf")
    plt.close(fig)
    print("Saved score_distribution.pdf")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rag_results = load_results("rag")
    base_results = load_results("baseline")
    rag_summary = compute_summary(rag_results)
    base_summary = compute_summary(base_results)

    print(f"RAG results: {len(rag_results)} questions")
    print(f"Baseline results: {len(base_results)} questions")
    print(f"RAG overall: {rag_summary['overall']}")
    print(f"Baseline overall: {base_summary['overall']}")

    plot_overall_comparison(rag_summary, base_summary)
    plot_category_correctness(rag_summary, base_summary)
    plot_faithfulness_comparison(rag_summary, base_summary)
    plot_per_question_delta(rag_results, base_results)
    plot_tool_usage(rag_results)
    plot_score_distribution(rag_results, base_results)

    print(f"\nAll plots saved to {OUTPUT_DIR}")
