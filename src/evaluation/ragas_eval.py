"""Evaluation runner using LLM-as-judge for faithfulness, relevancy, and correctness.

We implement lightweight LLM-based metrics rather than importing RAGAS directly,
keeping dependencies minimal and giving full control over prompts.

Metrics:
  - **answer_relevancy**: Does the answer address the question?
  - **faithfulness**: Is the answer grounded in the retrieved context?
  - **correctness**: Does the answer match the reference answer?
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from src.chain.agent import AgentResponse, chat
from src.config import EVALUATION_DIR, settings
from src.evaluation.golden_set import GoldenQA

# Metric prompts

_RELEVANCY_PROMPT = """\
Rate how well the answer addresses the question on a scale of 0 to 1.
- 1.0: Fully addresses the question with specific, accurate details.
- 0.5: Partially addresses the question or is vague.
- 0.0: Does not address the question at all.

**Question:** {question}

**Answer:** {answer}

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief reason>"}}\
"""

_FAITHFULNESS_PROMPT = """\
Rate how well the answer is grounded in the provided context on a scale of 0 to 1.
- 1.0: Every claim in the answer is supported by the context.
- 0.5: Some claims are supported, others are not verifiable from the context.
- 0.0: The answer contains claims not found in the context, or context is empty.

If no context/tool results are provided, score 0.0.

**Question:** {question}

**Context (tool results):**
{context}

**Answer:** {answer}

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief reason>"}}\
"""

_CORRECTNESS_PROMPT = """\
Rate how well the generated answer matches the reference answer on a scale of 0 to 1.
- 1.0: Conveys the same key facts and details as the reference.
- 0.5: Partially correct — captures some key facts but misses others.
- 0.0: Incorrect or contradicts the reference answer.

Focus on factual alignment, not exact wording.

**Question:** {question}

**Reference answer:** {reference}

**Generated answer:** {answer}

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief reason>"}}\
"""


# Data structures

@dataclass
class MetricScore:
    """Score for a single metric on a single question."""

    score: float
    reason: str


@dataclass
class EvalResult:
    """Full evaluation result for a single question."""

    id: str
    category: str
    question: str
    reference_answer: str
    generated_answer: str
    tool_calls: list[dict] = field(default_factory=list)
    relevancy: MetricScore | None = None
    faithfulness: MetricScore | None = None
    correctness: MetricScore | None = None


@dataclass
class EvalSummary:
    """Aggregated evaluation summary."""

    total_questions: int
    avg_relevancy: float
    avg_faithfulness: float
    avg_correctness: float
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)


# Judge LLM

def _get_judge_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )


def _judge(prompt: str) -> dict:
    """Call the judge LLM and parse the JSON response."""
    llm = _get_judge_llm()
    resp = llm.invoke([
        SystemMessage(content="You are an evaluation judge. Respond only with valid JSON."),
        HumanMessage(content=prompt),
    ])
    text = resp.content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Judge returned invalid JSON: {text}")
        return {"score": 0.0, "reason": "Failed to parse judge response"}


# Core evaluation

def evaluate_single(
    qa: GoldenQA, use_rag: bool = True
) -> EvalResult:
    """Evaluate a single question with the RAG agent or baseline."""
    if use_rag:
        response: AgentResponse = chat(question=qa.question)
        answer = response.content
        tool_calls = response.tool_calls
    else:
        from src.evaluation.baseline import run_baseline

        answer = run_baseline(qa.question)
        tool_calls = []

    # Build context from tool results
    context = "\n---\n".join(
        f"[{tc['name']}]: {tc['result']}" for tc in tool_calls
    ) if tool_calls else "(no context — LLM-only)"

    # Score with judge
    relevancy = _judge(_RELEVANCY_PROMPT.format(
        question=qa.question, answer=answer,
    ))
    faithfulness = _judge(_FAITHFULNESS_PROMPT.format(
        question=qa.question, context=context, answer=answer,
    ))
    correctness = _judge(_CORRECTNESS_PROMPT.format(
        question=qa.question, reference=qa.reference_answer, answer=answer,
    ))

    return EvalResult(
        id=qa.id,
        category=qa.category,
        question=qa.question,
        reference_answer=qa.reference_answer,
        generated_answer=answer,
        tool_calls=tool_calls,
        relevancy=MetricScore(**relevancy),
        faithfulness=MetricScore(**faithfulness),
        correctness=MetricScore(**correctness),
    )


def evaluate_batch(
    questions: list[GoldenQA],
    use_rag: bool = True,
    label: str = "rag",
) -> list[EvalResult]:
    """Evaluate a batch of golden Q&A entries."""
    results: list[EvalResult] = []
    mode = "RAG" if use_rag else "Baseline"
    for i, qa in enumerate(questions):
        logger.info(f"[{mode}] Evaluating [{i + 1}/{len(questions)}] {qa.id}: {qa.question[:50]}...")
        try:
            result = evaluate_single(qa, use_rag=use_rag)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate {qa.id}: {e}")
            results.append(EvalResult(
                id=qa.id,
                category=qa.category,
                question=qa.question,
                reference_answer=qa.reference_answer,
                generated_answer=f"ERROR: {e}",
            ))

    # Save raw results
    output_dir = EVALUATION_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{label}_{timestamp}.json"

    serializable = []
    for r in results:
        d = {
            "id": r.id,
            "category": r.category,
            "question": r.question,
            "reference_answer": r.reference_answer,
            "generated_answer": r.generated_answer,
            "tool_calls": [
                {"name": tc["name"], "args": tc["args"]}
                for tc in r.tool_calls
            ],
            "relevancy": asdict(r.relevancy) if r.relevancy else None,
            "faithfulness": asdict(r.faithfulness) if r.faithfulness else None,
            "correctness": asdict(r.correctness) if r.correctness else None,
        }
        serializable.append(d)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")

    return results


def summarize(results: list[EvalResult]) -> EvalSummary:
    """Compute aggregate metrics from evaluation results."""
    scored = [r for r in results if r.relevancy is not None]
    n = len(scored)
    if n == 0:
        return EvalSummary(0, 0.0, 0.0, 0.0)

    avg_rel = sum(r.relevancy.score for r in scored) / n
    avg_faith = sum(r.faithfulness.score for r in scored) / n
    avg_corr = sum(r.correctness.score for r in scored) / n

    # By category
    categories: dict[str, list[EvalResult]] = {}
    for r in scored:
        categories.setdefault(r.category, []).append(r)

    by_cat = {}
    for cat, cat_results in categories.items():
        cn = len(cat_results)
        by_cat[cat] = {
            "count": cn,
            "avg_relevancy": sum(r.relevancy.score for r in cat_results) / cn,
            "avg_faithfulness": sum(r.faithfulness.score for r in cat_results) / cn,
            "avg_correctness": sum(r.correctness.score for r in cat_results) / cn,
        }

    return EvalSummary(
        total_questions=n,
        avg_relevancy=avg_rel,
        avg_faithfulness=avg_faith,
        avg_correctness=avg_corr,
        by_category=by_cat,
    )
