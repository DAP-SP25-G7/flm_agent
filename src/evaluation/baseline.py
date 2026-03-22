"""No-RAG baseline runner — LLM-only answers without retrieval."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config import settings

BASELINE_SYSTEM_PROMPT = """\
You are an academic advisor for FPT University's Bachelor of IT \
in Artificial Intelligence (BIT_AI) program, curriculum version K20-K21. \
Answer the student's question to the best of your knowledge. \
If you don't know, say so.\
"""


def _get_baseline_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )


def run_baseline(question: str) -> str:
    """Get an LLM-only answer (no tools, no retrieval)."""
    llm = _get_baseline_llm()
    messages = [
        SystemMessage(content=BASELINE_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    return response.content


def run_baseline_batch(questions: list[str]) -> list[str]:
    """Run baseline on a list of questions. Returns answers in order."""
    llm = _get_baseline_llm()
    answers: list[str] = []
    for i, q in enumerate(questions):
        logger.info(f"Baseline [{i + 1}/{len(questions)}]: {q[:60]}...")
        messages = [
            SystemMessage(content=BASELINE_SYSTEM_PROMPT),
            HumanMessage(content=q),
        ]
        resp = llm.invoke(messages)
        answers.append(resp.content)
    return answers
