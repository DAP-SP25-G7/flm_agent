"""Load and manage the golden Q&A evaluation test set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import EVALUATION_DIR

GOLDEN_QA_PATH = EVALUATION_DIR / "golden_qa.json"


@dataclass
class GoldenQA:
    """A single golden Q&A entry."""

    id: str
    category: str
    question: str
    reference_answer: str
    expected_tool: str
    expected_sources: list[str]


def load_golden_set(path: Path = GOLDEN_QA_PATH) -> list[GoldenQA]:
    """Load the golden Q&A set from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [GoldenQA(**item) for item in data]


def load_by_category(
    category: str, path: Path = GOLDEN_QA_PATH
) -> list[GoldenQA]:
    """Load golden Q&A entries filtered by category."""
    return [q for q in load_golden_set(path) if q.category == category]


def get_categories(path: Path = GOLDEN_QA_PATH) -> list[str]:
    """Return sorted list of unique categories in the golden set."""
    return sorted({q.category for q in load_golden_set(path)})
