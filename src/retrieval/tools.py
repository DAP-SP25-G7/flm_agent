"""LangChain tool definitions for the curriculum RAG agent.

Four tools:
1. vector_search     — semantic search over course/curriculum documents
2. prerequisite_lookup — deterministic prerequisite chain resolver
3. combo_navigator    — specialization combo/pathway browser
4. curriculum_browser — semester-by-semester course listing
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from src.config import PROCESSED_DIR, settings
from src.retrieval.vector_store import get_vector_store


# JSON data loaders

@lru_cache(maxsize=1)
def _load_course_index() -> dict:
    return json.loads((PROCESSED_DIR / "course_index.json").read_text("utf-8"))


@lru_cache(maxsize=1)
def _load_prerequisites() -> dict[str, str]:
    return json.loads((PROCESSED_DIR / "prerequisites.json").read_text("utf-8"))


@lru_cache(maxsize=1)
def _load_curriculum_map() -> dict[str, list[dict]]:
    return json.loads((PROCESSED_DIR / "curriculum_map.json").read_text("utf-8"))


@lru_cache(maxsize=1)
def _load_combo_map() -> dict[str, dict]:
    return json.loads((PROCESSED_DIR / "combo_map.json").read_text("utf-8"))


@lru_cache(maxsize=1)
def _load_plos() -> dict[str, str]:
    return json.loads(
        (PROCESSED_DIR / "program_learning_outcomes.json").read_text("utf-8")
    )


@lru_cache(maxsize=1)
def _load_programme_outcomes() -> dict[str, str]:
    return json.loads(
        (PROCESSED_DIR / "programme_outcomes.json").read_text("utf-8")
    )


def _resolve_course_code(raw: str, index: dict) -> tuple[str, str | None]:
    """Resolve a possibly partial or inexact course code to a known one.

    Returns (resolved_code, error_message). If the code is found exactly,
    error_message is None. If a unique prefix match is found, returns
    that match with None error. If ambiguous or not found, returns
    (raw, error_string) with suggestions.
    """
    code = raw.strip()
    if code in index:
        return code, None

    upper = code.upper()
    if upper in index:
        return upper, None

    prefix_matches = [c for c in index if c.upper().startswith(upper)]
    if len(prefix_matches) == 1:
        return prefix_matches[0], None
    if prefix_matches:
        listed = ", ".join(sorted(prefix_matches))
        return code, f"'{code}' is ambiguous. Did you mean one of: {listed}?"

    return code, f"Course '{code}' not found in the curriculum."


# Tool 1: Vector Search

@tool
def vector_search(query: str, course_code: str = "", entity_type: str = "") -> str:
    """Search course syllabi, curriculum, and pathway documents by semantic similarity.

    Use this for questions about: course content and descriptions, learning
    outcomes, teaching methods, assessment details, session topics, materials,
    or any open-ended questions about what a course covers.

    Args:
        query: The search query describing what information you need.
        course_code: Optional course code to scope results (e.g., "MAD101", "DPL302m").
                     Always provide this when asking about a specific course.
        entity_type: Optional filter — "syllabus", "curriculum", or "pathway".
    """
    store = get_vector_store()

    filter_dict = {}
    if course_code:
        resolved, err = _resolve_course_code(course_code, _load_course_index())
        if err:
            return err
        filter_dict["course_code"] = resolved
    if entity_type and entity_type in ("syllabus", "curriculum", "pathway"):
        filter_dict["entity_type"] = entity_type

    results = store.similarity_search(
        query,
        k=settings.top_k,
        filter=filter_dict if filter_dict else None,
    )

    if not results:
        return "No relevant documents found."

    parts: list[str] = []
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        source = meta.get("source_file", "unknown")
        section = meta.get("section", "unknown")
        course = meta.get("course_code", "")
        header = f"[{i}] {source} > {section}"
        if course:
            header += f" (Course: {course})"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


# Tool 2: Prerequisite Lookup

def _resolve_prereq_chain(
    code: str,
    prereqs: dict[str, str],
    index: dict,
    visited: set | None = None,
) -> list[dict]:
    """Recursively resolve the full prerequisite chain for a course."""
    if visited is None:
        visited = set()
    if code in visited:
        return []
    visited.add(code)

    direct = prereqs.get(code, "")
    if not direct:
        return []

    chain: list[dict] = [{"course": code, "requires": direct}]

    # Extract course codes from the prerequisite string
    # Match patterns like PFP191, AIL303m, DPL30x, etc.
    import re
    codes_in_prereq = re.findall(r"[A-Z]{2,4}\d{2,3}[a-z]?", direct)

    for dep_code in codes_in_prereq:
        if dep_code in prereqs:
            chain.extend(_resolve_prereq_chain(dep_code, prereqs, index, visited))

    return chain


@tool
def prerequisite_lookup(
    course_code: str,
    direction: str = "forward",
) -> str:
    """Look up prerequisites for a course or find courses unlocked by completing one.

    Use this for questions like: "What do I need before taking X?",
    "What's the full prerequisite chain for X?",
    "What courses can I take after completing X?"

    Args:
        course_code: The course code (e.g., "AIL303m", "NLP301c", "DPL302m").
        direction: "forward" to find what you need before taking this course,
                   "reverse" to find what courses this unlocks.
    """
    prereqs = _load_prerequisites()
    index = _load_course_index()
    code, err = _resolve_course_code(course_code, index)
    if err:
        return err

    if direction == "forward":
        if code not in prereqs:
            course_name = index.get(code, {}).get("name", code)
            return f"{code} ({course_name}) has no prerequisites."

        # Direct prerequisite
        direct = prereqs[code]
        course_name = index.get(code, {}).get("name", code)
        result = f"**Prerequisites for {code} ({course_name}):**\n"
        result += f"Direct: {direct}\n\n"

        # Full chain
        chain = _resolve_prereq_chain(code, prereqs, index)
        if len(chain) > 1:
            result += "**Full prerequisite chain:**\n"
            for step in chain:
                name = index.get(step["course"], {}).get("name", step["course"])
                result += f"- {step['course']} ({name}) requires: {step['requires']}\n"

        return result

    else:  # reverse
        course_name = index.get(code, {}).get("name", code)
        unlocked: list[str] = []
        for c, req_str in prereqs.items():
            if code in req_str:
                c_name = index.get(c, {}).get("name", c)
                unlocked.append(f"- {c} ({c_name}): requires {req_str}")

        if not unlocked:
            return f"No courses list {code} ({course_name}) as a prerequisite."

        result = f"**Courses unlocked by completing {code} ({course_name}):**\n"
        result += "\n".join(unlocked)
        return result


# Tool 3: Combo Navigator

@tool
def combo_navigator(combo_id: str = "", topic: str = "") -> str:
    """Browse specialization combos, elective pathways, and their course options.

    Use this for questions about: which combos/specializations are available,
    what courses are in a specific combo, elective options like TMI_ELE.

    Args:
        combo_id: Optional pathway ID (e.g., "AI17_COM_1", "TMI_ELE").
                  Leave empty to list all available pathways.
        topic: Optional topic name filter (e.g., "Applied Data Science").
    """
    combos = _load_combo_map()

    if not combo_id and not topic:
        # List all pathways
        result = "**Available pathways/combos:**\n\n"
        for pid, info in combos.items():
            desc = info.get("description", "")
            sem = info.get("semester", "")
            n_topics = len(info.get("topics", {}))
            result += f"- **{pid}**: {desc} (Semester {sem}, {n_topics} topic groups)\n"
        return result

    if combo_id:
        combo_id = combo_id.strip()
        if combo_id not in combos:
            return f"Pathway '{combo_id}' not found. Available: {', '.join(combos.keys())}"

        info = combos[combo_id]
        result = f"**Pathway: {combo_id}**\n"
        result += f"Description: {info.get('description', 'N/A')}\n"
        result += f"Semester: {info.get('semester', 'N/A')}\n"
        result += f"Curriculum: {info.get('curriculum', 'N/A')}\n\n"

        for topic_name, courses in info.get("topics", {}).items():
            result += f"### {topic_name}\n"
            for c in courses:
                status = "Active" if c.get("is_active") else "Inactive"
                result += f"- {c['code']} — {c['name']} ({status})\n"
            result += "\n"
        return result

    # Search by topic name
    results: list[str] = []
    for pid, info in combos.items():
        for topic_name, courses in info.get("topics", {}).items():
            if topic.lower() in topic_name.lower():
                lines = [f"**{pid} > {topic_name}:**"]
                for c in courses:
                    lines.append(f"- {c['code']} — {c['name']}")
                results.append("\n".join(lines))

    if not results:
        return f"No topics matching '{topic}' found."
    return "\n\n".join(results)


# Tool 4: Curriculum Browser

@tool
def curriculum_browser(semester: str = "", course_code: str = "") -> str:
    """Browse the K20-K21 curriculum by semester or look up specific course details.

    Use this for questions like: "What courses are in semester 3?",
    "How many credits is the program?", "Tell me about course PFP191",
    "What's the study plan?"

    Args:
        semester: Semester number ("0"-"9") or "all" for a full overview.
                  Leave empty if looking up a specific course.
        course_code: Optional specific course code to look up.
    """
    cmap = _load_curriculum_map()
    index = _load_course_index()

    if course_code:
        code, err = _resolve_course_code(course_code, index)
        if err:
            return err
        info = index[code]
        result = f"**{code} — {info.get('name', 'N/A')}**\n"
        result += f"- Credits: {info.get('credits', 'N/A')}\n"
        result += f"- Semester: {info.get('semester', 'N/A')}\n"
        prereq = info.get("prerequisite", "")
        result += f"- Prerequisites: {prereq if prereq else 'None'}\n"
        desc = info.get("description", "")
        if desc:
            result += f"- Description: {desc[:500]}{'...' if len(desc) > 500 else ''}\n"

        # PLO alignment from course index won't have this, but we can note it
        return result

    if semester and semester != "all":
        sem = semester.strip()
        if sem not in cmap:
            return f"Semester '{sem}' not found. Available: {', '.join(sorted(cmap.keys(), key=lambda x: int(x)))}"

        courses = cmap[sem]
        total_credits = sum(int(c.get("credits", 0)) for c in courses)
        result = f"**Semester {sem}** ({len(courses)} courses, {total_credits} credits):\n\n"
        for c in courses:
            prereq = c.get("prerequisite", "")
            prereq_str = f" [Requires: {prereq}]" if prereq else ""
            result += f"- **{c['code']}** — {c['name']} ({c['credits']} credits){prereq_str}\n"
        return result

    # Full overview
    result = "**K20-K21 Curriculum Overview (BIT_AI)**\n\n"
    total_credits = 0
    total_courses = 0
    for sem in sorted(cmap.keys(), key=lambda x: int(x)):
        courses = cmap[sem]
        sem_credits = sum(int(c.get("credits", 0)) for c in courses)
        total_credits += sem_credits
        total_courses += len(courses)
        result += f"**Semester {sem}:** {len(courses)} courses, {sem_credits} credits\n"
        for c in courses:
            result += f"  - {c['code']} — {c['name']} ({c['credits']}cr)\n"
        result += "\n"

    result += f"\n**Total: {total_courses} courses, {total_credits} credits**"
    return result


ALL_TOOLS = [vector_search, prerequisite_lookup, combo_navigator, curriculum_browser]
