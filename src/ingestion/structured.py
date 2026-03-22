"""Build structured JSON lookup tables from parsed documents.

Produces four files in data/processed/:
- course_index.json   — flat lookup {code: {name, credits, semester, ...}}
- prerequisites.json  — {code: raw_prerequisite_string}
- curriculum_map.json — {semester: [{code, name, credits, prerequisite}]}
- combo_map.json      — {pathway_id: {description, semester, topics: {name: [courses]}}}
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from src.ingestion.parser import (
    EntityType,
    ParsedDocument,
    _parse_data_table,
    _parse_kv_table,
    parse_curriculum_details,
)


def build_course_index(
    curriculum_doc: ParsedDocument,
    syllabus_docs: list[ParsedDocument],
) -> dict:
    """Build a flat course index from the curriculum and syllabi.

    Returns:
        {course_code: {name, credits, semester, description, prerequisite}}
    """
    index: dict[str, dict] = {}

    # Extract from curriculum details table
    if "Curriculum details" in curriculum_doc.sections:
        rows = parse_curriculum_details(curriculum_doc.sections["Curriculum details"])
        for row in rows:
            code = row.get("SubjectCode", "").strip()
            if not code:
                continue
            index[code] = {
                "name": row.get("Subject Name", "").strip(),
                "credits": row.get("NoCredit", "").strip(),
                "semester": row.get("Semester", "").strip(),
                "prerequisite": row.get("PreRequisite", "").strip(),
            }

    # Enrich with syllabus descriptions
    for doc in syllabus_docs:
        code = doc.general_info.get("Subject Code", "").strip()
        if not code:
            continue
        if code not in index:
            index[code] = {
                "name": doc.general_info.get("Syllabus Name", "").strip(),
                "credits": doc.general_info.get("NoCredit", "").strip(),
                "semester": "",
                "prerequisite": doc.general_info.get("Pre-Requisite", "").strip(),
            }
        entry = index[code]
        entry["description"] = doc.general_info.get("Description", "").strip()
        # Prefer syllabus prerequisite (more detailed) over curriculum's
        syllabus_prereq = doc.general_info.get("Pre-Requisite", "").strip()
        if syllabus_prereq and syllabus_prereq.lower() not in ("", "none"):
            entry["prerequisite"] = syllabus_prereq

    return index


def build_prerequisites(course_index: dict) -> dict[str, str]:
    """Extract prerequisite mapping from the course index.

    Returns:
        {course_code: prerequisite_string} for courses that have prerequisites.
    """
    _no_prereq = {"none", "không", "no", "không none", ""}
    prereqs: dict[str, str] = {}
    for code, info in course_index.items():
        prereq = info.get("prerequisite", "").strip()
        if prereq and prereq.lower() not in _no_prereq:
            prereqs[code] = prereq
    return prereqs


def build_curriculum_map(curriculum_doc: ParsedDocument) -> dict[str, list[dict]]:
    """Build semester-grouped course listing.

    Returns:
        {semester_number: [{code, name, credits, prerequisite}]}
    """
    semester_map: dict[str, list[dict]] = {}

    if "Curriculum details" not in curriculum_doc.sections:
        logger.warning("No 'Curriculum details' section found.")
        return semester_map

    rows = parse_curriculum_details(curriculum_doc.sections["Curriculum details"])
    for row in rows:
        semester = row.get("Semester", "").strip()
        code = row.get("SubjectCode", "").strip()
        if not semester or not code:
            continue

        if semester not in semester_map:
            semester_map[semester] = []

        semester_map[semester].append({
            "code": code,
            "name": row.get("Subject Name", "").strip(),
            "credits": row.get("NoCredit", "").strip(),
            "prerequisite": row.get("PreRequisite", "").strip(),
        })

    return semester_map


def build_combo_map(pathway_docs: list[ParsedDocument]) -> dict[str, dict]:
    """Build combo/pathway mapping.

    Returns:
        {pathway_id: {
            description: str,
            semester: str,
            curriculum: str,
            topics: {topic_name: [{code, name, is_active, is_approved}]}
        }}
    """
    combo_map: dict[str, dict] = {}

    for doc in pathway_docs:
        pathway_id = doc.entity_id
        general = doc.general_info

        entry: dict = {
            "description": general.get("Description", "").strip(),
            "semester": general.get("Semester", "").strip(),
            "curriculum": general.get("Curriculum", "").strip(),
            "topics": {},
        }

        # Each non-"General information" section is a topic with course options
        for section_name, section_text in doc.sections.items():
            if section_name == "General information":
                continue

            # Handle "Elective options" section (e.g., TMI_ELE)
            courses = _parse_data_table(section_text)
            topic_courses = []
            for course in courses:
                topic_courses.append({
                    "code": course.get("Subject Code", "").strip(),
                    "name": course.get("Syllabus Name", "").strip(),
                    "is_active": course.get("IsActive", "").strip().lower() == "yes",
                    "is_approved": course.get("IsApproved", "").strip().lower() == "yes",
                })

            if topic_courses:
                entry["topics"][section_name] = topic_courses

        combo_map[pathway_id] = entry

    return combo_map


def build_programme_outcomes(curriculum_doc: ParsedDocument) -> dict[str, str]:
    """Extract Programme Outcomes (POs) from curriculum.

    The table format is: | id | PO_code | description |
    (3 columns, no named headers matching standard patterns)

    Returns:
        {po_code: description}
    """
    outcomes: dict[str, str] = {}
    if "Programme outcomes" in curriculum_doc.sections:
        rows = _parse_data_table(curriculum_doc.sections["Programme outcomes"])
        for row in rows:
            # The table columns vary; try all plausible column names
            values = list(row.values())
            if len(values) >= 3:
                # Format: id_number | PO_code | description
                po_code = values[1].strip()
                desc = values[2].strip()
                if po_code:
                    outcomes[po_code] = desc
            elif len(values) == 2:
                po_code = values[0].strip()
                desc = values[1].strip()
                if po_code:
                    outcomes[po_code] = desc
    return outcomes


def build_program_learning_outcomes(curriculum_doc: ParsedDocument) -> dict[str, str]:
    """Extract Program Learning Outcomes (PLOs) from curriculum.

    The table format is: | number | PLO Name | PLO Description |

    Returns:
        {plo_name: description}
    """
    outcomes: dict[str, str] = {}
    if "Program learning outcomes" in curriculum_doc.sections:
        rows = _parse_data_table(curriculum_doc.sections["Program learning outcomes"])
        for row in rows:
            plo_name = row.get("PLO Name", "").strip()
            desc = row.get("PLO Description", "").strip()
            if plo_name:
                outcomes[plo_name] = desc
    return outcomes


def build_all(
    documents: list[ParsedDocument],
    output_dir: Path,
) -> dict[str, Path]:
    """Build all structured lookup tables and write them to output_dir.

    Returns a dict mapping table name to output file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate documents by type
    curricula = [d for d in documents if d.entity_type == EntityType.CURRICULUM]
    syllabi = [d for d in documents if d.entity_type == EntityType.SYLLABUS]
    pathways = [d for d in documents if d.entity_type == EntityType.PATHWAY]

    # Use K20-K21 curriculum (latest) — match by filename since entity_id
    # from H1 header can be ambiguous (e.g. "19D-20A" also contains "20")
    curriculum = None
    for c in curricula:
        if "k20_k21" in c.file_path.stem.lower():
            curriculum = c
            break
    if curriculum is None and curricula:
        curriculum = curricula[-1]

    if curriculum is None:
        logger.error("No curriculum document found.")
        return {}

    logger.info(f"Using curriculum: {curriculum.entity_id}")

    # Build tables
    course_index = build_course_index(curriculum, syllabi)
    prerequisites = build_prerequisites(course_index)
    curriculum_map = build_curriculum_map(curriculum)
    combo_map = build_combo_map(pathways)
    programme_outcomes = build_programme_outcomes(curriculum)
    plos = build_program_learning_outcomes(curriculum)

    # Write to JSON
    outputs: dict[str, Path] = {}
    tables = {
        "course_index": course_index,
        "prerequisites": prerequisites,
        "curriculum_map": curriculum_map,
        "combo_map": combo_map,
        "programme_outcomes": programme_outcomes,
        "program_learning_outcomes": plos,
    }

    for name, data in tables.items():
        out_path = output_dir / f"{name}.json"
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs[name] = out_path
        logger.info(f"Wrote {name}: {len(data)} entries -> {out_path}")

    return outputs
