"""Markdown parser for curriculum, syllabus, and pathway files.

Reads raw markdown files, classifies their entity type, and extracts
structured fields from their tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

# Entity classification regexes
RE_H1_SYLLABUS = re.compile(r"^# Syllabus:\s*(.*)", re.MULTILINE)
RE_H1_CURRICULUM = re.compile(r"^# Curriculum:\s*(.*)", re.MULTILINE)
RE_H1_PATHWAY = re.compile(r"^# Pathway:\s*(.*)", re.MULTILINE)
RE_H2 = re.compile(r"^##\s+(.*)", re.MULTILINE)

# Matches markdown table rows: | key | value | ...
RE_TABLE_ROW = re.compile(r"^\|(.+)\|$", re.MULTILINE)
# Separator rows like |---|---|
RE_SEPARATOR = re.compile(r"^[\s|:-]+$")


class EntityType(str, Enum):
    SYLLABUS = "syllabus"
    CURRICULUM = "curriculum"
    PATHWAY = "pathway"


@dataclass
class ParsedDocument:
    """Represents a parsed markdown document with its metadata and sections."""

    file_path: Path
    entity_type: EntityType
    entity_id: str  # e.g. "PFP191", "BIT_AI_K20D-21A", "AI17_COM_1"
    general_info: dict[str, str] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)  # section_name -> raw text
    raw_content: str = ""


def classify_entity(content: str) -> tuple[EntityType, str] | None:
    """Classify a markdown file by its H1 header and extract entity ID."""
    m = RE_H1_SYLLABUS.search(content)
    if m:
        return EntityType.SYLLABUS, m.group(1).strip()

    m = RE_H1_CURRICULUM.search(content)
    if m:
        return EntityType.CURRICULUM, m.group(1).strip()

    m = RE_H1_PATHWAY.search(content)
    if m:
        return EntityType.PATHWAY, m.group(1).strip()

    return None


def _split_sections(content: str) -> dict[str, str]:
    """Split markdown content into sections by H2 headers."""
    headers = list(RE_H2.finditer(content))
    sections: dict[str, str] = {}

    for i, match in enumerate(headers):
        name = match.group(1).strip()
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
        sections[name] = content[start:end].strip()

    return sections


def _parse_kv_table(text: str) -> dict[str, str]:
    """Parse a key-value markdown table (| key: | value |) into a dict.

    Handles both formats:
    - ``| Key: | Value |`` (syllabus general info)
    - ``| Key | Value |`` (pathway general info)
    """
    result: dict[str, str] = {}
    for row_match in RE_TABLE_ROW.finditer(text):
        cells = [c.strip() for c in row_match.group(1).split("|")]
        if len(cells) >= 2 and not RE_SEPARATOR.match(row_match.group(0)):
            key = cells[0].rstrip(":").strip()
            value = cells[1].strip()
            if key and not RE_SEPARATOR.match(cells[0]):
                result[key] = value
    return result


def _parse_data_table(text: str) -> list[dict[str, str]]:
    """Parse a standard markdown data table into a list of row dicts.

    Expects format:
    | Header1 | Header2 | ...
    |---------|---------|---
    | val1    | val2    | ...
    """
    rows_raw = RE_TABLE_ROW.findall(text)
    if len(rows_raw) < 3:  # need header + separator + at least one data row
        return []

    # Keep ALL columns including empty-header ones, to maintain column alignment.
    # The regex captures between first and last |, so split gives correct cells.
    raw_headers = [h.strip() for h in rows_raw[0].split("|")]

    data_rows: list[dict[str, str]] = []
    for raw_row in rows_raw[2:]:  # skip header and separator
        if RE_SEPARATOR.match("|" + raw_row + "|"):
            continue
        raw_cells = [c.strip() for c in raw_row.split("|")]

        # Build dict, skipping columns with empty headers (row-number columns)
        row_dict = {}
        for j, header in enumerate(raw_headers):
            if header:  # only include named columns
                row_dict[header] = raw_cells[j] if j < len(raw_cells) else ""
        if row_dict:
            data_rows.append(row_dict)

    return data_rows


def parse_curriculum_details(text: str) -> list[dict[str, str]]:
    """Parse the 'Curriculum details' section into a list of course entries."""
    return _parse_data_table(text)


def parse_clo_plo_map(text: str) -> list[dict[str, str]]:
    """Parse CLOs-PLOs mapping table."""
    return _parse_data_table(text)


def parse_file(file_path: Path) -> ParsedDocument | None:
    """Parse a single markdown file into a ParsedDocument.

    Returns None if the file cannot be classified.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.error(f"Encoding failure on {file_path.name}. Must be UTF-8.")
        return None

    result = classify_entity(content)
    if result is None:
        logger.warning(f"Cannot classify {file_path.name} — skipping.")
        return None

    entity_type, entity_id = result
    sections = _split_sections(content)

    general_info = {}
    if "General information" in sections:
        general_info = _parse_kv_table(sections["General information"])

    return ParsedDocument(
        file_path=file_path,
        entity_type=entity_type,
        entity_id=entity_id,
        general_info=general_info,
        sections=sections,
        raw_content=content,
    )


def parse_corpus(data_dir: Path) -> list[ParsedDocument]:
    """Parse all markdown files in a directory."""
    md_files = sorted(data_dir.glob("*.[mM][dD]"))
    logger.info(f"Parsing {len(md_files)} files from {data_dir}")

    documents: list[ParsedDocument] = []
    for fp in md_files:
        doc = parse_file(fp)
        if doc:
            documents.append(doc)

    counts = {}
    for doc in documents:
        counts[doc.entity_type.value] = counts.get(doc.entity_type.value, 0) + 1
    logger.info(f"Parsed: {counts}")

    return documents
