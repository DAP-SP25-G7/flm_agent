"""Tests for the structured data extractor."""

from pathlib import Path

import pytest

from src.ingestion.parser import EntityType, parse_corpus
from src.ingestion.structured import (
    build_combo_map,
    build_course_index,
    build_curriculum_map,
    build_prerequisites,
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture
def parsed_docs():
    """Parse the full corpus once for all tests."""
    if not RAW_DIR.exists():
        pytest.skip("data/raw not found")
    return parse_corpus(RAW_DIR)


@pytest.fixture
def curriculum_doc(parsed_docs):
    for doc in parsed_docs:
        if doc.entity_type == EntityType.CURRICULUM and "20" in doc.entity_id:
            return doc
    pytest.skip("K20-K21 curriculum not found")


@pytest.fixture
def syllabus_docs(parsed_docs):
    return [d for d in parsed_docs if d.entity_type == EntityType.SYLLABUS]


@pytest.fixture
def pathway_docs(parsed_docs):
    return [d for d in parsed_docs if d.entity_type == EntityType.PATHWAY]


class TestCourseIndex:
    def test_contains_expected_courses(self, curriculum_doc, syllabus_docs):
        index = build_course_index(curriculum_doc, syllabus_docs)
        assert len(index) > 0
        # Check a known course from K20-K21
        assert "PFP191" in index
        assert index["PFP191"]["semester"] == "1"
        assert index["PFP191"]["credits"] == "3"

    def test_enriched_with_syllabus_data(self, curriculum_doc, syllabus_docs):
        index = build_course_index(curriculum_doc, syllabus_docs)
        # Courses with syllabi should have descriptions
        pfp = index.get("PFP191", {})
        assert pfp.get("description", "") != "" or pfp.get("name", "") != ""


class TestPrerequisites:
    def test_extracts_prereqs(self, curriculum_doc, syllabus_docs):
        index = build_course_index(curriculum_doc, syllabus_docs)
        prereqs = build_prerequisites(index)
        assert len(prereqs) > 0
        # CSD203 requires PFP191
        assert "CSD203" in prereqs

    def test_excludes_none_prereqs(self, curriculum_doc, syllabus_docs):
        index = build_course_index(curriculum_doc, syllabus_docs)
        prereqs = build_prerequisites(index)
        for code, prereq in prereqs.items():
            assert prereq.lower() not in ("none", "không", "")


class TestCurriculumMap:
    def test_has_semesters(self, curriculum_doc):
        cmap = build_curriculum_map(curriculum_doc)
        assert len(cmap) > 0
        # Should have semesters 0-9
        assert "1" in cmap
        assert "5" in cmap

    def test_semester_courses(self, curriculum_doc):
        cmap = build_curriculum_map(curriculum_doc)
        sem1 = cmap.get("1", [])
        codes = [c["code"] for c in sem1]
        assert "PFP191" in codes


class TestComboMap:
    def test_has_pathways(self, pathway_docs):
        combo = build_combo_map(pathway_docs)
        assert len(combo) > 0
        assert "AI17_COM_1" in combo

    def test_pathway_has_topics(self, pathway_docs):
        combo = build_combo_map(pathway_docs)
        com1 = combo.get("AI17_COM_1", {})
        topics = com1.get("topics", {})
        assert len(topics) > 0
