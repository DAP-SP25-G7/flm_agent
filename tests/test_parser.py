"""Tests for the markdown parser."""

from pathlib import Path

import pytest

from src.ingestion.parser import (
    EntityType,
    classify_entity,
    parse_corpus,
    parse_file,
    _parse_kv_table,
    _parse_data_table,
    _split_sections,
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


class TestClassifyEntity:
    def test_syllabus(self):
        content = "# Syllabus: PFP191\n\n## General information\n"
        result = classify_entity(content)
        assert result is not None
        assert result[0] == EntityType.SYLLABUS
        assert result[1] == "PFP191"

    def test_curriculum(self):
        content = "# Curriculum: 20D-21A\n\n## General information\n"
        result = classify_entity(content)
        assert result is not None
        assert result[0] == EntityType.CURRICULUM
        assert result[1] == "20D-21A"

    def test_pathway(self):
        content = "# Pathway: AI17_COM_1\n\n## General information\n"
        result = classify_entity(content)
        assert result is not None
        assert result[0] == EntityType.PATHWAY
        assert result[1] == "AI17_COM_1"

    def test_unknown(self):
        content = "# Unknown header\n\nSome text."
        assert classify_entity(content) is None


class TestSplitSections:
    def test_basic_split(self):
        content = "# Title\n\n## Section A\nContent A\n\n## Section B\nContent B"
        sections = _split_sections(content)
        assert "Section A" in sections
        assert "Section B" in sections
        assert "Content A" in sections["Section A"]
        assert "Content B" in sections["Section B"]


class TestParseKvTable:
    def test_syllabus_format(self):
        text = """|         Pre-Requisite: | PFP191, DBI202 |   |
|           Description: | Some course    |   |"""
        result = _parse_kv_table(text)
        assert result["Pre-Requisite"] == "PFP191, DBI202"
        assert result["Description"] == "Some course"

    def test_pathway_format(self):
        text = """| Key         | Value              |
|-------------|--------------------|
| Description | Subject 1 of Combo |
| Curriculum  | BIT_AI_*           |
| Semester    | 5                  |"""
        result = _parse_kv_table(text)
        assert result["Description"] == "Subject 1 of Combo"
        assert result["Semester"] == "5"


class TestParseDataTable:
    def test_basic_table(self):
        text = """| SubjectCode | Subject Name | Semester | NoCredit | PreRequisite |
|-------------|-------------|----------|----------|-------------|
| PFP191      | Programming  | 1        | 3        |             |
| CSD203      | Data Struct  | 2        | 3        | PFP191      |"""
        rows = _parse_data_table(text)
        assert len(rows) == 2
        assert rows[0]["SubjectCode"] == "PFP191"
        assert rows[1]["PreRequisite"] == "PFP191"


class TestParseRealFiles:
    """Integration tests that read actual data files."""

    @pytest.mark.skipif(not RAW_DIR.exists(), reason="data/raw not found")
    def test_parse_syllabus_pfp191(self):
        fp = RAW_DIR / "PFP191.MD"
        if not fp.exists():
            pytest.skip("PFP191.MD not found")
        doc = parse_file(fp)
        assert doc is not None
        assert doc.entity_type == EntityType.SYLLABUS
        assert "General information" in doc.sections
        assert doc.general_info.get("Subject Code") is not None

    @pytest.mark.skipif(not RAW_DIR.exists(), reason="data/raw not found")
    def test_parse_curriculum_k20(self):
        fp = RAW_DIR / "Curriculum_k20_k21.MD"
        if not fp.exists():
            pytest.skip("Curriculum_k20_k21.MD not found")
        doc = parse_file(fp)
        assert doc is not None
        assert doc.entity_type == EntityType.CURRICULUM
        assert "Curriculum details" in doc.sections

    @pytest.mark.skipif(not RAW_DIR.exists(), reason="data/raw not found")
    def test_parse_pathway(self):
        fp = RAW_DIR / "AI17_COM_1.MD"
        if not fp.exists():
            pytest.skip("AI17_COM_1.MD not found")
        doc = parse_file(fp)
        assert doc is not None
        assert doc.entity_type == EntityType.PATHWAY
        assert doc.entity_id == "AI17_COM_1"

    @pytest.mark.skipif(not RAW_DIR.exists(), reason="data/raw not found")
    def test_parse_full_corpus(self):
        docs = parse_corpus(RAW_DIR)
        assert len(docs) > 0
        types = {d.entity_type for d in docs}
        assert EntityType.SYLLABUS in types
        assert EntityType.CURRICULUM in types
        assert EntityType.PATHWAY in types
