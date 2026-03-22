"""Document chunker for the RAG pipeline.

Splits parsed markdown documents into LangChain Document chunks with
rich metadata, using section-level splitting with a secondary split
for oversized chunks.
"""

from __future__ import annotations

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from src.config import settings
from src.ingestion.parser import EntityType, ParsedDocument

# Minimum chunk length in characters — skip fragments below this
MIN_CHUNK_LENGTH = 50

# Same config as corpus_statistics notebook
HEADERS_TO_SPLIT_ON = [
    ("#", "Entity"),
    ("##", "Section"),
]


def _build_metadata(doc: ParsedDocument, section_name: str) -> dict:
    """Build metadata dict for a chunk."""
    meta: dict = {
        "entity_type": doc.entity_type.value,
        "entity_id": doc.entity_id,
        "section": section_name,
        "source_file": doc.file_path.name,
    }

    if doc.entity_type == EntityType.SYLLABUS:
        meta["course_code"] = doc.general_info.get("Subject Code", "").strip()
        meta["course_name"] = doc.general_info.get("Syllabus Name", "").strip()
        meta["credits"] = doc.general_info.get("NoCredit", "").strip()
        prereq = doc.general_info.get("Pre-Requisite", "").strip()
        meta["has_prerequisites"] = bool(
            prereq and prereq.lower() not in ("none", "không", "")
        )
    elif doc.entity_type == EntityType.CURRICULUM:
        meta["curriculum_code"] = doc.general_info.get("CurriculumCode", "").strip()
    elif doc.entity_type == EntityType.PATHWAY:
        meta["pathway_id"] = doc.entity_id
        meta["semester"] = doc.general_info.get("Semester", "").strip()

    return meta


def chunk_document(doc: ParsedDocument) -> list[Document]:
    """Chunk a single parsed document into LangChain Documents.

    Uses MarkdownHeaderTextSplitter for section-level splitting, then
    applies RecursiveCharacterTextSplitter on chunks exceeding the
    configured chunk_size.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
    )
    secondary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # Split by markdown headers
    header_chunks = md_splitter.split_text(doc.raw_content)

    result: list[Document] = []
    for chunk in header_chunks:
        section_name = chunk.metadata.get("Section", "Unknown")
        base_metadata = _build_metadata(doc, section_name)
        # Merge header metadata
        base_metadata.update(chunk.metadata)

        content = chunk.page_content.strip()

        # Skip chunks that are too small to be useful for retrieval
        if len(content) < MIN_CHUNK_LENGTH:
            continue

        # Apply secondary split if content is too large
        if len(content) > settings.chunk_size * 4:  # rough char estimate
            sub_chunks = secondary_splitter.split_text(content)
            for i, sub_content in enumerate(sub_chunks):
                if len(sub_content.strip()) < MIN_CHUNK_LENGTH:
                    continue
                meta = {**base_metadata, "chunk_index": i}
                result.append(Document(page_content=sub_content, metadata=meta))
        else:
            result.append(Document(page_content=content, metadata=base_metadata))

    return result


def chunk_corpus(documents: list[ParsedDocument]) -> list[Document]:
    """Chunk all parsed documents into LangChain Documents."""
    all_chunks: list[Document] = []

    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    logger.info(
        f"Chunked {len(documents)} documents into {len(all_chunks)} chunks."
    )
    return all_chunks
