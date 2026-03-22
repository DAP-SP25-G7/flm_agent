"""Embed document chunks and upsert to Pinecone vector store."""

from __future__ import annotations

import hashlib
import time

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from pinecone import Pinecone, ServerlessSpec

from src.config import settings


def _get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def _ensure_index(pc: Pinecone) -> None:
    """Create the Pinecone index if it doesn't already exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        logger.info(f"Creating Pinecone index '{settings.pinecone_index_name}'...")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            logger.info("Waiting for index to be ready...")
            time.sleep(2)
        logger.info("Index created and ready.")
    else:
        logger.info(f"Index '{settings.pinecone_index_name}' already exists.")


def _chunk_id(chunk: Document) -> str:
    """Generate a deterministic ID for a chunk based on its content + metadata."""
    source = chunk.metadata.get("source_file", "")
    section = chunk.metadata.get("section", "")
    idx = chunk.metadata.get("chunk_index", 0)
    key = f"{source}:{section}:{idx}:{chunk.page_content[:100]}"
    return hashlib.md5(key.encode()).hexdigest()


def _sanitize_metadata(metadata: dict) -> dict:
    """Ensure all metadata values are Pinecone-compatible types."""
    sanitized = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        elif isinstance(v, list):
            # Pinecone supports lists of strings
            sanitized[k] = [str(item) for item in v]
        else:
            sanitized[k] = str(v)
    return sanitized


def embed_and_upsert(
    chunks: list[Document],
    batch_size: int = 100,
) -> int:
    """Embed all chunks and upsert to Pinecone.

    Returns the number of vectors upserted.
    """
    pc = _get_pinecone_client()
    _ensure_index(pc)
    index = pc.Index(settings.pinecone_index_name)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )

    total_upserted = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]

        logger.info(
            f"Embedding batch {i // batch_size + 1} "
            f"({len(batch)} chunks)..."
        )
        vectors = embeddings.embed_documents(texts)

        records = []
        for chunk, vector in zip(batch, vectors):
            records.append({
                "id": _chunk_id(chunk),
                "values": vector,
                "metadata": {
                    **_sanitize_metadata(chunk.metadata),
                    "text": chunk.page_content,
                },
            })

        index.upsert(vectors=records)
        total_upserted += len(records)
        logger.info(f"Upserted {total_upserted}/{len(chunks)} vectors.")

    return total_upserted
