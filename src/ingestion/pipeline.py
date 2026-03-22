"""Orchestrates the full ingestion pipeline: parse -> structured -> chunk -> embed."""

from __future__ import annotations

from loguru import logger

from src.config import PROCESSED_DIR, RAW_DATA_DIR
from src.ingestion.chunker import chunk_corpus
from src.ingestion.embedder import embed_and_upsert
from src.ingestion.parser import parse_corpus
from src.ingestion.structured import build_all


def run_ingestion(*, skip_embed: bool = False) -> dict:
    """Run the full ingestion pipeline.

    Args:
        skip_embed: If True, skip the embedding/Pinecone step (useful for testing).

    Returns:
        Summary dict with counts and file paths.
    """
    logger.info("=" * 60)
    logger.info("Starting ingestion pipeline")
    logger.info("=" * 60)

    # Step 1: Parse raw markdown files
    logger.info("Step 1: Parsing corpus...")
    documents = parse_corpus(RAW_DATA_DIR)
    logger.info(f"Parsed {len(documents)} documents.")

    # Step 2: Build structured lookup tables
    logger.info("Step 2: Building structured lookup tables...")
    table_paths = build_all(documents, PROCESSED_DIR)
    logger.info(f"Built {len(table_paths)} lookup tables.")

    # Step 3: Chunk documents
    logger.info("Step 3: Chunking documents...")
    chunks = chunk_corpus(documents)
    logger.info(f"Produced {len(chunks)} chunks.")

    # Step 4: Embed and upsert to Pinecone
    if skip_embed:
        logger.info("Step 4: Skipping embedding (skip_embed=True).")
        n_upserted = 0
    else:
        logger.info("Step 4: Embedding and upserting to Pinecone...")
        n_upserted = embed_and_upsert(chunks)
        logger.info(f"Upserted {n_upserted} vectors to Pinecone.")

    summary = {
        "documents_parsed": len(documents),
        "lookup_tables": list(table_paths.keys()),
        "chunks_produced": len(chunks),
        "vectors_upserted": n_upserted,
    }

    logger.info("=" * 60)
    logger.info(f"Ingestion complete: {summary}")
    logger.info("=" * 60)

    return summary
