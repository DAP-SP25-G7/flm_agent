"""Pinecone vector store retriever factory."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.config import settings


@lru_cache(maxsize=1)
def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    """Return a PineconeVectorStore connected to the ingested index."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    return PineconeVectorStore(
        index=index,
        embedding=_get_embeddings(),
        text_key="text",
    )
