"""Split loaded documents into overlapping chunks."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Settings


def split_documents(documents, settings: Settings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)
