"""Build or load a persistent ChromaDB index."""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from rag.chunking import split_documents
from rag.config import Settings
from rag.loaders import load_documents


def make_embeddings(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_url)


def collection_count(db: Chroma) -> int:
    try:
        return int(db._collection.count())  # noqa: SLF001 - Chroma client access
    except Exception:
        return 0


def load_or_build_index(
    settings: Settings,
    embeddings: OllamaEmbeddings | None = None,
    rebuild: bool = False,
) -> Chroma:
    embeddings = embeddings or make_embeddings(settings)
    store_path = Path(settings.vector_store)

    if rebuild and store_path.exists():
        print(f"Removing existing vector store: {store_path}")
        shutil.rmtree(store_path)

    if store_path.exists() and any(store_path.iterdir()):
        print(f"Loading existing vector store from {store_path}...")
        db = Chroma(
            collection_name=settings.collection,
            embedding_function=embeddings,
            persist_directory=str(store_path),
        )
        print(f"  Indexed chunks: {collection_count(db)}")
        return db

    print(f"No vector store found. Indexing documents from {settings.data_dir}...")
    documents = load_documents(settings.data_dir)
    if not documents:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in '{settings.data_dir}'"
        )
    chunks = split_documents(documents, settings)
    print(f"  Indexing {len(chunks)} chunks...")
    store_path.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.collection,
        persist_directory=str(store_path),
    )
