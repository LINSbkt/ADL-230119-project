"""Runtime settings for the local RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    vector_store: Path
    collection: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embed_model: str = "nomic-embed-text"
    llm_model: str = "qwen2.5:0.5b"
    temperature: float = 0.2
    ollama_url: str = "http://127.0.0.1:11434"
    top_k: int = 5
    score_threshold: float = 0.3
    strict_context: bool = True
    num_predict: int = 256

    @classmethod
    def from_env(
        cls,
        data_dir: str | Path | None = None,
        vector_store: str | Path | None = None,
        collection: str | None = None,
        strict_context: bool | None = None,
        llm_model: str | None = None,
    ) -> "Settings":
        resolved_data = Path(data_dir or _env("RAG_DATA_DIR", str(SOURCE_ROOT / "data" / "pdfs")))
        if not resolved_data.is_absolute():
            resolved_data = (SOURCE_ROOT / resolved_data).resolve()
        else:
            resolved_data = resolved_data.resolve()

        resolved_store = Path(
            vector_store
            or _env("RAG_VECTOR_STORE", str(resolved_data / "vector_store"))
        )
        if not resolved_store.is_absolute():
            resolved_store = (SOURCE_ROOT / resolved_store).resolve()
        else:
            resolved_store = resolved_store.resolve()

        return cls(
            data_dir=resolved_data,
            vector_store=resolved_store,
            collection=collection or _env("RAG_COLLECTION", "rag_documents"),
            chunk_size=int(_env("RAG_CHUNK_SIZE", "1000")),
            chunk_overlap=int(_env("RAG_CHUNK_OVERLAP", "200")),
            embed_model=_env("RAG_EMBED_MODEL", "nomic-embed-text"),
            llm_model=llm_model or _env("RAG_LLM_MODEL", "qwen2.5:0.5b"),
            temperature=float(_env("RAG_TEMPERATURE", "0.2")),
            ollama_url=_env("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            top_k=int(_env("RAG_TOP_K", "5")),
            score_threshold=float(_env("RAG_SCORE_THRESHOLD", "0.3")),
            strict_context=Settings._parse_bool(strict_context, _env("RAG_STRICT_CONTEXT", "true")),
            num_predict=int(_env("RAG_NUM_PREDICT", "256")),
        )

    @staticmethod
    def _parse_bool(explicit: bool | None, raw: str) -> bool:
        if explicit is not None:
            return explicit
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def prompt_template(self) -> str:
        if self.strict_context:
            return (
                "You are a helpful academic assistant for Fulbright University Vietnam.\n"
                "Answer the question using ONLY the provided document context.\n"
                "If the context does not contain the answer, say that the handbook does not provide that information.\n"
                "Do not invent course codes, credit numbers, or policies.\n"
                "Keep the answer to a short paragraph unless the question asks for a list.\n"
                "When possible, mention the source file and page number.\n\n"
                "CONTEXT FROM DOCUMENTS:\n{context}\n---\nQUESTION: {question}\n\nANSWER:"
            )
        return (
            "You are a helpful AI assistant that answers questions based on provided documents.\n\n"
            "CONTEXT FROM DOCUMENTS:\n{context}\n---\nQUESTION: {question}\n\n"
            "INSTRUCTIONS:\n"
            "1. If the context contains relevant information, answer based on that.\n"
            "2. If the context is insufficient, use your general knowledge and say so.\n"
            "3. Be concise - 2-4 sentences unless more detail is needed.\n\n"
            "ANSWER:"
        )
