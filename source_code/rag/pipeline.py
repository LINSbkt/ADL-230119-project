"""Retrieve relevant chunks and generate an answer with Ollama."""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

from rag.config import Settings
from rag.indexing import load_or_build_index, make_embeddings


def format_source_label(metadata: dict[str, Any]) -> str:
    source = str(metadata.get("source", "Unknown"))
    page = metadata.get("page")
    if page in (None, "", 0):
        return source
    return f"{source} (page {page})"


class RAGPipeline:
    def __init__(self, settings: Settings, rebuild: bool = False):
        self.settings = settings
        self.embeddings = make_embeddings(settings)
        self.db: Chroma = load_or_build_index(
            settings, embeddings=self.embeddings, rebuild=rebuild
        )
        self.llm = OllamaLLM(
            model=settings.llm_model,
            base_url=settings.ollama_url,
            temperature=settings.temperature,
            num_predict=settings.num_predict,
        )

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        raw_results = self.db.similarity_search_with_score(question, k=self.settings.top_k)
        sources = []
        for doc, score in raw_results:
            sources.append(
                {
                    "source": str(doc.metadata.get("source", "Unknown")),
                    "page": doc.metadata.get("page"),
                    "file_type": doc.metadata.get("file_type", "unknown"),
                    "label": format_source_label(doc.metadata),
                    "score": round(1 / (1 + float(score)), 4),
                    "content": doc.page_content,
                }
            )
        return sources

    def ask(self, question: str) -> dict[str, Any]:
        sources = self.retrieve(question)
        avg_score = sum(float(s["score"]) for s in sources) / len(sources) if sources else 0.0
        used_fallback = not sources or avg_score < self.settings.score_threshold

        if used_fallback:
            if self.settings.strict_context:
                context = (
                    "No sufficiently relevant documents were retrieved. "
                    "Say that the available documents do not contain this information."
                )
            else:
                context = "No sufficiently relevant documents found. Use your general knowledge."
        else:
            context = "\n\n---\n\n".join(
                f"[{s['label']}]\n{s['content']}" for s in sources
            )

        prompt = (
            self.settings.prompt_template()
            .replace("{context}", context)
            .replace("{question}", question)
        )
        answer = str(self.llm.invoke(prompt)).strip()
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "used_fallback": used_fallback,
            "avg_score": round(avg_score, 4),
        }
