#!/usr/bin/env python3
"""Task 1 companion CLI: TXT documents, matching the professor's baseline usage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.config import Settings
from rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG Q&A for TXT files")
    parser.add_argument("--question", "-q", default="How many students are there at Fulbright?")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the TXT vector store")
    args = parser.parse_args()

    settings = Settings.from_env(
        data_dir=ROOT / "data" / "txt",
        vector_store=ROOT / "data" / "txt" / "vector_store",
        collection="rag_txt",
        strict_context=False,
    )
    print(f"Loading Ollama embeddings: {settings.embed_model}")
    pipeline = RAGPipeline(settings, rebuild=args.rebuild)
    print(f"Connecting to Ollama ({settings.llm_model})...")
    print(f"Q: {args.question}")
    result = pipeline.ask(args.question)
    print("\nA: " + result["answer"] + "\n")
    if result["used_fallback"]:
        print("(answered from model knowledge — no relevant documents found)")
        return
    print("Sources:")
    for source in result["sources"][:3]:
        print(f"   {source['label']}  score={source['score']}")


if __name__ == "__main__":
    main()
