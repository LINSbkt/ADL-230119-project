#!/usr/bin/env python3
"""Task 1 CLI: extract, chunk, embed, and query PDF documents."""

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
    parser = argparse.ArgumentParser(description="Local RAG Q&A for PDF files")
    parser.add_argument(
        "--question",
        "-q",
        default="What is the laboratory safety rule described in the sample PDF?",
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "pdfs_task1"),
        help="Folder containing PDF files",
    )
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    settings = Settings.from_env(
        data_dir=data_dir,
        vector_store=data_dir / "vector_store",
        collection="rag_pdf",
        strict_context=True,
    )
    print(f"PDF data directory: {settings.data_dir}")
    print(f"Loading Ollama embeddings: {settings.embed_model}")
    pipeline = RAGPipeline(settings, rebuild=args.rebuild)
    print(f"Connecting to Ollama ({settings.llm_model})...")
    print(f"Q: {args.question}")
    result = pipeline.ask(args.question)
    print("\nA: " + result["answer"] + "\n")
    if result["used_fallback"]:
        print("(no sufficiently relevant PDF chunks retrieved)")
        return
    print("Sources:")
    for source in result["sources"][:5]:
        print(f"   {source['label']}  score={source['score']}")


if __name__ == "__main__":
    main()
