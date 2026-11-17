#!/usr/bin/env python3
"""Task 2 CLI: Integrated Sciences major handbook chatbot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.config import Settings
from rag.pipeline import RAGPipeline


SAMPLE_QUESTION = (
    "What are the two concentrations in the B.S. Integrated Sciences major?"
)


def build_pipeline(rebuild: bool = False, llm_model: str | None = None) -> RAGPipeline:
    settings = Settings.from_env(
        data_dir=ROOT / "data" / "pdfs",
        vector_store=ROOT / "data" / "pdfs" / "vector_store",
        collection="is_handbook",
        strict_context=True,
        llm_model=llm_model,
    )
    print(f"Handbook directory: {settings.data_dir}")
    print(f"Loading Ollama embeddings: {settings.embed_model}")
    pipeline = RAGPipeline(settings, rebuild=rebuild)
    print(f"Connecting to Ollama ({settings.llm_model})...")
    return pipeline


def print_result(result: dict) -> None:
    print(f"Q: {result['question']}")
    print("\nA: " + result["answer"] + "\n")
    if result["used_fallback"]:
        print("(no sufficiently relevant handbook chunks retrieved)")
        return
    print("Sources:")
    for source in result["sources"][:5]:
        print(f"   {source['label']}  score={source['score']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="IS Major Handbook RAG chatbot")
    parser.add_argument("--question", "-q", default=SAMPLE_QUESTION)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--llm", default=None, help="Optional Ollama LLM name")
    args = parser.parse_args()

    pipeline = build_pipeline(rebuild=args.rebuild, llm_model=args.llm)
    if args.interactive:
        print("Type a question, or 'exit' to quit.")
        while True:
            question = input("\nQ> ").strip()
            if not question or question.lower() in {"exit", "quit"}:
                break
            print_result(pipeline.ask(question))
        return
    print_result(pipeline.ask(args.question))


if __name__ == "__main__":
    main()
