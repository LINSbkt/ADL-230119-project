#!/usr/bin/env python3
"""Run handbook test questions and save actual RAG outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.config import Settings
from rag.pipeline import RAGPipeline


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the IS handbook chatbot")
    parser.add_argument(
        "--questions",
        default=str(ROOT / "evaluation" / "questions.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "evaluation" / "results.json"),
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--llm", default=None)
    args = parser.parse_args()

    questions_path = Path(args.questions)
    output_path = Path(args.output)
    items = load_questions(questions_path)

    settings = Settings.from_env(
        data_dir=ROOT / "data" / "pdfs",
        vector_store=ROOT / "data" / "pdfs" / "vector_store",
        collection="is_handbook",
        strict_context=True,
        llm_model=args.llm,
    )
    pipeline = RAGPipeline(settings, rebuild=args.rebuild)

    results = []
    for item in items:
        print(f"Asking: {item['id']} - {item['question']}", flush=True)
        response = pipeline.ask(item["question"])
        results.append(
            {
                "id": item["id"],
                "category": item.get("category"),
                "expected_notes": item.get("expected_notes"),
                "question": item["question"],
                "answer": response["answer"],
                "used_fallback": response["used_fallback"],
                "avg_score": response["avg_score"],
                "sources": [
                    {
                        "source": s["source"],
                        "page": s["page"],
                        "score": s["score"],
                    }
                    for s in response["sources"]
                ],
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
        "ollama_url": settings.ollama_url,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
