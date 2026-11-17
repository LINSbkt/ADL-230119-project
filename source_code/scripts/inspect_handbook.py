#!/usr/bin/env python3
"""Print handbook extraction statistics without calling Ollama."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.chunking import split_documents
from rag.config import Settings
from rag.loaders import load_pdf_file


def main() -> None:
    handbook = ROOT / "data" / "pdfs" / "IS_MajorHandbook.pdf"
    docs = load_pdf_file(handbook)
    settings = Settings.from_env(data_dir=handbook.parent)
    chunks = split_documents(docs, settings)
    print(f"File: {handbook.name}")
    print(f"Pages with text: {len(docs)}")
    print(f"Chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap}): {len(chunks)}")
    print("Sample pages:")
    for doc in docs[:3]:
        preview = " ".join(doc.page_content.split())[:180]
        print(f"  page {doc.metadata['page']}: {preview}")
    needles = [
        "Environmental Science and Sustainability",
        "Biology and Health Science",
        "Introductory Biology",
        "General Chemistry",
        "Capstone",
        "MLO",
    ]
    print("Keyword page hits:")
    joined_pages = [(d.metadata["page"], d.page_content) for d in docs]
    for needle in needles:
        hits = [page for page, text in joined_pages if needle.lower() in text.lower()]
        print(f"  {needle}: pages {hits[:8]}")


if __name__ == "__main__":
    main()
