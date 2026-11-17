#!/usr/bin/env python3
"""Offline tests for PDF/TXT loading and chunking. No Ollama required."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.chunking import split_documents
from rag.config import Settings
from rag.loaders import load_documents, load_pdf_file, load_txt_file
from scripts.make_sample_pdf import write_text_pdf


def write_sample_pdf(path: Path) -> None:
    write_text_pdf(
        path,
        [
            ["Laboratory Safety Rule: goggles must be worn at all times."],
            ["Waste bottles must be labeled before disposal."],
        ],
    )


class LoaderTests(unittest.TestCase):
    def test_txt_loader_keeps_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "note.txt"
            path.write_text("Fulbright enrolled 832 students.", encoding="utf-8")
            docs = load_txt_file(path)
            self.assertEqual(len(docs), 1)
            self.assertIn("832 students", docs[0].page_content)
            self.assertEqual(docs[0].metadata["source"], "note.txt")
            self.assertEqual(docs[0].metadata["file_type"], "txt")

    def test_pdf_loader_extracts_text_and_page_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "lab.pdf"
            write_sample_pdf(pdf_path)
            docs = load_pdf_file(pdf_path)
            self.assertGreaterEqual(len(docs), 1)
            joined = " ".join(d.page_content for d in docs)
            self.assertIn("goggles", joined.lower())
            self.assertEqual(docs[0].metadata["file_type"], "pdf")
            self.assertEqual(docs[0].metadata["page"], 1)
            if len(docs) > 1:
                self.assertEqual(docs[1].metadata["page"], 2)

    def test_directory_loader_skips_temp_files_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "keep.txt").write_text("Keep this handbook sentence about credits.", encoding="utf-8")
            (root / "._hidden.txt").write_text("ignore me", encoding="utf-8")
            docs = load_documents(root)
            sources = {d.metadata["source"] for d in docs}
            self.assertIn("keep.txt", sources)
            self.assertNotIn("._hidden.txt", sources)
            settings = Settings.from_env(data_dir=root, vector_store=root / "vector_store")
            chunks = split_documents(docs, settings)
            self.assertGreaterEqual(len(chunks), 1)

    def test_is_handbook_pdf_extracts_key_facts(self) -> None:
        handbook = ROOT / "data" / "pdfs" / "IS_MajorHandbook.pdf"
        self.assertTrue(handbook.exists(), "IS_MajorHandbook.pdf should be copied into data/pdfs")
        docs = load_pdf_file(handbook)
        self.assertGreaterEqual(len(docs), 10)
        text = "\n".join(d.page_content for d in docs)
        self.assertIn("Integrated Sciences", text)
        self.assertIn("Introductory Biology", text)
        self.assertIn("General Chemistry", text)
        pages = [d.metadata["page"] for d in docs]
        self.assertEqual(pages, list(range(1, len(docs) + 1)))


if __name__ == "__main__":
    unittest.main()
