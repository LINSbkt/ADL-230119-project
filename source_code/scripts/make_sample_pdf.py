#!/usr/bin/env python3
"""Create a small known-content PDF used to demonstrate Task 1."""

from __future__ import annotations

import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_text_pdf(path: Path, pages: list[list[str]]) -> None:
    pdf = FPDF(format="letter")
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    for lines in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        for line in lines:
            pdf.multi_cell(w=pdf.epw, h=8, text=line)
            pdf.ln(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))


def main() -> None:
    out = ROOT / "data" / "pdfs_task1" / "sample_lab_policy.pdf"
    write_text_pdf(
        out,
        [
            [
                "Fulbright Teaching Laboratory Policy",
                "Safety rule: chemical splash goggles must be worn at all times.",
                "Food and drinks are not allowed in the laboratory.",
            ],
            [
                "Waste Management",
                "All chemical waste bottles must be labeled before disposal.",
                "Emergency showers are located next to the main exit.",
            ],
        ],
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
