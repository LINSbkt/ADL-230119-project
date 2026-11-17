# Source code

Python 3.10 RAG application for the CS311 final project.

- `rag/` — shared pipeline (load, chunk, index, ask)
- `run_txt.py` — TXT chatbot (baseline workflow)
- `run_pdf.py` — PDF chatbot (Task 1)
- `run_chatbot.py` — IS handbook CLI (Task 2)
- `app.py` — Streamlit UI (Task 4)
- `evaluate.py` — question set runner
- `data/pdfs/IS_MajorHandbook.pdf` — Task 2 knowledge source
- `data/txt/` — original Fulbright TXT files (copies; originals untouched)

Install from this folder:

```bash
pip install -r requirements.txt
```

See the repository root `README.md` for full usage and Docker instructions.
