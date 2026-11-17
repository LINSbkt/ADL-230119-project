# Building and Improving a Local RAG System

CS311 - Advanced Deep Learning final project (Khanh Tran - 230119). This repository implements a fully local Retrieval-Augmented Generation (RAG) assistant for Fulbright University Vietnam’s **B.S. Integrated Sciences** major handbook.

## Table of content

| Folder | Contents |
| --- | --- |
| `source_code/` | Python RAG pipeline, CLI tools, Streamlit UI, handbook PDF, evaluation files |
| `docker/` | Dockerfile, Compose files, and deployment notes |
| `report/` | PDF report |
| `demo_video/` | The foour required recordings |

## Architecture

```
TXT / PDF documents
        │  pypdf + UTF-8 loaders
        ▼
 RecursiveCharacterTextSplitter (1000 / 200)
        │  Ollama nomic-embed-text
        ▼
     ChromaDB (persistent)
        │  top-k similarity search
        ▼
 Prompt + Ollama qwen2.5:0.5b
        ▼
 CLI, evaluation script, or Streamlit UI
```

Everything runs on the local machine.

## Requirements

- Python 3.10
- [Ollama](https://ollama.com/download) with:
  - `ollama pull nomic-embed-text`
  - `ollama pull qwen2.5:0.5b`
- Docker Desktop (WSL2 backend on Windows)

Create the environment:

```bash
conda create -n rag python=3.10
conda activate rag
pip install -r source_code/requirements.txt
```

Start Ollama (`ollama serve` if it is not already running). Default API: `http://127.0.0.1:11434`.

## Usage

All commands below assume the working directory is `source_code/`.

TXT baseline:

```bash
python run_txt.py
python run_txt.py --question "Where is the FUV campus?"
```

Task 1 PDF demo (small known-content PDF):

```bash
python scripts/make_sample_pdf.py
python run_pdf.py --rebuild --question "What is the laboratory safety rule?"
```

Task 2 Integrated Sciences handbook chatbot:

```bash
python run_chatbot.py --rebuild --question "What are the two IS concentrations?"
python run_chatbot.py --interactive
```

The interactive command stays at the `Q>` prompt until you type `exit`. To launch Streamlit, exit the CLI first and run the next command in PowerShell; commands typed at `Q>` are treated as chatbot questions.

Task 4 web UI:

```bash
streamlit run app.py
```

Then open http://localhost:8501.

Offline PDF/TXT tests (no Ollama required):

```bash
python tests/test_pdf_support.py
```

Handbook evaluation. This rebuilds the index and overwrites the previous output:

```bash
python evaluate.py --rebuild
```

Raw answers and retrieval data are written to `source_code/evaluation/results.json`. The saved run's manual per-question analysis is in `source_code/evaluation/results.md`.

## Docker

See `docker/README.md`. Short version from `docker/`:

```bash
docker compose up -d --build
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen2.5:0.5b
docker compose ps
```

Open http://localhost:8501 after the `rag` service is healthy. Run the `ollama pull`
commands from a second terminal in this same `docker/` directory. The handbook index is
created automatically on first use if it does not already exist.

Volumes:

- `../source_code/data` is mounted at `/app/data` so PDFs and ChromaDB persist on the host
- Ollama weights persist in the `ollama_models` volume

If Ollama is already installed on Windows/macOS:

```bash
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up --build rag
```

## Configuration

Environment variables (optional):

| Variable | Default |
| --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` |
| `RAG_EMBED_MODEL` | `nomic-embed-text` |
| `RAG_LLM_MODEL` | `qwen2.5:0.5b` |
| `RAG_DATA_DIR` | `source_code/data/pdfs` |
| `RAG_VECTOR_STORE` | `<data_dir>/vector_store` |
| `RAG_TOP_K` | `5` |
| `RAG_STRICT_CONTEXT` | `true` for handbook mode |

## Known limitations

- PDF extraction reads selectable text only. Text inside images needs OCR, which is not included.
- `qwen2.5:0.5b` is small and can still answer incompletely or incorrectly even with good retrieval.
- The Docker Desktop Compose stack and browser Q&A were tested on Windows on 2026-11-16. Verify persistence after restarting the containers when reproducing the deployment.

