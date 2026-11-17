# Docker deployment

The compose file starts two services:

1. `ollama` — local LLM and embedding server
2. `rag` — Streamlit chatbot with mounted document and vector-store volumes

Persistent data:

- Handbook PDFs and ChromaDB files: `../source_code/data`
- Ollama model weights: Docker volume `ollama_models`

## Build and run 

From this `docker/` directory:

```bash
docker compose up --build
```

Pull models once the Ollama container is healthy:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen2.5:0.5b
```

Open http://localhost:8501

## Windows with Docker Desktop and a host Ollama install

```powershell
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up --build rag
```

## CLI inside the container

```bash
docker compose exec rag python run_chatbot.py --question "...?"
```
