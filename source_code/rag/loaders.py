"""Load TXT and PDF files into LangChain documents."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

SKIP_PREFIXES = ("._", "~$", ".ds_store")
SKIP_DIR_NAMES = {"vector_store", "__macosx", "__pycache__"}


def is_ignored(path: Path) -> bool:
    name = path.name.lower()
    if any(name.startswith(prefix) or name == prefix.lstrip(".") for prefix in SKIP_PREFIXES):
        return True
    return any(part.lower() in SKIP_DIR_NAMES for part in path.parts)


def load_txt_file(path: Path) -> list[Document]:
    content = path.read_text(encoding="utf-8")
    return [
        Document(
            page_content=content,
            metadata={
                "source": path.name,
                "path": str(path),
                "file_type": "txt",
                "page": 1,
            },
        )
    ]


def load_pdf_file(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    documents: list[Document] = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        cleaned = "\n".join(line.strip() for line in text.splitlines()).strip()
        if not cleaned:
            continue
        documents.append(
            Document(
                page_content=cleaned,
                metadata={
                    "source": path.name,
                    "path": str(path),
                    "file_type": "pdf",
                    "page": index + 1,
                },
            )
        )
    return documents


def load_documents(directory: str | Path) -> list[Document]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    documents: list[Document] = []
    files = sorted(
        p for p in root.iterdir() if p.is_file() and not is_ignored(p)
    )
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            print(f"  Loading TXT: {path.name}")
            documents.extend(load_txt_file(path))
        elif suffix == ".pdf":
            print(f"  Loading PDF: {path.name}")
            documents.extend(load_pdf_file(path))
    return documents
