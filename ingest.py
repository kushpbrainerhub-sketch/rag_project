"""Ingest documents from ./data into a persistent Chroma vector store."""

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")


def _default_chroma_dir() -> str:
    """Prefer a local ./chroma_db (persists across runs), but fall back to the OS temp
    directory if that's not writable. On Streamlit Community Cloud the repo is baked into
    a read-only image layer: creating a brand-new file in chroma_db/ can still succeed (it
    lands on the writable overlay), but writing to a chroma.sqlite3 that was already
    committed to git fails -- so this must probe the existing file itself, not just the
    directory, or it wrongly concludes the directory is writable."""
    preferred = Path("chroma_db")
    sqlite_file = preferred / "chroma.sqlite3"
    try:
        if sqlite_file.exists():
            with open(sqlite_file, "r+b"):
                pass
        else:
            preferred.mkdir(exist_ok=True)
            probe = preferred / ".write_test"
            probe.write_text("ok")
            probe.unlink()
        return str(preferred)
    except OSError:
        return str(Path(tempfile.gettempdir()) / "rag_project_chroma_db")


CHROMA_DIR = os.environ.get("CHROMA_DIR", _default_chroma_dir())
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text) pairs. Non-paginated files use page 0."""
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print(f"Skipping {path.name}: install pypdf to ingest PDFs (pip install pypdf)")
            return []
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(0, path.read_text(encoding="utf-8", errors="ignore"))]


def load_file(path: Path, source: str) -> list[tuple[str, int, str]]:
    """Return (source, page, text) triples for one file, skipping empty pages."""
    return [(source, page, text) for page, text in extract_pages(path) if text.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def load_documents(data_dir: Path) -> list[tuple[str, int, str]]:
    supported = {".txt", ".md", ".pdf"}
    docs = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported:
            docs.extend(load_file(path, str(path.relative_to(data_dir))))
    return docs


def ingest_documents(documents: list[tuple[str, int, str]], model: SentenceTransformer, collection, show_progress: bool = False) -> int:
    """Chunk, embed, and upsert (source, page, text) triples into a Chroma collection. Returns chunk count."""
    ids, chunks, metadatas = [], [], []
    for source, page, text in documents:
        for i, chunk in enumerate(chunk_text(text)):
            chunk_id = hashlib.sha256(f"{source}:{page}:{i}".encode()).hexdigest()
            ids.append(chunk_id)
            chunks.append(chunk)
            metadatas.append({"source": source, "page": page, "chunk": i})

    if not chunks:
        return 0

    embeddings = model.encode(chunks, show_progress_bar=show_progress).tolist()
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Folder of source documents")
    parser.add_argument("--reset", action="store_true", help="Delete the collection before ingesting")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    documents = load_documents(data_dir)
    if not documents:
        print(f"No supported documents (.txt, .md, .pdf) found in {data_dir}/")
        return

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if args.reset:
        client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [c.name for c in client.list_collections()] else None
    collection = client.get_or_create_collection(COLLECTION_NAME)

    n_sources = len({source for source, _, _ in documents})
    print(f"Embedding chunks from {len(documents)} page(s)/section(s) across {n_sources} file(s)...")
    n = ingest_documents(documents, model, collection, show_progress=True)
    if n == 0:
        print("No chunks produced from documents.")
        return

    print(f"Ingested {n} chunks into collection '{COLLECTION_NAME}' at {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
