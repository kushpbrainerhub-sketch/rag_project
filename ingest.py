"""Ingest documents from ./data into a persistent Chroma vector store."""

import argparse
import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print(f"Skipping {path.name}: install pypdf to ingest PDFs (pip install pypdf)")
            return ""
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


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


def load_documents(data_dir: Path) -> list[tuple[str, str]]:
    supported = {".txt", ".md", ".pdf"}
    docs = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported:
            text = read_text(path)
            if text.strip():
                docs.append((str(path.relative_to(data_dir)), text))
    return docs


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

    ids, chunks, metadatas = [], [], []
    for source, text in documents:
        for i, chunk in enumerate(chunk_text(text)):
            chunk_id = hashlib.sha256(f"{source}:{i}".encode()).hexdigest()
            ids.append(chunk_id)
            chunks.append(chunk)
            metadatas.append({"source": source, "chunk": i})

    if not chunks:
        print("No chunks produced from documents.")
        return

    print(f"Embedding {len(chunks)} chunks from {len(documents)} document(s)...")
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()

    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    print(f"Ingested {len(chunks)} chunks into collection '{COLLECTION_NAME}' at {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
