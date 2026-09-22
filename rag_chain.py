"""Retrieve relevant chunks from Chroma and answer questions with a Groq LLM."""

import argparse
import os
import sys

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from ingest import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided context. "
    "If the context does not contain the answer, say you don't know."
)


class RagChain:
    def __init__(self, top_k: int = TOP_K, groq_model: str = GROQ_MODEL):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment or .env")

        self.top_k = top_k
        self.groq_model = groq_model
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)
        self.groq = Groq(api_key=api_key)

    def retrieve(self, query: str) -> list[dict]:
        embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=embedding, n_results=self.top_k)

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {"text": doc, "source": meta.get("source"), "distance": dist}
            for doc, meta, dist in zip(docs, metadatas, distances)
        ]

    def build_prompt(self, query: str, chunks: list[dict]) -> str:
        context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
        return f"Context:\n{context}\n\nQuestion: {query}"

    def generate(self, query: str, chunks: list[dict]) -> str:
        prompt = self.build_prompt(query, chunks)
        response = self.groq.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def ask(self, query: str) -> str:
        chunks = self.retrieve(query)
        if not chunks:
            return "No relevant documents found. Have you run ingest.py?"
        return self.generate(query, chunks)


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument("query", nargs="*", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    args = parser.parse_args()

    chain = RagChain(top_k=args.top_k)

    if args.query:
        print(chain.ask(" ".join(args.query)))
        return

    print("RAG chat (Ctrl+C to exit)")
    while True:
        try:
            query = input("\n> ")
        except (KeyboardInterrupt, EOFError):
            break
        if not query.strip():
            continue
        print(chain.ask(query))


if __name__ == "__main__":
    main()
