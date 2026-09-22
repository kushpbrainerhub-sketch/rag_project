"""Retrieve relevant chunks from Chroma and answer questions with a Groq LLM."""

import argparse
import os
import re
import sys

import chromadb
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from ingest import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 4
RRF_K = 60  # reciprocal rank fusion constant

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided context. "
    "If the context does not contain the answer, say you don't know. "
    "Cite sources inline like [source, p.N] (omit the page if none is given)."
)

HISTORY_TURNS = 6  # messages (3 user/assistant pairs) kept for conversational context


class RagChain:
    def __init__(self, top_k: int = TOP_K, groq_model: str = GROQ_MODEL):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment or .env")

        self.top_k = top_k
        self.groq_model = groq_model
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)
        self.groq = Groq(api_key=api_key)
        self.refresh_index()

    def refresh_index(self) -> None:
        """(Re)build the in-memory BM25 index from the current collection contents.
        Call this after ingesting new documents into a live RagChain (e.g. from app.py)."""
        data = self.collection.get(include=["documents", "metadatas"])
        self._bm25_ids = data["ids"]
        self._bm25_texts = data["documents"]
        self._bm25_metas = data["metadatas"]
        tokenized = [self._tokenize(t) for t in self._bm25_texts]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _vector_search(self, query: str, n: int) -> list[dict]:
        embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=embedding, n_results=n)

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "id": cid,
                "text": doc,
                "source": meta.get("source"),
                "page": meta.get("page") or None,
                "distance": dist,
            }
            for cid, doc, meta, dist in zip(ids, docs, metadatas, distances)
        ]

    def _bm25_search(self, query: str, n: int) -> list[dict]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            {
                "id": self._bm25_ids[i],
                "text": self._bm25_texts[i],
                "source": self._bm25_metas[i].get("source"),
                "page": self._bm25_metas[i].get("page") or None,
                "distance": None,
            }
            for i in ranked[:n] if scores[i] > 0
        ]

    @staticmethod
    def _fuse(result_lists: list[list[dict]], pool_size: int) -> list[dict]:
        """Reciprocal rank fusion: merge ranked lists into one, deduped by id."""
        scores: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for hits in result_lists:
            for rank, hit in enumerate(hits):
                scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1 / (RRF_K + rank + 1)
                by_id.setdefault(hit["id"], hit)
        ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:pool_size]
        return [by_id[i] for i in ranked_ids]

    def retrieve(self, query: str) -> list[dict]:
        pool = max(20, self.top_k * 4)
        vector_hits = self._vector_search(query, pool)
        bm25_hits = self._bm25_search(query, pool)
        candidates = self._fuse([vector_hits, bm25_hits], pool_size=max(15, self.top_k * 3))
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        for c, score in zip(candidates, rerank_scores):
            c["rerank_score"] = float(score)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

        return candidates[: self.top_k]

    @staticmethod
    def label(chunk: dict) -> str:
        return f"{chunk['source']}, p.{chunk['page']}" if chunk.get("page") else chunk["source"]

    def build_prompt(self, query: str, chunks: list[dict]) -> str:
        context = "\n\n".join(f"[{self.label(c)}]\n{c['text']}" for c in chunks)
        return f"Context:\n{context}\n\nQuestion: {query}"

    def condense_question(self, query: str, history: list[dict]) -> str:
        """Rewrite a follow-up question into a standalone one using chat history."""
        if not history:
            return query
        convo = "\n".join(f"{h['role']}: {h['content']}" for h in history[-HISTORY_TURNS:])
        prompt = (
            "Given the conversation so far and a follow-up question, rewrite the follow-up "
            "as a standalone question that includes any context it implicitly refers to. "
            "Reply with only the rewritten question, nothing else.\n\n"
            f"Conversation:\n{convo}\n\nFollow-up question: {query}\nStandalone question:"
        )
        response = self.groq.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def generate(self, query: str, chunks: list[dict], history: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in (history or [])[-HISTORY_TURNS:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": self.build_prompt(query, chunks)})

        response = self.groq.chat.completions.create(model=self.groq_model, messages=messages)
        return response.choices[0].message.content

    def ask(self, query: str, history: list[dict] | None = None) -> str:
        history = history or []
        standalone_query = self.condense_question(query, history)
        chunks = self.retrieve(standalone_query)
        if not chunks:
            return "No relevant documents found. Have you run ingest.py?"
        return self.generate(query, chunks, history)


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
    history: list[dict] = []
    while True:
        try:
            query = input("\n> ")
        except (KeyboardInterrupt, EOFError):
            break
        if not query.strip():
            continue
        answer = chain.ask(query, history)
        print(answer)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
