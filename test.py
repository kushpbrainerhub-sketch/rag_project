"""Local smoke test for the ingest -> retrieve -> Groq pipeline."""

import os
import sys

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

from ingest import CHROMA_DIR, COLLECTION_NAME
from rag_chain import RagChain

load_dotenv()

TEST_QUERIES = [
    "What is this document about?",
]


def check(label: str, condition: bool, hint: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {hint}" if hint and not condition else ""))
    return condition


def main() -> int:
    ok = True

    ok &= check("GROQ_API_KEY is set", bool(os.environ.get("GROQ_API_KEY")), "add it to .env")
    ok &= check(f"{CHROMA_DIR}/ exists", os.path.isdir(CHROMA_DIR), "run ingest.py first")

    if not ok:
        print("\nAborting: fix the above before running queries.")
        return 1

    try:
        chain = RagChain()
    except Exception as e:
        check("RagChain initializes", False, str(e))
        return 1
    check("RagChain initializes", True)

    count = chain.collection.count()
    ok &= check(f"collection '{COLLECTION_NAME}' has chunks", count > 0, "run ingest.py first")
    if not ok:
        return 1
    print(f"       -> {count} chunks in collection")

    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        chunks = chain.retrieve(query)
        if not check("retrieved chunks", len(chunks) > 0):
            ok = False
            continue
        for c in chunks:
            print(f"  - [{chain.label(c)}] rerank_score={c['rerank_score']:.3f}")

        ok &= check("BM25 index built", chain.bm25 is not None, "collection may be empty")

        try:
            answer = chain.ask(query)
        except Exception as e:
            check("Groq call succeeds", False, str(e))
            ok = False
            continue
        check("Groq call succeeds", True)
        print(f"Answer: {answer}")

    print("\n" + ("All checks passed." if ok else "Some checks failed."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
