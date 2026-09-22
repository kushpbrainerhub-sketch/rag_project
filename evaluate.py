"""Evaluate RAG quality with RAGAS: faithfulness, answer relevancy, context precision/recall."""

import os
import sys
import types

# ragas (as of 0.2.x/0.4.x) unconditionally imports a langchain_community submodule for
# Vertex AI that no longer ships in current langchain-community releases (its integrations
# are being split into standalone packages). We only ever use Groq, never Vertex AI, so this
# stub satisfies that dead import without patching the installed package or pinning an old
# langchain-community (which would drag in a much older, incompatible langchain).
_stub = types.ModuleType("langchain_community.chat_models.vertexai")


class ChatVertexAI:  # never instantiated, just needs to exist for the import to succeed
    pass


_stub.ChatVertexAI = ChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _stub)

from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from rag_chain import GROQ_MODEL, RagChain

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


class SentenceTransformerEmbeddings(Embeddings):
    """Adapts our already-loaded SentenceTransformer to LangChain's Embeddings interface,
    so RAGAS can reuse it instead of downloading/loading a separate embedding model."""

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text])[0].tolist()

# Extend this with real question/ground-truth pairs for your own documents.
EVAL_SET = [
    {
        "question": "How many weeks does the learning roadmap in this book span?",
        "ground_truth": "The roadmap spans 16 weeks, about 4 months.",
    },
    {
        "question": "What should you do right after finishing the 16-week program?",
        "ground_truth": (
            "Move into the job-search phase: start sending applications and doing "
            "screening calls and technical interviews."
        ),
    },
    {
        "question": "What broad range of topics does the book cover?",
        "ground_truth": (
            "Python fundamentals, classical machine learning, deep learning including "
            "transformer architectures, system design, and interview preparation."
        ),
    },
]


def build_eval_dataset(chain: RagChain) -> Dataset:
    rows = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}
    for item in EVAL_SET:
        question = item["question"]
        chunks = chain.retrieve(question)
        answer = chain.generate(question, chunks, history=[]) if chunks else "No relevant documents found."

        rows["user_input"].append(question)
        rows["response"].append(answer)
        rows["retrieved_contexts"].append([c["text"] for c in chunks])
        rows["reference"].append(item["ground_truth"])

        print(f"Q: {question}\nA: {answer}\n")

    return Dataset.from_dict(rows)


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set in .env")
        return

    chain = RagChain()
    if chain.collection.count() == 0:
        print("No documents ingested. Run ingest.py first.")
        return

    print(f"Running {len(EVAL_SET)} eval question(s) through the pipeline...\n")
    dataset = build_eval_dataset(chain)

    judge_llm = LangchainLLMWrapper(
        ChatGroq(model=GROQ_MODEL, api_key=os.environ["GROQ_API_KEY"], temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(SentenceTransformerEmbeddings(chain.embedder))

    # Groq's free-tier rate limits choke under ragas's default concurrency (16 workers),
    # which causes judge calls to queue behind 429s and eventually time out. A small,
    # patient worker pool trades speed for actually finishing every call.
    run_config = RunConfig(max_workers=2, timeout=120)

    print("Scoring with RAGAS (this makes several LLM judge calls per question)...\n")
    result = evaluate(
        dataset, metrics=METRICS, llm=judge_llm, embeddings=judge_embeddings, run_config=run_config
    )

    df = result.to_pandas()
    metric_cols = [m.name for m in METRICS]
    print(df[["user_input", *metric_cols]].to_string(index=False))

    print("\nAverages:")
    for name in metric_cols:
        print(f"  {name}: {df[name].mean():.3f}")


if __name__ == "__main__":
    main()
