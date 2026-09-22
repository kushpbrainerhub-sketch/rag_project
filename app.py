"""Streamlit chat UI for the RAG pipeline."""

from pathlib import Path

import streamlit as st

from ingest import CHROMA_DIR, DATA_DIR, ingest_documents, load_file
from rag_chain import RagChain

st.set_page_config(page_title="RAG Chat", page_icon="assets/favicon.png", layout="centered")

logo_col, title_col = st.columns([1, 9], vertical_alignment="center")
with logo_col:
    st.image("assets/favicon.png", width=48)
with title_col:
    st.title("RAG Chat")
st.caption("Ask questions grounded in your own documents.")


@st.cache_resource
def load_chain():
    return RagChain()


try:
    chain = load_chain()
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    st.stop()

count = chain.collection.count()
if count == 0:
    st.warning(f"No documents in '{CHROMA_DIR}/'. Add files to data/ and run `python ingest.py` first.")

st.sidebar.metric("Chunks indexed", count)
top_k = st.sidebar.slider("Chunks to retrieve", min_value=1, max_value=10, value=chain.top_k)
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

st.sidebar.divider()
st.sidebar.subheader("Upload documents")
uploaded_files = st.sidebar.file_uploader(
    "Add PDFs, .txt, or .md files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)
if uploaded_files and st.sidebar.button(f"Ingest {len(uploaded_files)} file(s)"):
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)

    documents = []
    with st.spinner(f"Reading {len(uploaded_files)} file(s)..."):
        for uploaded in uploaded_files:
            dest = data_dir / uploaded.name
            dest.write_bytes(uploaded.getbuffer())
            file_docs = load_file(dest, uploaded.name)
            if file_docs:
                documents.extend(file_docs)
            else:
                st.sidebar.warning(f"No text extracted from {uploaded.name}")

    if documents:
        with st.spinner(f"Embedding {len(documents)} document(s)..."):
            n_chunks = ingest_documents(documents, chain.embedder, chain.collection)
            chain.refresh_index()
        st.sidebar.success(f"Ingested {n_chunks} chunks from {len(documents)} file(s).")
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

def render_sources(chunks: list[dict]) -> None:
    with st.expander("Sources"):
        for s in chunks:
            st.markdown(f"- **{chain.label(s)}** (relevance={s['rerank_score']:.3f})")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

query = st.chat_input("Ask a question about your documents...")
if query:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain.top_k = top_k
            standalone_query = chain.condense_question(query, history)
            chunks = chain.retrieve(standalone_query)
            answer = chain.generate(query, chunks, history) if chunks else "No relevant documents found."
        st.markdown(answer)
        if chunks:
            render_sources(chunks)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": chunks})
