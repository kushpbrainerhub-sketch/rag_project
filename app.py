"""Streamlit chat UI for the RAG pipeline."""

import streamlit as st

from ingest import CHROMA_DIR
from rag_chain import RagChain

st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("📄 RAG Chat")


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

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** (distance={s['distance']:.4f})")

query = st.chat_input("Ask a question about your documents...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain.top_k = top_k
            chunks = chain.retrieve(query)
            answer = chain.generate(query, chunks) if chunks else "No relevant documents found."
        st.markdown(answer)
        if chunks:
            with st.expander("Sources"):
                for s in chunks:
                    st.markdown(f"- **{s['source']}** (distance={s['distance']:.4f})")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": chunks})
