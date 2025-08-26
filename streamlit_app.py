import os
from pathlib import Path
import streamlit as st

from src.config import (
    DEFAULT_DOCS_DIR,
    DEFAULT_CHROMA_DIR,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_EMBEDDING_MODEL,
)
from src.ingest import build_or_update_vectorstore
from src.rag import build_chain

st.set_page_config(page_title="AI Research Assistant (RAG)", page_icon="📚", layout="wide")

# --- Sidebar: Settings ---
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
    type="password",
    help="Get one from platform.openai.com",
)
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

model_name = st.sidebar.text_input("Chat Model", value=os.getenv("OPENAI_MODEL", OPENAI_MODEL))
embed_model = st.sidebar.text_input("Embedding Model", value=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))

persist_dir = st.sidebar.text_input("Chroma Persist Dir", value=DEFAULT_CHROMA_DIR)
docs_dir = st.sidebar.text_input("Docs Directory", value=DEFAULT_DOCS_DIR)

k = st.sidebar.slider("Top-K chunks", min_value=2, max_value=10, value=4)
mmr = st.sidebar.checkbox("Use MMR (diverse retrieval)", value=True)
temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.2, 0.1)

st.sidebar.markdown("---")
if st.sidebar.button("♻️ Reset Vector Store"):
    from src.ingest import reset_vectorstore
    reset_vectorstore(persist_dir)
    st.sidebar.success("Vector store reset. Re-ingest documents.")

# --- Main: Header ---
st.title("📚 AI Personal Research Assistant (RAG)")
st.caption("Load PDFs → Build Vector Store → Ask source-grounded questions with citations.")

# --- Upload PDFs ---
st.subheader("📤 Upload PDFs")
uploaded_files = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    Path(docs_dir).mkdir(parents=True, exist_ok=True)
    for f in uploaded_files:
        dest = Path(docs_dir) / f.name
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
    st.success(f"Saved {len(uploaded_files)} file(s) to {docs_dir}")

if st.button("🧠 Build/Update Index"):
    if not api_key:
        st.error("Please provide your OpenAI API key in the sidebar.")
    else:
        with st.spinner("Building vector store..."):
            vs, n = build_or_update_vectorstore(doc_dir=docs_dir, persist_dir=persist_dir)
        if n > 0:
            st.success(f"Indexed {n} chunks from PDFs in {docs_dir} 👍")
        else:
            st.warning("No PDFs found. Upload some and try again.")

# --- Chat Section ---
st.subheader("💬 Chat with your documents")
if "chain" not in st.session_state:
    try:
        chain, retriever = build_chain(
            model_name=model_name,
            temperature=temperature,
            persist_dir=persist_dir,
            k=k,
            mmr=mmr,
        )
        st.session_state.chain = chain
        st.session_state.retriever = retriever
    except Exception as e:
        st.session_state.chain = None
        st.error(f"Initialize chain after you build the index. Details: {e}")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

user_q = st.text_input("Ask a question about your PDFs:")
go = st.button("🚀 Ask")

def render_sources(query: str, k: int = 6):
    try:
        docs = st.session_state.retriever.get_relevant_documents(query)
        st.markdown("**Sources**")
        for d in docs[:k]:
            src = Path(d.metadata.get("source", "unknown")).name
            page = d.metadata.get("page", "NA")
            with st.expander(f"{src} (p.{page})"):
                snippet = (d.page_content[:1000] + "…") if len(d.page_content) > 1000 else d.page_content
                st.write(snippet)
    except Exception as e:
        st.info(f"No sources available yet. {e}")

if go and user_q:
    if not api_key:
        st.error("Please provide your OpenAI API key in the sidebar.")
    else:
        with st.spinner("Thinking..."):
            try:
                if st.session_state.chain is None:
                    st.session_state.chain, st.session_state.retriever = build_chain(
                        model_name=model_name,
                        temperature=temperature,
                        persist_dir=persist_dir,
                        k=k,
                        mmr=mmr,
                    )
                out = st.session_state.chain.invoke(user_q)
                answer = out.content if hasattr(out, "content") else str(out)
                st.session_state.history.append(("user", user_q))
                st.session_state.history.append(("assistant", answer))
            except Exception as e:
                st.error(f"Error: {e}")

# Render chat
for role, msg in st.session_state.history[-10:]:
    if role == "user":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Assistant:** {msg}")

# Show sources for last user question
if user_q:
    render_sources(user_q)
