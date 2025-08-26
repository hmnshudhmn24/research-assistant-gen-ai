from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from .config import DEFAULT_DOCS_DIR, DEFAULT_CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_EMBEDDING_MODEL

def load_pdfs_from_dir(directory: str) -> List:
    docs = []
    dir_path = Path(directory)
    for pdf in dir_path.glob("**/*.pdf"):
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
    return docs

def split_docs(docs: List):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def build_or_update_vectorstore(
    doc_dir: str = DEFAULT_DOCS_DIR,
    persist_dir: str = DEFAULT_CHROMA_DIR,
):
    raw_docs = load_pdfs_from_dir(doc_dir)
    if not raw_docs:
        return None, 0

    chunks = split_docs(raw_docs)
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    # If store exists, we can add new docs; else create fresh
    vs = Chroma(
        collection_name="research-assistant",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # Add with metadata (source, page)
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    vs.add_texts(texts=texts, metadatas=metadatas)
    vs.persist()
    return vs, len(chunks)

def reset_vectorstore(persist_dir: str = DEFAULT_CHROMA_DIR):
    p = Path(persist_dir)
    if p.exists():
        for item in p.glob("**/*"):
            try:
                item.unlink()
            except IsADirectoryError:
                pass
        # remove empty dirs
        for d in sorted(p.glob("**/*"), reverse=True):
            if d.is_dir():
                try:
                    d.rmdir()
                except Exception:
                    pass
