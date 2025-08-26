from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from .config import (
    DEFAULT_CHROMA_DIR,
    OPENAI_MODEL,
    OPENAI_EMBEDDING_MODEL,
)

BASE_PROMPT = """You are an expert research assistant. Use ONLY the provided context to answer the user's question.
- If the answer is not in the context, say you don't know.
- Be concise, cite sources with (source: filename p.<page>) at the end.
- If there are multiple relevant snippets, synthesize them.

Question: {question}

Context:
{context}

Answer:
"""

def make_retriever(persist_dir: str = DEFAULT_CHROMA_DIR, k: int = 4, mmr: bool = True):
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    vs = Chroma(
        collection_name="research-assistant",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    retriever = vs.as_retriever(
        search_kwargs={"k": k, "fetch_k": max(k * 3, 12)} if mmr else {"k": k},
        search_type="mmr" if mmr else "similarity",
    )
    return retriever

def _format_docs(docs: List[Document]) -> str:
    formatted = []
    for d in docs:
        src = d.metadata.get("source", "unknown").split("/")[-1]
        page = d.metadata.get("page", "NA")
        formatted.append(f"[{src} p.{page}] {d.page_content}")
    return "\n\n---\n\n".join(formatted)

def build_chain(
    model_name: str = OPENAI_MODEL,
    temperature: float = 0.2,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    k: int = 4,
    mmr: bool = True,
):
    retriever = make_retriever(persist_dir=persist_dir, k=k, mmr=mmr)
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    prompt = ChatPromptTemplate.from_template(BASE_PROMPT)

    chain = {
        "context": retriever | _format_docs,
        "question": RunnablePassthrough(),
    } | prompt | llm

    return chain, retriever

def ask_question(chain, question: str) -> Dict[str, Any]:
    # returns an AIMessage (LC object) with .content
    ai_msg = chain.invoke(question)
    return {"answer": ai_msg.content}
