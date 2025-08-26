# 📚 AI Personal Research Assistant (RAG + LLMs)

An end-to-end **Retrieval-Augmented Generation** app: upload PDFs (research papers, notes, docs), index them with **ChromaDB**, and ask questions via **OpenAI LLMs**. Answers are grounded in your documents and include **source citations**. Built with **LangChain** + **Streamlit**.



## ✨ Features

- **📥 PDF ingestion** — drag & drop one or many PDFs
- **🔎 Chunking + Embedding** — smart splits with `RecursiveCharacterTextSplitter`
- **🧠 Vector store (persistent)** — `ChromaDB` stored in `./storage/chroma`
- **💬 Chat UI** — Streamlit interface with history
- **📌 Citations** — shows filenames + pages; expandable source snippets
- **🎛️ Controls** — change top-K, MMR, temperature, model names
- **♻️ Reset** — one click to wipe and rebuild the index



## 🏗️ Architecture (RAG Flow)

1. **Load PDFs** → `PyPDFLoader`
2. **Split** into overlapping chunks → `RecursiveCharacterTextSplitter`
3. **Embed** chunks → `OpenAIEmbeddings (text-embedding-3-small)`
4. **Store** vectors → `Chroma` (persisted to disk)
5. **Retrieve** top-K chunks via similarity or **MMR** (diverse)
6. **Generate** final answer with **ChatOpenAI** using a grounded prompt
7. **Cite** sources → `(source: filename p.<page>)`

```
PDFs → Loader → Splitter → Embeddings → Chroma (persist)
                                  ↑                     ↓
                             Retriever  ←  Question  → LLM (Answer + Citations)
```



## 🧰 Tech Stack

- **LangChain** (`langchain`, `langchain-community`, `langchain-openai`)
- **ChromaDB** for vector storage
- **OpenAI** GPT (default: `gpt-4o-mini`) and embeddings (`text-embedding-3-small`)
- **Streamlit** for the UI
- **pypdf** for robust PDF parsing



## 🚀 Quickstart

### 1) Clone & install
```bash
git clone https://github.com/yourname/ai-research-assistant.git
cd ai-research-assistant
python -m venv .venv && source .venv/bin/activate # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Configure keys
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

> You can also paste your key in the Streamlit sidebar at runtime.

### 3) Run the app
```bash
streamlit run streamlit_app.py
```

Open the displayed local URL in your browser.


## 🖥️ Using the App

1. **Upload PDFs** under “📤 Upload PDFs”
2. Click **🧠 Build/Update Index** to embed and persist them (first time required)
3. Ask questions in **💬 Chat**  
4. Expand **Sources** to inspect the actual snippets and pages used

**Tips**  
- Use **specific** questions for better grounding  
- Increase **Top-K** to gather more evidence; enable **MMR** for diversity  
- If answers look off, **Reset Vector Store** and re-index



## 🔧 Configuration

Edit `.env` or use sidebar inputs:

| Key | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI access | _required_ |
| `OPENAI_MODEL` | Chat model | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | Embeddings | `text-embedding-3-small` |
| `CHROMA_DIR` | Vector store path | `./storage/chroma` |
| `DOCS_DIR` | PDF storage | `./data/uploads` |
| `CHUNK_SIZE` | Chunk chars | `1000` |
| `CHUNK_OVERLAP` | Overlap chars | `200` |



## 🧠 How It Works (Deeper Dive)

### Chunking
We split docs with multiple separators (`\n\n`, `\n`, space, empty) to preserve semantics while filling chunks to ~1000 characters with **200 overlap**. This helps the retriever catch cross-sentence reasoning.

### Embeddings & Vector Store
Each chunk is embedded via `OpenAIEmbeddings` and stored in **Chroma** with metadata:
- `source`: original file path
- `page`: page number

### Retrieval
The retriever supports:
- **Similarity** search (`k` nearest)
- **MMR** (Maximal Marginal Relevance): balances relevance + diversity (great for long PDFs)

### Generation
A concise, **context-only** prompt is given to `ChatOpenAI`. If the fact isn’t present in context, the model is instructed to say it doesn’t know — reducing hallucination.



## 🧪 Quality Tips

- **Multiple PDFs?** Index them together — RAG will pull the best chunks across files
- **Ask follow-ups** that reference previous answers; RAG remains grounded
- **Tune `k`**: try 4–8 for most research papers
- **Check sources**: if a claim looks surprising, expand the source snippet and verify



## 🔒 Privacy & Localness

- Your PDFs are stored locally under `./data/uploads`  
- The vector store is persisted under `./storage/chroma`  
- Only **text embeddings & prompts** are sent to OpenAI’s API

> Want fully local inference? You can swap `OpenAIEmbeddings` and `ChatOpenAI` for local providers (e.g., `HuggingFaceEmbeddings` + `Ollama`) with minor code changes.



## 🛠️ Common Issues & Fixes

- **No API key / auth errors**  
  Set `OPENAI_API_KEY` in `.env` or paste in the sidebar.

- **“No PDFs found” after ingest**  
  Ensure files are saved to `DOCS_DIR` (shown in sidebar). Re-click **Build/Update Index**.

- **Weird answers**  
  Increase **Top-K** and enable **MMR**. Re-ingest after adding more PDFs.

- **Version conflicts**  
  Use the exact `requirements.txt` and a fresh virtual environment.



## 🧩 Extending the Project

- **Add a FastAPI backend** for multi-user serving
- **Citations UI** with clickable anchors to exact pages
- **Multi-modal**: OCR scanned PDFs (e.g., `pytesseract`) + images
- **Summarization mode**: produce section-wise summaries per paper


