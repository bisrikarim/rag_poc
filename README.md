# AIA RAG POC

> Retrieval-Augmented Generation on AIA / AlethAIA Confluence documentation.  
> 100% local — no cloud, no API key required.

**Stack:** PyMuPDF · Sentence-Transformers · ChromaDB · FastAPI · Ollama (mistral:7b)

---

## Quick Start

### 1. Clone & create virtual environment

```powershell
git clone https://github.com/bisrikarim/rag_poc.git
cd rag_poc

python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Pull the Ollama model

```powershell
ollama pull mistral:7b
```

### 4. Add your PDFs and ingest

```powershell
# Drop your PDFs into the pdfs/ folder, then:
python ingest.py --pdf_dir ./pdfs
```

### 5. Start the application (2 terminals)

**Terminal 1 — Ollama**
```powershell
ollama serve
```

**Terminal 2 — API server**
```powershell
.venv\Scripts\Activate.ps1
uvicorn server:app --reload --port 8000
```

### 6. Open the UI

**http://localhost:8000**

---

## Project Structure

```
rag_poc/
├── ingest.py           ← Parse PDFs → chunks → embeddings → ChromaDB
├── server.py           ← FastAPI server + Ollama query
├── static/
│   └── index.html      ← Web interface
├── pdfs/               ← Drop your PDFs here (gitignored)
├── chroma_db/          ← Vector database, auto-created (gitignored)
├── requirements.txt
└── DOCUMENTATION.md    ← Full documentation (RAG intro, RAG vs Fine-Tuning, user manual)
```

---

## Full Documentation

See **[DOCUMENTATION.md](./DOCUMENTATION.md)** for:
- Introduction to RAG
- RAG vs Fine-Tuning comparison
- Detailed architecture
- Complete user manual
- Troubleshooting guide

---

*Note: `pdfs/`, `chroma_db/`, and `.venv/` are gitignored — your documents never leave your machine.*