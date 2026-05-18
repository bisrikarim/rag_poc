# RAG POC

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=flat-square&logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-mistral%3A7b-black?style=flat-square&logo=ollama&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5%2B-FF6B35?style=flat-square&logo=databricks&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-3.0%2B-F5A623?style=flat-square&logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Local Only](https://img.shields.io/badge/100%25-Local%20%26%20Private-27e8a7?style=flat-square&logo=shield&logoColor=white)

> Retrieval-Augmented Generation on internal Confluence documentation.  
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