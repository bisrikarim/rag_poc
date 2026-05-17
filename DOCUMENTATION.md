# AIA RAG — Complete Documentation

> **Language:** English  
> **Project:** Retrieval-Augmented Generation POC on AIA / AlethAIA Confluence documentation  
> **Stack:** PyMuPDF · Sentence-Transformers · ChromaDB · FastAPI · Ollama (mistral:7b)

---

## Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [RAG vs Fine-Tuning — Key Differences](#2-rag-vs-fine-tuning--key-differences)
3. [Project Architecture](#3-project-architecture)
4. [Installation & Setup](#4-installation--setup)
5. [User Manual](#5-user-manual)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. What is RAG?

### The Problem

Large Language Models (LLMs) like Mistral or GPT are trained on massive datasets — but that training has a **knowledge cutoff**. They know nothing about:

- Your internal documentation
- Events after their training date
- Private or confidential information specific to your organization

If you ask a raw LLM *"What are the Vault principals used in AIA métier?"*, it simply cannot know. It will either say it doesn't know, or — worse — hallucinate a plausible-sounding but wrong answer.

### The RAG Solution

**Retrieval-Augmented Generation (RAG)** solves this by giving the LLM the relevant information *at query time*, directly in the prompt.

The pipeline works in two phases:

#### Phase 1 — Indexing (done once)

```
Your PDFs
    │
    ▼
Text Extraction (PyMuPDF)
    │
    ▼
Chunking  ──── Split text into overlapping segments of ~500 chars
    │
    ▼
Embedding ──── Convert each chunk into a vector (numerical representation)
    │           using a sentence-transformer model
    ▼
Vector Store ── Store all vectors in ChromaDB (local database on disk)
```

#### Phase 2 — Query (done at each question)

```
User Question
    │
    ▼
Embed the question ──── Same model, same vector space
    │
    ▼
Similarity Search ────── Find the TOP-K most relevant chunks in ChromaDB
    │
    ▼
Build Prompt ─────────── Inject retrieved chunks as context into the LLM prompt
    │
    ▼
LLM Generation ────────── Ollama (mistral:7b) generates a grounded answer
    │
    ▼
Answer + Sources displayed in the web UI
```

The LLM never "memorized" your docs — it reads them on the fly, every single time.

---

## 2. RAG vs Fine-Tuning — Key Differences

Both RAG and Fine-Tuning are ways to make an LLM useful on a specific domain. They solve different problems and have very different trade-offs.

### Fine-Tuning

Fine-tuning means **retraining** a base model on your own dataset, so that the model's weights (internal parameters) are updated to reflect your domain knowledge.

```
Base Model  +  Your training data  ──►  New specialized model
```

**When it makes sense:**
- You want the model to adopt a specific writing style or tone
- Your domain uses very specific vocabulary or syntax (e.g. medical, legal)
- You need faster inference (smaller specialized model)
- The knowledge is stable and doesn't change often

**Limitations:**
- Expensive: requires GPU compute, expertise, and time
- Knowledge becomes stale: re-training needed for every doc update
- Risk of "catastrophic forgetting" — the model may lose general capabilities
- No traceability: you cannot know *which document* contributed to an answer
- Minimum dataset size: typically thousands of examples needed

### RAG

RAG keeps the base model **untouched** and retrieves context dynamically at query time.

```
Base Model  +  Vector DB (your docs)  ──►  Grounded answers at runtime
```

**When it makes sense:**
- Your documentation changes frequently (new pages, updates)
- You need source attribution ("this answer comes from document X")
- You want to add knowledge without retraining
- You have limited compute resources
- You're building a POC quickly

**Limitations:**
- Answer quality depends on retrieval quality (chunk size, embedding model)
- Long-context documents may be split poorly
- The LLM still needs to be capable enough to reason over the retrieved context
- Slower than fine-tuned models (retrieval + full prompt on each query)

### Summary Table

| Criteria | RAG | Fine-Tuning |
|---|---|---|
| **Cost** | Low (CPU/local possible) | High (GPU, weeks of work) |
| **Setup time** | Hours to days | Weeks to months |
| **Knowledge update** | Re-ingest PDFs (minutes) | Re-train the model (days) |
| **Source traceability** | ✅ Yes, by design | ❌ No |
| **Works on private data** | ✅ Yes, nothing leaves your machine | ✅ Yes, but data used for training |
| **Handles doc changes** | ✅ Immediately | ❌ Requires new training run |
| **Hallucination risk** | Low (grounded in context) | Medium (baked-in knowledge) |
| **Best for** | Q&A on internal docs, wikis, support | Style transfer, domain syntax |

### What this POC uses

This project is a **RAG** system. The choice is deliberate:

- AIA documentation on Confluence evolves regularly
- You need to trace which document answered which question
- The entire stack runs 100% locally on your laptop (no cloud, no GPU required)
- A new document is available instantly after re-running `ingest.py`

---

## 3. Project Architecture

```
rag_poc/
├── ingest.py           ← Step 1: Parse PDFs → chunks → embeddings → ChromaDB
├── server.py           ← Step 2: FastAPI server — search + Ollama call
├── static/
│   └── index.html      ← Web UI (plain HTML/CSS/JS, no framework)
├── pdfs/               ← Drop your PDFs here (gitignored)
├── chroma_db/          ← Auto-created vector database (gitignored)
├── .venv/              ← Python virtual environment (gitignored)
├── requirements.txt    ← Python dependencies
├── .gitignore          ← Excludes PDFs, venv, chroma_db
└── DOCUMENTATION.md    ← This file
```

### Data flow diagram

```
┌─────────────┐     PyMuPDF      ┌──────────┐   MiniLM    ┌────────────┐
│   PDFs      │ ──────────────►  │  Chunks  │ ──────────► │  ChromaDB  │
│  (./pdfs/)  │   text extract   │ ~500 chars│  embeddings │ (./chroma) │
└─────────────┘                  └──────────┘             └────────────┘
                                                                  │
                                                           vector search
                                                                  │
┌─────────────┐   HTTP/JSON   ┌───────────┐   Top-5 chunks  ┌────┴───────┐
│  Browser    │ ◄──────────── │  FastAPI  │ ◄────────────── │  Query     │
│ (localhost) │               │ :8000     │                  │  Embedding │
└─────────────┘               └─────┬─────┘                  └────────────┘
                                    │
                              prompt + context
                                    │
                              ┌─────▼─────┐
                              │  Ollama   │
                              │ mistral:7b│
                              │ :11434    │
                              └───────────┘
```

---

## 4. Installation & Setup

### Prerequisites

| Tool | Version | Check |
|---|---|---|
| Python | 3.10+ | `python --version` |
| Ollama | latest | `ollama --version` |
| Git | any | `git --version` |

### Step 1 — Clone the repository

```powershell
git clone https://github.com/YOUR_USERNAME/aia-rag-poc.git
cd aia-rag-poc
```

### Step 2 — Create and activate the virtual environment

```powershell
# Create the venv
python -m venv .venv

# Activate — PowerShell
.venv\Scripts\Activate.ps1

# Activate — CMD
.venv\Scripts\activate.bat
```

You should see `(.venv)` at the start of your prompt.

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

Expected output: all packages installed, no errors.  
The sentence-transformer model (~90 MB) will be downloaded automatically on first run.

### Step 4 — Pull the Ollama model

```powershell
ollama pull mistral:7b
```

This downloads ~4.1 GB once. You can verify it's available:

```powershell
ollama list
# Should show: mistral:7b
```

### Step 5 — Add your PDFs and ingest

```powershell
# Copy your PDFs into the pdfs/ folder, then:
python ingest.py --pdf_dir ./pdfs
```

Expected output:
```
📂 PDF folder: ./pdfs
🤖 Loading embedding model: all-MiniLM-L6-v2 …
✅ ChromaDB collection created: 'aia_docs'
📄 Processing: Configuration_ssh_...pdf
  → 12 chunks generated
...
✅ Ingestion complete — 87 chunks stored in ChromaDB
```

You only need to re-run `ingest.py` when you add or update documents.

---

## 5. User Manual

### Starting the application

You need **two terminals**, both with the venv activated.

**Terminal 1 — Start Ollama**

```powershell
ollama serve
```

Leave this running. Ollama will listen on `http://localhost:11434`.

**Terminal 2 — Start the API server**

```powershell
# Activate venv first
.venv\Scripts\Activate.ps1

# Start FastAPI
uvicorn server:app --reload --port 8000
```

Expected output:
```
🤖 Loading embedding model…
📦 Connecting to ChromaDB…
✅ Server ready — Ollama model: mistral:7b
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Open the web interface**

Navigate to: **http://localhost:8000**

---

### Using the web interface

#### Status indicator (top right)

| Color | Meaning |
|---|---|
| 🟢 Green | Server online, ChromaDB connected, shows chunk count |
| 🔵 Blue pulsing | A query is being processed |
| 🔴 Red | Server offline — check your terminals |

#### Asking a question

1. Type your question in the input bar at the bottom
2. Press **Enter** or click **Send →**
3. Wait for the response (~30–90 seconds on CPU — this is normal for a 7B model)

You can also click one of the **suggestion chips** to start with a pre-written example question.

#### Reading the answer

The answer appears in the main chat area.  
It is generated by Ollama (mistral:7b) based **only** on the content retrieved from your documents — not from general internet knowledge.

#### Sources panel (right sidebar)

For each answer, the **5 most relevant document chunks** are displayed:

- **Filename** — which PDF the chunk came from
- **Score** — relevance percentage (higher = more relevant to your question)
- **Excerpt** — first 200 characters of the chunk

This allows you to verify and trace every answer back to its source document.

---

### Example questions for AIA documentation

Here are questions that work well with the indexed documents:

```
What is the "golden ticket" problem in SSH authentication?
How does the Vault principal segmentation work?
What are the SSH prerequisites on target machines?
Explain the K3S ephemeral cluster deployment pipeline.
What is the role of the SOT in the AIA métier architecture?
How does JWT authentication work with Vault?
What happens if the SSH configuration is broken on a target?
What is the difference between the Producer and Consumer workloads?
```

---

### Adding new documents

1. Copy new PDFs into the `pdfs/` folder
2. Re-run ingestion (this replaces the existing index):

```powershell
python ingest.py --pdf_dir ./pdfs
```

3. Restart the FastAPI server to reload ChromaDB:

```powershell
# Ctrl+C to stop, then:
uvicorn server:app --reload --port 8000
```

---

### Stopping the application

- **Terminal 2** (FastAPI): `Ctrl+C`
- **Terminal 1** (Ollama): `Ctrl+C`

Your ChromaDB index is saved on disk — no need to re-ingest next time.

---

## 6. Troubleshooting

### Status dot stays red

The FastAPI server is not reachable.

```powershell
# Check that uvicorn is running on port 8000
# If you see an error like "Address already in use":
uvicorn server:app --reload --port 8001
# Then open http://localhost:8001
```

### "Ollama est hors ligne" error in the UI

The Ollama service is not running.

```powershell
# Start it in Terminal 1:
ollama serve
```

### Ollama takes too long / times out

Normal behavior on CPU for a 7B model. First query after starting is slower (model load).  
If it consistently times out, the timeout in `server.py` can be increased:

```python
# server.py, line ~80
timeout=300   # increase from 180 to 300 seconds
```

### "Collection not found" error on server start

The ChromaDB has not been created yet. Run ingestion first:

```powershell
python ingest.py --pdf_dir ./pdfs
```

### Answers seem irrelevant or off-topic

The retrieval may not be finding the right chunks. Try:
- Rephrasing your question with more specific terms
- Checking that the relevant PDF was actually ingested (look at the Sources panel)
- Re-running ingestion if you recently added documents

### PowerShell execution policy error on `.venv\Scripts\Activate.ps1`

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then retry activation.

---

## Pushing to GitHub

```powershell
# In the project folder
git init
git add .
git commit -m "feat: initial RAG POC on AIA documentation"

# Create the repo on github.com first, then:
git remote add origin https://github.com/YOUR_USERNAME/aia-rag-poc.git
git branch -M main
git push -u origin main
```

The `.gitignore` ensures that `pdfs/`, `chroma_db/`, and `.venv/` are **never pushed**.  
Only source code and documentation are committed.

---

*Last updated: May 2026*