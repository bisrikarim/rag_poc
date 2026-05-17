"""
server.py — Serveur FastAPI RAG local (Ollama + ChromaDB)
Usage: uvicorn server:app --reload --port 8000
"""

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR   = "./chroma_db"
COLLECTION   = "docs"
EMBED_MODEL  = "all-MiniLM-L6-v2"
TOP_K        = 5
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b"
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🤖 Chargement du modèle d'embeddings…")
embed_model = SentenceTransformer(EMBED_MODEL)

print("📦 Connexion à ChromaDB…")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION)

print("✅ Serveur prêt — modèle Ollama :", OLLAMA_MODEL)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {
        "status": "ok",
        "chunks_in_db": collection.count(),
        "ollama": "ok" if ollama_ok else "offline",
        "model": OLLAMA_MODEL,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question vide.")

    # 1. Embedding de la question
    q_embedding = embed_model.encode([req.question]).tolist()[0]

    # 2. Recherche vectorielle dans ChromaDB
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # 3. Construction du contexte
    context_parts = []
    for doc, meta in zip(docs, metadatas):
        context_parts.append(
            f"[Source: {meta['source']} | chunk {meta['chunk_index']}]\n{doc}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # 4. Prompt RAG (format Mistral [INST])
    prompt = f"""[INST] Tu es un assistant expert en documentation technique.
Reponds uniquement en te basant sur le contexte fourni.
Si la reponse n'est pas dans le contexte, dis-le clairement.
Reponds en francais, de facon precise et structuree.

Contexte extrait des documents :

{context}

---

Question : {req.question} [/INST]"""

    # 5. Appel Ollama (100% local)
    try:
        ollama_resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 1024,
                    "num_ctx":     4096,
                }
            },
            timeout=180
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=502, detail="Ollama est hors ligne. Lance 'ollama serve'.")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Ollama a mis trop de temps a repondre.")

    if ollama_resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama error: {ollama_resp.text}")

    answer = ollama_resp.json().get("response", "").strip()

    # 6. Sources retournées
    sources = [
        {
            "source":      meta["source"],
            "chunk_index": meta["chunk_index"],
            "score":       round(1 - dist, 3),
            "excerpt":     doc[:200] + "..." if len(doc) > 200 else doc
        }
        for doc, meta, dist in zip(docs, metadatas, distances)
    ]

    return QueryResponse(answer=answer, sources=sources)


# Sert l'interface web statique (index.html dans ./static/)
app.mount("/", StaticFiles(directory="static", html=True), name="static")