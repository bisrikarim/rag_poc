"""
server.py — Serveur FastAPI RAG local (Ollama + ChromaDB + Historique persistant)
Usage: uvicorn server:app --reload --port 8000
"""

import json
import uuid
import requests
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR    = "./chroma_db"
COLLECTION    = "docs"
EMBED_MODEL   = "all-MiniLM-L6-v2"
TOP_K         = 5
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "mistral:7b"
HISTORY_FILE  = "./conversations.json"
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chargement au démarrage ───────────────────────────────────────────────────
print("🤖 Chargement du modèle d'embeddings…")
embed_model = SentenceTransformer(EMBED_MODEL)

print("📦 Connexion à ChromaDB…")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION)

print("✅ Serveur prêt — modèle Ollama :", OLLAMA_MODEL)


# ── Helpers historique ────────────────────────────────────────────────────────
def load_history() -> dict:
    if Path(HISTORY_FILE).exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_history(history: dict):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ── Modèles Pydantic ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None   # None = nouvelle conversation


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    conversation_id: str
    title: str


class ConversationMeta(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


# ── Routes ────────────────────────────────────────────────────────────────────
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


@app.get("/conversations", response_model=list[ConversationMeta])
def list_conversations():
    history = load_history()
    convs = []
    for conv_id, conv in history.items():
        convs.append(ConversationMeta(
            id=conv_id,
            title=conv["title"],
            created_at=conv["created_at"],
            updated_at=conv["updated_at"],
            message_count=len(conv["messages"]),
        ))
    # Plus récentes en premier
    convs.sort(key=lambda x: x.updated_at, reverse=True)
    return convs


@app.get("/conversations/{conv_id}")
def get_conversation(conv_id: str):
    history = load_history()
    if conv_id not in history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return history[conv_id]


@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    history = load_history()
    if conv_id not in history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    del history[conv_id]
    save_history(history)
    return {"deleted": conv_id}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question vide.")

    history = load_history()

    # Créer ou charger la conversation
    if req.conversation_id and req.conversation_id in history:
        conv = history[req.conversation_id]
        conv_id = req.conversation_id
    else:
        conv_id = str(uuid.uuid4())
        conv = {
            "id": conv_id,
            "title": req.question[:60] + ("…" if len(req.question) > 60 else ""),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
        }

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

    # 3. Contexte documentaire
    context_parts = []
    for doc, meta in zip(docs, metadatas):
        context_parts.append(
            f"[Source: {meta['source']} | chunk {meta['chunk_index']}]\n{doc}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # 4. Historique de conversation pour le prompt (max 6 derniers échanges)
    history_text = ""
    for msg in conv["messages"][-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    # 5. Prompt RAG + historique (format Mistral [INST])
    prompt = f"""[INST] Tu es un assistant expert en documentation technique.
Reponds uniquement en te basant sur le contexte documentaire fourni.
Si la reponse n'est pas dans le contexte, dis-le clairement.
Reponds en francais, de facon precise et structuree.

Contexte documentaire :
{context}

{"Historique de la conversation :" + chr(10) + history_text if history_text else ""}

Question : {req.question} [/INST]"""

    # 6. Appel Ollama
    try:
        ollama_resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 1024, "num_ctx": 4096}
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

    # 7. Sauvegarder les messages dans l'historique
    conv["messages"].append({"role": "user",      "content": req.question})
    conv["messages"].append({"role": "assistant",  "content": answer})
    conv["updated_at"] = datetime.now().isoformat()

    history[conv_id] = conv
    save_history(history)

    # 8. Sources
    sources = [
        {
            "source":      meta["source"],
            "chunk_index": meta["chunk_index"],
            "score":       round(1 - dist, 3),
            "excerpt":     doc[:200] + "..." if len(doc) > 200 else doc
        }
        for doc, meta, dist in zip(docs, metadatas, distances)
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        conversation_id=conv_id,
        title=conv["title"],
    )


# Sert l'interface web statique
app.mount("/", StaticFiles(directory="static", html=True), name="static")