# RAG POC — Documentation AIA

Stack : PyMuPDF · Sentence-Transformers · ChromaDB · FastAPI · Ollama (mistral:7b)
Tout tourne en local, aucune clé API requise.

---

## 1. Créer et activer le venv

```powershell
# Dans le dossier du projet
python -m venv .venv

# Activation (PowerShell)
.venv\Scripts\Activate.ps1

# Activation (CMD)
.venv\Scripts\activate.bat
```

Tu dois voir `(.venv)` au début de ta ligne de commande.

---

## 2. Installer les dépendances

```powershell
pip install pymupdf sentence-transformers chromadb fastapi uvicorn requests
```

---

## 3. Télécharger le modèle Ollama

```powershell
ollama pull mistral:7b
```

Taille : ~4.1 GB — à faire une seule fois.

---

## 4. Structure du projet

```
rag_poc/
├── ingest.py          <- ingestion des PDFs
├── server.py          <- serveur FastAPI + Ollama
├── static/
│   └── index.html     <- interface web
├── pdfs/              <- mets tes PDFs ici
├── chroma_db/         <- créé automatiquement
└── .venv/             <- environnement Python isolé
```

---

## 5. Ingestion des PDFs

Place tes PDFs dans `pdfs/` puis, avec le venv activé :

```powershell
python ingest.py --pdf_dir ./pdfs
```

Durée estimée : 1-3 minutes pour 6 PDFs.
La base ChromaDB est persistée dans `./chroma_db/` — pas besoin de re-ingérer à chaque démarrage.

---

## 6. Lancer le POC (2 terminaux)

**Terminal 1 — Ollama**
```powershell
ollama serve
```

**Terminal 2 — Serveur FastAPI (venv activé)**
```powershell
.venv\Scripts\Activate.ps1
uvicorn server:app --reload --port 8000
```

---

## 7. Ouvrir l'interface

http://localhost:8000

Le point de statut en haut à droite passe au vert quand tout est prêt.

---

## Notes

- Réponse Ollama : ~30-90s sur CPU (mistral:7b Q4 sans GPU dédié — normal)
- Pour re-ingérer après ajout de PDFs : relancer `python ingest.py --pdf_dir ./pdfs`
- Pour changer de modèle : modifier `OLLAMA_MODEL` dans `server.py` + `ollama pull <modele>`