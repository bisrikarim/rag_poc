"""
ingest.py — Ingestion des PDFs dans ChromaDB
Usage: python ingest.py --pdf_dir ./pdfs
"""

import argparse
import os
import re
import sys

import fitz  # pymupdf
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR  = "./chroma_db"
COLLECTION  = "aia_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE  = 500   # caractères
CHUNK_OVERLAP = 80
# ─────────────────────────────────────────────────────────────────────────────


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte brut d'un PDF avec PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n".join(pages)


def chunk_text(text: str, source: str) -> list[dict]:
    """Découpe le texte en chunks avec overlap."""
    # Nettoyage basique
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({
                "id":       f"{source}__chunk{idx}",
                "text":     chunk.strip(),
                "metadata": {"source": source, "chunk_index": idx}
            })
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def ingest(pdf_dir: str):
    print(f"📂 Dossier PDFs : {pdf_dir}")

    # Chargement du modèle d'embeddings
    print(f"🤖 Chargement du modèle d'embeddings : {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)

    # Init ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # Supprime la collection si elle existe déjà (re-ingestion propre)
    try:
        client.delete_collection(COLLECTION)
        print("🗑️  Ancienne collection supprimée.")
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)
    print(f"✅ Collection ChromaDB créée : '{COLLECTION}'")

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("❌ Aucun PDF trouvé dans le dossier.")
        sys.exit(1)

    total_chunks = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\n📄 Traitement : {pdf_file}")

        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print("  ⚠️  Aucun texte extrait (PDF image ?), ignoré.")
            continue

        chunks = chunk_text(text, source=pdf_file)
        print(f"  → {len(chunks)} chunks générés")

        # Embeddings par batch
        texts     = [c["text"]     for c in chunks]
        ids       = [c["id"]       for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        embeddings = model.encode(texts, show_progress_bar=True).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        total_chunks += len(chunks)

    print(f"\n✅ Ingestion terminée — {total_chunks} chunks stockés dans ChromaDB ({CHROMA_DIR})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="./pdfs", help="Dossier contenant les PDFs")
    args = parser.parse_args()
    ingest(args.pdf_dir)