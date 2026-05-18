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
COLLECTION  = "docs"
EMBED_MODEL   = "paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE    = 800   # caractères
CHUNK_OVERLAP = 200
# ─────────────────────────────────────────────────────────────────────────────


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte brut d'un PDF avec PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n".join(pages)


def chunk_text(text: str, source: str) -> list[dict]:
    """Découpe le texte en chunks sémantiques basés sur les paragraphes."""
    # Nettoyage
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # Découpage par paragraphes (sauts de ligne doubles)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    chunks = []
    idx = 0
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # Si le paragraphe seul dépasse CHUNK_SIZE, on le découpe à taille fixe
        if para_len > CHUNK_SIZE:
            # Flush le buffer courant d'abord
            if current_parts:
                chunk_text_val = " ".join(current_parts).strip()
                chunks.append({
                    "id":       f"{source}__chunk{idx}",
                    "text":     chunk_text_val,
                    "metadata": {"source": source, "chunk_index": idx}
                })
                idx += 1
                current_parts = []
                current_len = 0
            # Découpage fixe avec overlap
            start = 0
            while start < para_len:
                sub = para[start:start + CHUNK_SIZE]
                if sub.strip():
                    chunks.append({
                        "id":       f"{source}__chunk{idx}",
                        "text":     sub.strip(),
                        "metadata": {"source": source, "chunk_index": idx}
                    })
                    idx += 1
                start += CHUNK_SIZE - CHUNK_OVERLAP
            continue

        # Si l'ajout du paragraphe dépasse CHUNK_SIZE, on flush puis on démarre un nouveau chunk
        # en conservant le dernier paragraphe comme overlap
        if current_len + para_len + 1 > CHUNK_SIZE and current_parts:
            chunk_text_val = " ".join(current_parts).strip()
            chunks.append({
                "id":       f"{source}__chunk{idx}",
                "text":     chunk_text_val,
                "metadata": {"source": source, "chunk_index": idx}
            })
            idx += 1
            # Overlap : on garde les derniers paragraphes qui tiennent dans CHUNK_OVERLAP
            overlap_parts: list[str] = []
            overlap_len = 0
            for p in reversed(current_parts):
                if overlap_len + len(p) + 1 <= CHUNK_OVERLAP:
                    overlap_parts.insert(0, p)
                    overlap_len += len(p) + 1
                else:
                    break
            current_parts = overlap_parts
            current_len = overlap_len

        current_parts.append(para)
        current_len += para_len + 1

    # Dernier chunk
    if current_parts:
        chunk_text_val = " ".join(current_parts).strip()
        chunks.append({
            "id":       f"{source}__chunk{idx}",
            "text":     chunk_text_val,
            "metadata": {"source": source, "chunk_index": idx}
        })

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