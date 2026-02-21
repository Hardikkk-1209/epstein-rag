import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# ── Config ──
DATA_DIR = "data"
INDEX_PATH = "vectorstore/faiss.index"
CHUNKS_PATH = "vectorstore/chunks.pkl"
CHUNK_SIZE = 500        # words per chunk
CHUNK_OVERLAP = 50      # overlapping words between chunks

model = SentenceTransformer("all-MiniLM-L6-v2")  # free, fast, accurate

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return " ".join(page.get_text() for page in doc)

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text, source):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append({"text": chunk, "source": source})
    return chunks

def ingest():
    all_chunks = []

    for root, dirs, files in os.walk(DATA_DIR):
        for filename in files:
            path = os.path.join(root, filename)
            print(f"Processing: {filename}")

            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(path)
            elif filename.endswith(".txt"):
                text = extract_text_from_txt(path)
            else:
                continue

            chunks = chunk_text(text, source=filename)
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks extracted")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Generating embeddings...")

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"✅ FAISS index saved — {index.ntotal} vectors indexed")

if __name__ == "__main__":
    ingest()