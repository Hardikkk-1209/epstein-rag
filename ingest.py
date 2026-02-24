import os
import faiss
import numpy as np
import pickle
import json
from sentence_transformers import SentenceTransformer

# ── Config ──
CHUNKS_DIR = "data/chunks"        # ← folder with your .jsonl files
INDEX_PATH = "vectorstore/faiss.index"
CHUNKS_PATH = "vectorstore/chunks.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_jsonl_chunks():
    all_chunks = []

    if not os.path.exists(CHUNKS_DIR):
        print(f"❌ Chunks directory not found: {CHUNKS_DIR}")
        return all_chunks

    files = [f for f in os.listdir(CHUNKS_DIR) if f.endswith(".jsonl")]
    print(f"Found {len(files)} .jsonl files\n")

    for filename in sorted(files):
        path = os.path.join(CHUNKS_DIR, filename)
        file_chunks = 0

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    # Normalize to match what rag.py expects
                    all_chunks.append({
                        "text": chunk.get("text", ""),
                        "source": chunk.get("source_filename", filename),
                        "chunk_id": chunk.get("chunk_id", ""),
                        "doc_id": chunk.get("doc_id", ""),
                        "page_start": chunk.get("page_start", None),
                        "page_end": chunk.get("page_end", None),
                    })
                    file_chunks += 1
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  Skipping bad line in {filename}: {e}")

        print(f"  ✅ {filename} → {file_chunks} chunks")

    return all_chunks

def ingest():
    print("Loading chunks from JSONL files...\n")
    all_chunks = load_jsonl_chunks()

    if not all_chunks:
        print("❌ No chunks loaded. Check your data/chunks folder.")
        return

    # Filter out empty text
    all_chunks = [c for c in all_chunks if c["text"].strip()]
    print(f"\nTotal chunks loaded: {len(all_chunks)}")

    print("\nGenerating embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\n✅ Done! {index.ntotal} vectors indexed and saved.")

if __name__ == "__main__":
    ingest()