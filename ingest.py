import os
import fitz
import faiss
import numpy as np
import pickle
import json
from sentence_transformers import SentenceTransformer

# ── Config ──
DATA_DIR = "data"
INDEX_PATH = "vectorstore/faiss.index"
CHUNKS_PATH = "vectorstore/chunks.pkl"
MEDIA_INDEX_PATH = "vectorstore/media_index.json"
EXTRACTED_IMAGES_DIR = "data/extracted_images"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

SUPPORTED_IMAGES = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
SUPPORTED_VIDEOS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

model = SentenceTransformer("all-MiniLM-L6-v2")

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

def extract_images_from_pdfs():
    os.makedirs(EXTRACTED_IMAGES_DIR, exist_ok=True)
    total = 0

    for root, dirs, files in os.walk(DATA_DIR):
        for filename in files:
            if not filename.endswith(".pdf"):
                continue
            path = os.path.join(root, filename)
            doc = fitz.open(path)

            for page_num, page in enumerate(doc):
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    img_name = f"{filename}_page{page_num+1}_img{img_index+1}.{image_ext}"
                    img_path = os.path.join(EXTRACTED_IMAGES_DIR, img_name)

                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    total += 1

    print(f"✅ Extracted {total} images from PDFs")

def build_media_index():
    media = []
    scan_dirs = [DATA_DIR, EXTRACTED_IMAGES_DIR]

    for scan_dir in scan_dirs:
        if not os.path.exists(scan_dir):
            continue
        for root, dirs, files in os.walk(scan_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                path = os.path.join(root, filename)

                if ext in SUPPORTED_IMAGES:
                    media.append({
                        "type": "image",
                        "filename": filename,
                        "path": path,
                        "keywords": filename.lower().replace("_", " ").replace("-", " ")
                    })
                elif ext in SUPPORTED_VIDEOS:
                    media.append({
                        "type": "video",
                        "filename": filename,
                        "path": path,
                        "keywords": filename.lower().replace("_", " ").replace("-", " ")
                    })

    with open(MEDIA_INDEX_PATH, "w") as f:
        json.dump(media, f, indent=2)

    print(f"✅ Media index saved — {len(media)} media files indexed")

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

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"✅ FAISS index saved — {index.ntotal} vectors indexed")

    # Extract images and build media index
    print("\nExtracting images from PDFs...")
    extract_images_from_pdfs()
    build_media_index()

if __name__ == "__main__":
    ingest()