import faiss
import pickle
import numpy as np
import requests
import os
import json
from sentence_transformers import SentenceTransformer

INDEX_PATH = "vectorstore/faiss.index"
CHUNKS_PATH = "vectorstore/chunks.pkl"
MEDIA_INDEX_PATH = "vectorstore/media_index.json"
TOP_K = 5
OLLAMA_MODEL = "phi3"

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    all_chunks = pickle.load(f)

try:
    with open(MEDIA_INDEX_PATH, "r") as f:
        media_index = json.load(f)
except:
    media_index = []

def retrieve(query: str):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")
    distances, indices = index.search(query_vector, TOP_K)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                "text": all_chunks[idx]["text"],
                "source": all_chunks[idx]["source"],
                "score": float(distances[0][i])
            })
    return results

def find_related_media(query: str, answer: str = "", max_results: int = 3):
    stop_words = {"the", "a", "an", "in", "on", "at", "to", "for",
                  "of", "and", "or", "is", "was", "were", "are",
                  "what", "who", "how", "did", "do", "does", "with",
                  "from", "by", "about", "that", "this", "it", "be",
                  "have", "has", "had", "not", "but", "they", "he",
                  "she", "we", "you", "i", "me", "my", "his", "her",
                  "named", "newly", "unsealed", "documents", "jeffrey",
                  "epstein", "files", "legal", "services", "magna",
                  "case", "court", "document", "page", "all"}

    query_words = set(query.lower().split()) - stop_words

    if not query_words:
        return []

    scored = []
    for item in media_index:
        # Only match against extracted images
        if "extracted_images" not in item["path"]:
            continue

        filename_words = set(item["keywords"].split()) - stop_words
        matches = query_words & filename_words

        if len(matches) >= 1:
            scored.append((len(matches), item))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [item for _, item in scored[:max_results]]

def build_prompt(query: str, chunks: list):
    context = "\n\n---\n\n".join(
        [f"[Source: {c['source']}]\n{c['text']}" for c in chunks]
    )
    return f"""You are an expert research assistant analyzing the Epstein files.
Answer questions in detailed, comprehensive paragraphs based STRICTLY on the document excerpts below.
Include specific names, dates, case numbers, and legal details wherever available.
If the answer is not in the documents say: "This information is not found in the available Epstein files."
Do not speculate beyond what the documents contain.

DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{query}

ANSWER:"""

def ask(query: str):
    chunks = retrieve(query)

    if not chunks:
        return {
            "answer": "No relevant documents found for your query.",
            "sources": [],
            "media": []
        }

    prompt = build_prompt(query, chunks)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    answer = response.json().get("response", "Error generating response.")
    media = find_related_media(query, answer)

    return {
        "answer": answer,
        "sources": list(set(c["source"] for c in chunks)),
        "media": media
    }