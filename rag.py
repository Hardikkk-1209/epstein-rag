import faiss
import pickle
import numpy as np
import requests
import os
from sentence_transformers import SentenceTransformer

INDEX_PATH = "vectorstore/faiss.index"
CHUNKS_PATH = "vectorstore/chunks.pkl"
TOP_K = 3
OLLAMA_MODEL = "phi3"

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    all_chunks = pickle.load(f)

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

def build_prompt(query: str, chunks: list):
    context = "\n\n---\n\n".join(
        [f"[Source: {c['source']}]\n{c['text']}" for c in chunks]
    )
    return f"""You are an expert research assistant analyzing the Epstein files.
Answer questions in DETAILED, comprehensive paragraphs based STRICTLY on the document excerpts below.
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
            "sources": []
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

    return {
        "answer": answer,
        "sources": list(set(c["source"] for c in chunks))
    }