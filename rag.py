import faiss
import pickle
import numpy as np
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

# ─────────────────────────────────────────────
# LOAD ENV
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. Model calls will fail.")

client = genai.Client(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

INDEX_PATH = os.path.join(BASE_DIR, "vectorstore/faiss.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "vectorstore/chunks.pkl")
MEDIA_INDEX_PATH = os.path.join(BASE_DIR, "vectorstore/media_index.json")

TOP_K_FETCH = 40
TOP_K_FINAL = 5

# ─────────────────────────────────────────────
# LAZY LOADING — model/index load on first request
# ─────────────────────────────────────────────

_model = None
_index = None
_all_chunks = None
_media_index = None

def get_resources():
    global _model, _index, _all_chunks, _media_index

    if _model is None:
        print("Loading SentenceTransformer model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    if _index is None:
        print("Loading FAISS index...")
        _index = faiss.read_index(INDEX_PATH)

    if _all_chunks is None:
        print("Loading chunks...")
        with open(CHUNKS_PATH, "rb") as f:
            _all_chunks = pickle.load(f)

    if _media_index is None:
        try:
            with open(MEDIA_INDEX_PATH, "r") as f:
                _media_index = json.load(f)
        except:
            _media_index = []

    return _model, _index, _all_chunks, _media_index

# ─────────────────────────────────────────────
# FILTER CLASSIFIER
# ─────────────────────────────────────────────

FILTER_KEYWORDS = {
    "court": ["court", "judge", "docket", "indictment", "motion", "order", "complaint", "sdny"],
    "depo": ["deposition", "testimony", "sworn", "witness", "q:", "a:"],
    "flight": ["flight log", "flight manifest", "pilot", "aircraft", "visoski"],
    "media": ["miami herald", "new york times", "reporter", "journalist", "article"],
}

def classify_chunk(text: str) -> set:
    text_lower = text.lower()
    buckets = set()
    for category, keywords in FILTER_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            buckets.add(category)
    return buckets

# ─────────────────────────────────────────────
# EPSTEIN RELEVANCE CHECK
# ─────────────────────────────────────────────

EPSTEIN_KEYWORDS = {
    "epstein", "maxwell", "giuffre", "wexner", "prince andrew",
    "flight logs", "deposition", "trial", "indictment",
    "donald trump", "bill gates"
}

def is_epstein_related(query: str) -> bool:
    query_lower = query.lower()
    if len(query_lower.split()) <= 2:
        return True
    return any(keyword in query_lower for keyword in EPSTEIN_KEYWORDS)

# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

def retrieve(query: str, active_filters=None):
    model, index, all_chunks, _ = get_resources()

    query_vector = np.array(model.encode([query])).astype("float32")

    fetch_k = TOP_K_FETCH if active_filters else TOP_K_FINAL
    distances, indices = index.search(query_vector, fetch_k)

    candidates = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            candidates.append({
                "text": all_chunks[idx]["text"],
                "source": all_chunks[idx]["source"],
                "score": float(distances[0][i]),
            })

    if not active_filters:
        return candidates[:TOP_K_FINAL]

    filtered = [
        c for c in candidates
        if classify_chunk(c["text"]) & set(active_filters)
    ]

    if len(filtered) < 2:
        return candidates[:TOP_K_FINAL]

    return filtered[:TOP_K_FINAL]

# ─────────────────────────────────────────────
# MEDIA MATCHING
# ─────────────────────────────────────────────

def find_related_media(query: str, answer: str = "", max_results: int = 3):
    _, _, _, media_index = get_resources()

    stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "is"}
    query_words = set(query.lower().split()) - stop_words

    scored = []
    for item in media_index:
        if "extracted_images" not in item["path"]:
            continue

        filename_words = set(item["keywords"].split()) - stop_words
        matches = query_words & filename_words

        if matches:
            scored.append((len(matches), item))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [item for _, item in scored[:max_results]]

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

def build_prompt_strict(query: str, chunks: list):
    context = "\n\n---\n\n".join(
        [f"[Source: {c['source']}]\n{c['text']}" for c in chunks]
    )
    return f"""You are a legal research analyst reviewing Epstein court documents.

STRICT RULES:
- Answer ONLY from excerpts.
- Cite sources inline.
- If missing say: "This information is not found in the available Epstein files."

DOCUMENT EXCERPTS:
{context}

QUESTION:
{query}

ANSWER:"""

def build_prompt_summary(query: str, chunks: list):
    context = "\n\n---\n\n".join(
        [f"[Source: {c['source']}]\n{c['text']}" for c in chunks]
    )
    return f"""Summarize the findings clearly for a general audience.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{query}

SUMMARY:"""

def build_prompt_timeline(query: str, chunks: list):
    context = "\n\n---\n\n".join(
        [f"[Source: {c['source']}]\n{c['text']}" for c in chunks]
    )
    return f"""Create a chronological timeline of events.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{query}

TIMELINE:"""

MODE_PROMPT_BUILDERS = {
    "strict": build_prompt_strict,
    "summary": build_prompt_summary,
    "timeline": build_prompt_timeline,
}

# ─────────────────────────────────────────────
# SAFE GEMINI RESPONSE PARSER
# ─────────────────────────────────────────────

def extract_text(response):
    try:
        if hasattr(response, "text") and response.text:
            return response.text
        if hasattr(response, "candidates"):
            return response.candidates[0].content.parts[0].text
    except:
        pass
    return "No response generated."

# ─────────────────────────────────────────────
# MAIN ASK FUNCTION
# ─────────────────────────────────────────────

def ask(query: str, mode="strict", filters=None):

    if not is_epstein_related(query):
        return {
            "answer": "I can only answer questions related to the Epstein case, associated individuals, and related investigations.",
            "sources": [],
            "media": [],
        }

    chunks = retrieve(query, active_filters=filters if filters else None)

    if not chunks:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "media": [],
        }

    prompt_builder = MODE_PROMPT_BUILDERS.get(mode, build_prompt_strict)
    prompt = prompt_builder(query, chunks)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        answer = extract_text(response)
    except Exception as e:
        answer = f"Model error: {str(e)}"

    media = find_related_media(query, answer)

    return {
        "answer": answer,
        "sources": list(set(c["source"] for c in chunks)),
        "media": media,
        "mode": mode,
        "filters_applied": filters or [],
    }