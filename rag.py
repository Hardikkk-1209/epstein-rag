import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from pinecone import Pinecone

# ─────────────────────────────────────────────
# LOAD ENV
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"), override=False)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "epstein-rag")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set.")
if not PINECONE_API_KEY:
    print("WARNING: PINECONE_API_KEY not set.")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TOP_K_FETCH = 40
TOP_K_FINAL = 5

# ─────────────────────────────────────────────
# LAZY LOADING
# ─────────────────────────────────────────────

_model = None
_index = None
_gemini_client = None


def get_model():
    global _model
    if _model is None:
        print("Loading SentenceTransformer model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_pinecone_index():
    global _index
    if _index is None:
        print("Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = pc.Index(PINECONE_INDEX_NAME)
        print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    return _index


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


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
# RETRIEVAL via PINECONE
# ─────────────────────────────────────────────

def retrieve(query: str, active_filters=None):
    model = get_model()
    index = get_pinecone_index()

    query_vector = model.encode([query])[0].tolist()

    fetch_k = TOP_K_FETCH if active_filters else TOP_K_FINAL

    results = index.query(
        vector=query_vector,
        top_k=fetch_k,
        include_metadata=True
    )

    candidates = []
    for match in results.matches:
        metadata = match.metadata or {}
        candidates.append({
            "text": metadata.get("text", ""),
            "source": metadata.get("source", "Unknown"),
            "score": float(match.score),
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
        response = get_gemini_client().models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt,
        )
        answer = extract_text(response)
    except Exception as e:
        answer = f"Model error: {str(e)}"

    return {
        "answer": answer,
        "sources": list(set(c["source"] for c in chunks)),
        "media": [],
        "mode": mode,
        "filters_applied": filters or [],
    }