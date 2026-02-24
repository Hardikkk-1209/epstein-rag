import faiss
import pickle
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from google import genai

# ── Config ──
INDEX_PATH = "vectorstore/faiss.index"
CHUNKS_PATH = "vectorstore/chunks.pkl"
MEDIA_INDEX_PATH = "vectorstore/media_index.json"

TOP_K = 5   # reduced to prevent token overflow

# 🔴 PUT YOUR GEMINI API KEY HERE
GEMINI_API_KEY = "AIzaSyC59vRsthxt6LnbTAQSQcmq5OVIxmExrfI"

# ── Gemini Client (NEW SDK) ──
client = genai.Client(api_key=GEMINI_API_KEY)

# ── Load models and index ──
model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    all_chunks = pickle.load(f)

try:
    with open(MEDIA_INDEX_PATH, "r") as f:
        media_index = json.load(f)
except:
    media_index = []

# ── Keyword filter ──
EPSTEIN_KEYWORDS = {
    "abuse","accusers","acosta","alex acosta","alan dershowitz","allegations","andrew albert christian edward","arrests","associate","attorney","attorney general","bail","black book","bill clinton","bribery","brunel","jean-luc brunel","case","case files","charges","child abuse","civil lawsuit","client list","clinton","co-conspirator","compensation program","conspiracy","court","court filings","deposition","dershowitz","documents","docket","epstein","jeffrey epstein","epstein estate","epstein victims compensation program","evidence","exhibits","fbi","federal case","federal court","financial records","flight logs","forfeiture","ghislaine","ghislaine maxwell","giuffre","virginia giuffre","grand jury","grooming","hawking","stephen hawking","high-profile associates","house oversight","immunity deal","indictment","investigation","island","little saint james","great saint james","jeffrey","judge","judicial review","kaufmann","knowledge records","lawsuit","legal filings","law enforcement","lolita express","leslie wexner","wexner","manhattan jail","metropolitan correctional center","maxwell trial","miami herald","julie k brown","minor allegations","modeling agency","non-prosecution agreement","nygard","peter nygard","new york federal court","offshore accounts","oversight hearing","palm beach","palm beach police","pilot","lawrence visoski","plea deal","prince andrew","prosecution","questioning transcript","ransome","maria farmer","annie farmer","recruitment","redacted names","restitution","rico","schwarzman","stephen schwarzman","sealed records","settlement","sex offender","sex trafficking","sexual abuse","southern district of new york","subpoena","sweetheart deal","teneo","testimony","trafficking","trial","trial transcript","underage allegations","unsealed documents","u.s. virgin islands","denise george","victim","victim statements","victim impact","virgin islands litigation","witness","wire transfer","zorro ranch","new mexico ranch","new york mansion","palm beach mansion","private jet","flight manifest","address book","contact list","affidavit","grand jury records","civil complaint","criminal complaint","federal indictment","plea agreement","non-prosecution deal","sealed exhibit","unredacted files","court transcript","evidence list","discovery files","deposition transcript","island visitors","estate litigation","receiver","trust records","financial network","associates list","socialite network","media investigation","document dump","unsealed docket","federal prosecutors","defense attorneys","witness list","pilot testimony","house managers","oversight committee","legal motion","jury selection","sentencing","appeal","custody records","detention hearing","bail hearing","survivor testimony","accuser statements","civil settlement","confidential settlement","ndas","non disclosure agreement","immunity clause","federal bureau of investigation","department of justice","sdny prosecutors","u.s. attorney","victim advocacy","support fund","recruiter allegations","assistant names","house staff","security staff","chauffeur","butler","scheduler","calendar records","visitor logs","security footage","phone records","email records","travel itinerary","passport records","offshore trust","shell companies","financial transfers","charter flights","island staff","estate manager","property records","zoning records","search warrants","evidence locker","case timeline","media coverage","investigative reporting","document archive","public records"
}

def is_epstein_related(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in EPSTEIN_KEYWORDS)

# ── Retrieval ──
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

# ── Media Matching ──
def find_related_media(query: str, answer: str = "", max_results: int = 3):

    stop_words = {
        "the","a","an","in","on","at","to","for","of","and","or","is","was","were","are",
        "what","who","how","did","do","does","with","from","by","about","that","this","it",
        "be","have","has","had","not","but","they","he","she","we","you","i","me","my",
        "his","her","named","newly","unsealed","documents","jeffrey","epstein","files",
        "legal","services","magna","case","court","document","page","all"
    }

    query_words = set(query.lower().split()) - stop_words
    if not query_words:
        return []

    scored = []
    for item in media_index:
        if "extracted_images" not in item["path"]:
            continue

        filename_words = set(item["keywords"].split()) - stop_words
        matches = query_words & filename_words

        if len(matches) >= 1:
            scored.append((len(matches), item))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [item for _, item in scored[:max_results]]

# ── Prompt Builder ──
def build_prompt(query: str, chunks: list):

    context = "\n\n---\n\n".join(
        [f"[Source: {c['source']}]\n{c['text']}" for c in chunks]
    )

    return f"""You are a research assistant with access ONLY to the Epstein court documents and House Oversight files.

STRICT RULES:
1. Answer based STRICTLY on the document excerpts below.
2. Include specific names, dates, case numbers, and legal details wherever available.
3. If the answer is not in the documents say: "This information is not found in the available Epstein files."
4. Do not speculate beyond what the documents contain.

DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{query}

ANSWER:"""

# ── Main Ask Function ──
def ask(query: str):

    if not is_epstein_related(query):
        return {
            "answer": "I can only answer questions related to the Epstein case, associated individuals, and related investigations.",
            "sources": [],
            "media": []
        }

    chunks = retrieve(query)

    if not chunks:
        return {
            "answer": "No relevant documents found for your query.",
            "sources": [],
            "media": []
        }

    prompt = build_prompt(query, chunks)

    # ✅ NEW GEMINI CALL (WORKING)
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    answer = response.text if hasattr(response, "text") else str(response)

    media = find_related_media(query, answer)

    return {
        "answer": answer,
        "sources": list(set(c["source"] for c in chunks)),
        "media": media
    }