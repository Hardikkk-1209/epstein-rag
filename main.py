from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag import ask

app = FastAPI(title="Epstein Files RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve media files
app.mount("/media", StaticFiles(directory="data"), name="media")
app.mount("/extracted", StaticFiles(directory="data/extracted_images"), name="extracted")

class Query(BaseModel):
    question: str

@app.get("/")
def serve_ui():
    return FileResponse("chat.html")

@app.post("/ask")
def ask_question(query: Query):
    result = ask(query.question)
    return result

@app.get("/health")
def health():
    return {"status": "running"}