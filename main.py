import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from rag import ask

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, "data")

# create folder automatically (required for Hugging Face)
os.makedirs(MEDIA_DIR, exist_ok=True)

app = FastAPI(title="Epstein Files RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

class AskRequest(BaseModel):
    query: str
    mode: Optional[str] = "strict"
    filters: Optional[List[str]] = None

@app.post("/ask")
def handle_ask(req: AskRequest):
    return ask(req.query, mode=req.mode, filters=req.filters)

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "chat.html"))
@app.get("/models")
def list_models():
    from rag import get_gemini_client
    models = [m.name for m in get_gemini_client().models.list()]
    return {"models": models}