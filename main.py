import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from rag import ask
import os

DATA_DIR = os.path.join(BASE_DIR, "data")

# create folder if missing
os.makedirs(DATA_DIR, exist_ok=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Epstein Files RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
MEDIA_DIR = os.path.join(BASE_DIR, "data")

# create folder automatically (required for Hugging Face)
os.makedirs(MEDIA_DIR, exist_ok=True)

app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "chat.html"))