import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from rag import ask

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Epstein Files RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=os.path.join(BASE_DIR, "data")), name="media")

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "chat.html"))