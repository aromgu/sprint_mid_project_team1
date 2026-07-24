from __future__ import annotations

import os
import time
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import analysis, documents, state
from backend.routers import evaluation
from backend.services.rag_client import RAGClient
from backend.services.state_service import UserStateService
from src.search.service import SearchService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.local", override=True)

app = FastAPI(title="RFP Action Copilot API", version="0.1.0")
allowed_origins = [origin.strip() for origin in os.getenv("RAG_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def timing_middleware(request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
    print(f"[api-timing] {request.method} {request.url.path} {response.status_code} {elapsed_ms:.1f}ms")
    return response

app.state.search_service = SearchService()
app.state.rag_client = RAGClient(search_service=app.state.search_service)
app.state.state_service = UserStateService(PROJECT_ROOT / "backend/data/user_state.json")
app.include_router(documents.router)
app.include_router(analysis.router)
app.include_router(state.router)
app.include_router(evaluation.router)


@app.on_event("startup")
async def prewarm_models() -> None:
    """Pay model-load cost at startup so the first user request stays responsive."""
    if os.getenv("RAG_PREWARM_DENSE", "true").casefold() not in {"0", "false", "no", "off"}:
        await asyncio.to_thread(app.state.search_service.dense._load_model)
