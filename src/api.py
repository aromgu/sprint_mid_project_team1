from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.generation.models import AnswerResponse
from src.generation.openai_generator import OpenAIRAGService
from src.search.service import PROJECT_ROOT, SearchService


app = FastAPI(title="RFP RAG API", version="0.1.0")


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    retriever: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[str] | None = None
    content_types: list[str] | None = None
    neighbor_window: int | None = Field(default=None, ge=0, le=5)


@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    return SearchService()


@lru_cache(maxsize=1)
def get_rag_service() -> OpenAIRAGService:
    return OpenAIRAGService(search_service=get_search_service())


@app.get("/health")
def health() -> dict:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env.local", override=True)
    service = get_search_service()
    return {
        "status": "ok", "chunk_count": len(service.chunks),
        "default_retriever": service.default_retriever,
        "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/search")
def search(request: SearchRequest) -> dict:
    try:
        results = get_search_service().search(
            request.question, request.retriever, request.top_k,
            set(request.document_ids) if request.document_ids else None,
            set(request.content_types) if request.content_types else None,
            request.neighbor_window,
        )
        return {"question": request.question, "results": [result.to_dict() for result in results]}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/answer", response_model=AnswerResponse)
def answer(request: SearchRequest) -> AnswerResponse:
    try:
        return get_rag_service().answer(
            request.question, request.retriever, request.top_k,
            set(request.document_ids) if request.document_ids else None,
            set(request.content_types) if request.content_types else None,
            request.neighbor_window,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {type(exc).__name__}") from exc
