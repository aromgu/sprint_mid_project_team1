from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelAnswer(BaseModel):
    answer: str = Field(description="Korean answer grounded only in the supplied sources")
    is_answerable: bool
    source_ids: list[str] = Field(description="Only source labels such as S1 and S2")
    caveat: str | None = None


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    requirement_ids: list[str]
    quote: str | None = None
    score: float | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    is_answerable: bool
    answer_status: Literal["answered", "partially_answered", "unanswerable"] | None = None
    caveat: str | None = None
    confidence: float | None = None
    conflicts: list[str] = []
    citations: list[Citation]
    retrieved_chunk_ids: list[str]
    retriever: str
    model: str
    search_latency_ms: float
    generation_latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
