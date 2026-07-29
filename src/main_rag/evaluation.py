"""Adapters between Main Advanced retrieval dictionaries and Golden v3 evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluation.golden_v3 import requirement_ids


@dataclass(slots=True)
class EvaluationChunk:
    chunk_id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    requirement_ids: tuple[str, ...]


@dataclass(slots=True)
class EvaluationResult:
    chunk: EvaluationChunk
    score: float
    rank: int
    latency_ms: float | None
    context_text: str


def adapt_retrieval_results(
    rows: list[dict[str, Any]], *, latency_ms: float | None = None
) -> list[EvaluationResult]:
    adapted = []
    for rank, row in enumerate(rows, 1):
        metadata = row.get("metadata") or {}
        text = str(row.get("text") or "")
        page_start = int(
            row.get("page") or metadata.get("page_start") or metadata.get("page") or 0
        )
        page_end = int(metadata.get("page_end") or page_start)
        raw_section = metadata.get("section_path")
        section_path = (
            tuple(str(item) for item in raw_section)
            if isinstance(raw_section, list)
            else ((str(raw_section),) if raw_section else ())
        )
        adapted.append(
            EvaluationResult(
                chunk=EvaluationChunk(
                    chunk_id=str(row.get("chunk_id") or row.get("id")),
                    document_id=str(metadata.get("document_id") or ""),
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    section_path=section_path,
                    requirement_ids=tuple(requirement_ids(text)),
                ),
                score=float(row.get("score") or 0.0),
                rank=rank,
                latency_ms=latency_ms,
                context_text=text,
            )
        )
    return adapted
