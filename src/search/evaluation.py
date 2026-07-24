from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from src.search.models import SearchResult


@dataclass(slots=True)
class GoldenQuery:
    question_id: str
    question: str
    reference_context_ids: list[str]
    reference_document_ids: list[str] = field(default_factory=list)
    reference_pages: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict) -> "GoldenQuery":
        return cls(
            question_id=str(payload["question_id"]),
            question=str(payload["question"]),
            reference_context_ids=list(payload.get("reference_context_ids", [])),
            reference_document_ids=list(payload.get("reference_document_ids", [])),
            reference_pages=[int(page) for page in payload.get("reference_pages", [])],
        )


def recall_at_k(results: list[SearchResult], references: set[str], k: int) -> float:
    if not references:
        return 0.0
    retrieved = {result.chunk.chunk_id for result in results[:k]}
    return len(retrieved & references) / len(references)


def reciprocal_rank(results: list[SearchResult], references: set[str], k: int) -> float:
    for rank, result in enumerate(results[:k], 1):
        if result.chunk.chunk_id in references:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[SearchResult], references: set[str], k: int) -> float:
    if not references:
        return 0.0
    relevance = [1 if result.chunk.chunk_id in references else 0 for result in results[:k]]
    dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(relevance, 1))
    ideal_count = min(len(references), k)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def page_hit(results: list[SearchResult], pages: set[int], k: int) -> float:
    if not pages:
        return 0.0
    for result in results[:k]:
        if any(result.chunk.page_start <= page <= result.chunk.page_end for page in pages):
            return 1.0
    return 0.0


def document_hit(results: list[SearchResult], documents: set[str], k: int) -> float:
    if not documents:
        return 0.0
    return float(any(result.chunk.document_id in documents for result in results[:k]))


def evaluate_queries(
    service, queries: list[GoldenQuery], retriever: str, top_k: int = 10,
    scope: str = "document",
) -> tuple[list[dict], dict]:
    if scope not in {"document", "global"}:
        raise ValueError("scope must be 'document' or 'global'")
    rows: list[dict] = []
    for item in queries:
        document_ids = set(item.reference_document_ids) if scope == "document" else None
        results = service.search(
            item.question, retriever=retriever, top_k=top_k, document_ids=document_ids or None,
        )
        references = set(item.reference_context_ids)
        row = {
            "question_id": item.question_id,
            "question": item.question,
            "retriever": retriever,
            "recall@1": recall_at_k(results, references, 1),
            "recall@3": recall_at_k(results, references, 3),
            "recall@5": recall_at_k(results, references, 5),
            "recall@10": recall_at_k(results, references, 10),
            "mrr@10": reciprocal_rank(results, references, 10),
            "ndcg@10": ndcg_at_k(results, references, 10),
            "document_hit@5": document_hit(results, set(item.reference_document_ids), 5),
            "page_hit@5": page_hit(results, set(item.reference_pages), 5),
            "latency_ms": results[0].latency_ms if results else 0.0,
            "retrieved_context_ids": [result.chunk.chunk_id for result in results],
        }
        rows.append(row)
    metric_names = [
        "recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10",
        "document_hit@5", "page_hit@5", "latency_ms",
    ]
    summary = {
        "retriever": retriever,
        "scope": scope,
        "query_count": len(rows),
        **{
            metric: statistics.fmean(row[metric] for row in rows) if rows else 0.0
            for metric in metric_names
        },
    }
    return rows, summary
