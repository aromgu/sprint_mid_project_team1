from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SearchChunk:
    chunk_id: str
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    section_path: list[str]
    requirement_ids: list[str]
    content_type: str
    text: str
    token_count: int
    ocr_applied: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchChunk":
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchResult:
    chunk: SearchChunk
    rank: int
    score: float
    retriever: str
    component_ranks: dict[str, int] = field(default_factory=dict)
    component_scores: dict[str, float] = field(default_factory=dict)
    latency_ms: float | None = None
    context_text: str | None = None
    context_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = self.chunk.to_dict()
        payload.update(
            {
                "rank": self.rank,
                "score": self.score,
                "retriever": self.retriever,
                "component_ranks": self.component_ranks,
                "component_scores": self.component_scores,
                "latency_ms": self.latency_ms,
                "context_text": self.context_text,
                "context_chunk_ids": self.context_chunk_ids,
            }
        )
        return payload


@dataclass(slots=True)
class SearchFilters:
    document_ids: set[str] | None = None
    content_types: set[str] | None = None

    def accepts(self, chunk: SearchChunk) -> bool:
        if self.document_ids and chunk.document_id not in self.document_ids:
            return False
        if self.content_types and chunk.content_type not in self.content_types:
            return False
        return True
