from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.search.bm25 import BM25SearchEngine
from src.search.dense import DenseSearchEngine
from src.search.hybrid import build_fusion_engine
from src.search.loader import load_chunks
from src.search.models import SearchFilters, SearchResult
from src.search.reranker import CrossEncoderReranker


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_search_config(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = _seen or set()
    if path in seen:
        raise ValueError(f"Circular search config inheritance: {path}")
    seen.add(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = config.pop("extends", None)
    if not parent:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(load_search_config(parent_path, seen), config)


class SearchService:
    def __init__(self, config_path: Path | None = None) -> None:
        config_path = config_path or PROJECT_ROOT / "configs" / "search.yaml"
        self.config = load_search_config(config_path)
        chunks_path = resolve_path(self.config["paths"]["chunks"])
        self.index_dir = resolve_path(self.config["paths"]["index_dir"])
        self.chunks_path = chunks_path
        self.chunks = load_chunks(chunks_path)
        self._engines: dict[str, Any] = {}
        self._document_sequences: dict[str, list] = {}
        self._positions: dict[str, tuple[str, int]] = {}
        for document_id in sorted({chunk.document_id for chunk in self.chunks}):
            sequence = sorted(
                (chunk for chunk in self.chunks if chunk.document_id == document_id),
                key=lambda chunk: (chunk.page_start, chunk.page_end, chunk.chunk_id),
            )
            self._document_sequences[document_id] = sequence
            for position, chunk in enumerate(sequence):
                self._positions[chunk.chunk_id] = (document_id, position)

    @property
    def default_retriever(self) -> str:
        return self.config.get("pipeline", {}).get("retriever", "hybrid")

    @property
    def available_retrievers(self) -> tuple[str, ...]:
        names = ["bm25", "dense", "hybrid"]
        if self.config.get("reranker", {}).get("enabled", False):
            names.append("reranked")
        return tuple(names)

    def get_engine(self, name: str):
        if name in self._engines:
            return self._engines[name]
        if name == "bm25":
            config = self.config["bm25"]
            engine = BM25SearchEngine(
                self.chunks, self.chunks_path, self.index_dir,
                k1=config["k1"], b=config["b"], tokenizer_config=config.get("tokenizer"),
            )
        elif name == "dense":
            config = self.config["dense"]
            engine = DenseSearchEngine(
                self.chunks, self.chunks_path, self.index_dir,
                model_name=config["model"], batch_size=config["batch_size"],
                device=config["device"], query_prefix=config.get("query_prefix", ""),
                passage_prefix=config.get("passage_prefix", ""),
            )
        elif name == "hybrid":
            engine = build_fusion_engine(
                self.config["fusion"], self.get_engine("bm25"), self.get_engine("dense")
            )
        elif name == "reranked":
            config = self.config["reranker"]
            if not config.get("enabled", False):
                raise ValueError("Retriever 'reranked' is disabled by config: reranker.enabled=false")
            if config.get("type", "cross_encoder") != "cross_encoder":
                raise ValueError(
                    f"Unknown reranker type {config.get('type')!r}; choose cross_encoder or disable it"
                )
            engine = CrossEncoderReranker(
                self.get_engine(config.get("base_retriever", "hybrid")),
                model_name=config["model"], candidate_k=config["candidate_k"],
                batch_size=config["batch_size"], max_length=config["max_length"],
                device=config["device"],
            )
        else:
            raise ValueError(f"Unknown retriever {name!r}; choose from {self.available_retrievers}")
        self._engines[name] = engine
        return engine

    @property
    def bm25(self):
        return self.get_engine("bm25")

    @property
    def dense(self):
        return self.get_engine("dense")

    @property
    def hybrid(self):
        return self.get_engine("hybrid")

    @property
    def reranked(self):
        return self.get_engine("reranked")

    def build_indexes(self, force: bool = False, include_dense: bool = True) -> None:
        self.get_engine("bm25").build(force=force)
        if include_dense:
            self.get_engine("dense").build(force=force)

    def search(
        self, query: str, retriever: str | None = None, top_k: int | None = None,
        document_ids: set[str] | None = None, content_types: set[str] | None = None,
        neighbor_window: int | None = None,
    ) -> list[SearchResult]:
        pipeline = self.config.get("pipeline", {})
        retriever = retriever or self.default_retriever
        top_k = top_k if top_k is not None else int(pipeline.get("top_k", 5))
        context = self.config.get("context_expansion", {})
        if neighbor_window is None:
            neighbor_window = int(context.get("window", 0)) if context.get("enabled", False) else 0
        filters = SearchFilters(document_ids=document_ids, content_types=content_types)
        results = self.get_engine(retriever).search(query, top_k=top_k, filters=filters)
        if neighbor_window > 0:
            self.expand_context(results, neighbor_window)
        return results

    def expand_context(self, results: list[SearchResult], window: int = 1) -> None:
        if window < 0:
            raise ValueError("Neighbor window must be non-negative")
        for result in results:
            document_id, position = self._positions[result.chunk.chunk_id]
            sequence = self._document_sequences[document_id]
            start = max(0, position - window)
            end = min(len(sequence), position + window + 1)
            context_chunks = sequence[start:end]
            result.context_chunk_ids = [chunk.chunk_id for chunk in context_chunks]
            result.context_text = "\n\n".join(chunk.text for chunk in context_chunks)
