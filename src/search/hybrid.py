from __future__ import annotations

import time

from src.search.models import SearchFilters, SearchResult


class RRFHybridSearchEngine:
    def __init__(
        self,
        bm25_engine,
        dense_engine,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        candidate_k: int = 20,
    ) -> None:
        self.bm25_engine = bm25_engine
        self.dense_engine = dense_engine
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        start = time.perf_counter()
        bm25_results = self.bm25_engine.search(query, self.candidate_k, filters)
        dense_results = self.dense_engine.search(query, self.candidate_k, filters)
        combined: dict[str, dict] = {}
        for name, weight, results in (
            ("bm25", self.bm25_weight, bm25_results),
            ("dense", self.dense_weight, dense_results),
        ):
            for result in results:
                item = combined.setdefault(
                    result.chunk.chunk_id,
                    {"chunk": result.chunk, "score": 0.0, "ranks": {}, "scores": {}},
                )
                item["score"] += weight / (self.rrf_k + result.rank)
                item["ranks"][name] = result.rank
                item["scores"][name] = result.score
        ordered = sorted(combined.values(), key=lambda item: (-item["score"], item["chunk"].chunk_id))
        latency_ms = (time.perf_counter() - start) * 1000
        return [
            SearchResult(
                chunk=item["chunk"],
                rank=rank,
                score=item["score"],
                retriever="hybrid_rrf",
                component_ranks=item["ranks"],
                component_scores=item["scores"],
                latency_ms=latency_ms,
            )
            for rank, item in enumerate(ordered[:top_k], 1)
        ]


class WeightedScoreHybridSearchEngine:
    """Fuse BM25 and dense candidates after per-query min-max normalization."""

    def __init__(self, bm25_engine, dense_engine, bm25_weight=1.0, dense_weight=1.0, candidate_k=20):
        self.bm25_engine = bm25_engine
        self.dense_engine = dense_engine
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.candidate_k = candidate_k

    @staticmethod
    def _normalized(results: list[SearchResult]) -> dict[str, float]:
        if not results:
            return {}
        values = [result.score for result in results]
        low, high = min(values), max(values)
        if high == low:
            return {result.chunk.chunk_id: 1.0 for result in results}
        return {result.chunk.chunk_id: (result.score - low) / (high - low) for result in results}

    def search(self, query: str, top_k: int = 10, filters: SearchFilters | None = None) -> list[SearchResult]:
        start = time.perf_counter()
        components = {
            "bm25": self.bm25_engine.search(query, self.candidate_k, filters),
            "dense": self.dense_engine.search(query, self.candidate_k, filters),
        }
        normalized = {name: self._normalized(results) for name, results in components.items()}
        weights = {"bm25": self.bm25_weight, "dense": self.dense_weight}
        combined: dict[str, dict] = {}
        for name, results in components.items():
            for result in results:
                item = combined.setdefault(result.chunk.chunk_id, {"chunk": result.chunk, "score": 0.0, "ranks": {}, "scores": {}})
                item["score"] += weights[name] * normalized[name][result.chunk.chunk_id]
                item["ranks"][name] = result.rank
                item["scores"][name] = result.score
        ordered = sorted(combined.values(), key=lambda item: (-item["score"], item["chunk"].chunk_id))
        latency_ms = (time.perf_counter() - start) * 1000
        return [SearchResult(item["chunk"], rank, item["score"], "hybrid_weighted_score", item["ranks"], item["scores"], latency_ms) for rank, item in enumerate(ordered[:top_k], 1)]


def build_fusion_engine(config: dict, bm25_engine, dense_engine):
    fusion_type = config.get("type", "rrf")
    common = {
        "bm25_weight": config.get("bm25_weight", 1.0),
        "dense_weight": config.get("dense_weight", 1.0),
        "candidate_k": config.get("candidate_k", 20),
    }
    if fusion_type == "rrf":
        return RRFHybridSearchEngine(bm25_engine, dense_engine, rrf_k=config.get("rrf_k", 60), **common)
    if fusion_type == "weighted_score":
        return WeightedScoreHybridSearchEngine(bm25_engine, dense_engine, **common)
    raise ValueError(f"Unknown fusion type {fusion_type!r}; choose rrf or weighted_score")
