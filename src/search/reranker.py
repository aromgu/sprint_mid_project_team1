from __future__ import annotations

import time

import numpy as np

from src.search.models import SearchFilters, SearchResult


class CrossEncoderReranker:
    def __init__(
        self,
        base_engine,
        model_name: str,
        candidate_k: int = 12,
        batch_size: int = 2,
        max_length: int = 512,
        device: str = "cpu",
    ) -> None:
        self.base_engine = base_engine
        self.model_name = model_name
        self.candidate_k = candidate_k
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.model = None

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
                local_files_only=True,
            )
        return self.model

    @staticmethod
    def order(scores: list[float]) -> list[int]:
        return sorted(range(len(scores)), key=lambda index: (-scores[index], index))

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        start = time.perf_counter()
        candidates = self.base_engine.search(query, self.candidate_k, filters)
        if not candidates:
            return []
        pairs = [(query, result.chunk.text) for result in candidates]
        raw_scores = self._load_model().predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores).reshape(-1).astype(float).tolist()
        ordered = self.order(scores)
        latency_ms = (time.perf_counter() - start) * 1000
        results = []
        for rank, candidate_index in enumerate(ordered[:top_k], 1):
            candidate = candidates[candidate_index]
            results.append(
                SearchResult(
                    chunk=candidate.chunk,
                    rank=rank,
                    score=scores[candidate_index],
                    retriever="cross_encoder_reranked",
                    component_ranks={**candidate.component_ranks, "reranker": rank},
                    component_scores={
                        **candidate.component_scores,
                        "rrf": candidate.score,
                        "reranker": scores[candidate_index],
                    },
                    latency_ms=latency_ms,
                )
            )
        return results

