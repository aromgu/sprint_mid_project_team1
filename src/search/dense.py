from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from src.search.loader import file_fingerprint
from src.search.models import SearchChunk, SearchFilters, SearchResult


class DenseSearchEngine:
    def __init__(
        self,
        chunks: list[SearchChunk],
        chunks_path: Path,
        index_dir: Path,
        model_name: str,
        batch_size: int = 8,
        device: str = "cpu",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ) -> None:
        self.chunks = chunks
        self.chunks_path = chunks_path
        self.index_dir = index_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.model = None
        self.embeddings: np.ndarray | None = None

    @property
    def cache_key(self) -> str:
        return self.model_name.replace("/", "--")

    @property
    def embeddings_path(self) -> Path:
        return self.index_dir / f"dense_{self.cache_key}.npy"

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / f"dense_{self.cache_key}.json"

    def _metadata(self) -> dict:
        return {
            "chunks_sha256": file_fingerprint(self.chunks_path),
            "chunk_count": len(self.chunks),
            "model": self.model_name,
            "passage_prefix": self.passage_prefix,
            "normalized": True,
        }

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=os.getenv("RAG_MODEL_LOCAL_ONLY", "true").lower()
                not in {"0", "false", "no"},
            )
        return self.model

    def is_cache_valid(self) -> bool:
        if not self.embeddings_path.exists() or not self.metadata_path.exists():
            return False
        try:
            saved = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            expected = self._metadata()
            return all(saved.get(key) == value for key, value in expected.items())
        except Exception:
            return False

    def build(self, force: bool = False, show_progress: bool = True) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if not force and self.is_cache_valid():
            self.embeddings = np.load(self.embeddings_path, mmap_mode="r")
            return
        model = self._load_model()
        passages = [self.passage_prefix + chunk.text for chunk in self.chunks]
        embeddings = model.encode(
            passages,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(self.embeddings_path, embeddings)
        self.metadata_path.write_text(
            json.dumps({**self._metadata(), "dimension": int(embeddings.shape[1])}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.embeddings = np.load(self.embeddings_path, mmap_mode="r")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if self.embeddings is None:
            self.build(show_progress=False)
        start = time.perf_counter()
        query_vector = self._load_model().encode(
            [self.query_prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        scores = np.asarray(self.embeddings @ query_vector)
        candidates = [
            (index, float(score))
            for index, score in enumerate(scores)
            if not filters or filters.accepts(self.chunks[index])
        ]
        candidates.sort(key=lambda item: (-item[1], self.chunks[item[0]].chunk_id))
        latency_ms = (time.perf_counter() - start) * 1000
        return [
            SearchResult(
                chunk=self.chunks[index],
                rank=rank,
                score=score,
                retriever="dense",
                component_ranks={"dense": rank},
                component_scores={"dense": score},
                latency_ms=latency_ms,
            )
            for rank, (index, score) in enumerate(candidates[:top_k], 1)
        ]
