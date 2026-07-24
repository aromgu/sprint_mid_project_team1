from __future__ import annotations

import json
import pickle
import time
import hashlib
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.search.loader import file_fingerprint
from src.search.models import SearchChunk, SearchFilters, SearchResult
from src.search.tokenization import build_tokenizer


class BM25SearchEngine:
    def __init__(
        self,
        chunks: list[SearchChunk],
        chunks_path: Path,
        index_dir: Path,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer_config: dict | None = None,
    ) -> None:
        self.chunks = chunks
        self.chunks_path = chunks_path
        self.index_dir = index_dir
        self.k1 = k1
        self.b = b
        self.tokenizer_config = tokenizer_config or {"type": "korean_ngram", "ngram_sizes": [2, 3]}
        self.tokenizer = build_tokenizer(self.tokenizer_config)
        self.index: BM25Okapi | None = None
        self.tokenized_corpus: list[list[str]] | None = None

    @property
    def cache_path(self) -> Path:
        return self.index_dir / f"bm25_{self.cache_key}.pkl"

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / f"bm25_{self.cache_key}.json"

    @property
    def cache_key(self) -> str:
        relevant = {"tokenizer": self.tokenizer_config, "k1": self.k1, "b": self.b}
        payload = json.dumps(relevant, sort_keys=True, ensure_ascii=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def _metadata(self) -> dict:
        return {
            "chunks_sha256": file_fingerprint(self.chunks_path),
            "chunk_count": len(self.chunks),
            "tokenizer_version": self.tokenizer.version,
            "tokenizer": self.tokenizer_config,
            "k1": self.k1,
            "b": self.b,
        }

    def is_cache_valid(self) -> bool:
        if not self.cache_path.exists() or not self.metadata_path.exists():
            return False
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8")) == self._metadata()
        except Exception:
            return False

    def build(self, force: bool = False) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if not force and self.is_cache_valid():
            with self.cache_path.open("rb") as handle:
                self.tokenized_corpus = pickle.load(handle)
        else:
            self.tokenized_corpus = [self.tokenizer(chunk.text) for chunk in self.chunks]
            with self.cache_path.open("wb") as handle:
                pickle.dump(self.tokenized_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.metadata_path.write_text(
                json.dumps(self._metadata(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        self.index = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if self.index is None:
            self.build()
        start = time.perf_counter()
        scores = self.index.get_scores(self.tokenizer(query))
        candidates = [
            (index, float(score))
            for index, score in enumerate(scores)
            if score > 0
            if not filters or filters.accepts(self.chunks[index])
        ]
        candidates.sort(key=lambda item: (-item[1], self.chunks[item[0]].chunk_id))
        latency_ms = (time.perf_counter() - start) * 1000
        return [
            SearchResult(
                chunk=self.chunks[index],
                rank=rank,
                score=score,
                retriever="bm25",
                component_ranks={"bm25": rank},
                component_scores={"bm25": score},
                latency_ms=latency_ms,
            )
            for rank, (index, score) in enumerate(candidates[:top_k], 1)
        ]
