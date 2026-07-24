from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.search.models import SearchChunk


def file_fingerprint(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_chunks(path: Path) -> list[SearchChunk]:
    chunks: list[SearchChunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                chunks.append(SearchChunk.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid chunk at {path}:{line_number}: {exc}") from exc
    if not chunks:
        raise ValueError(f"No chunks found in {path}")
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk IDs must be unique")
    return chunks

