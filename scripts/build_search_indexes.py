from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.search.service import PROJECT_ROOT, SearchService


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 and dense indexes for the RFP chunks.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "search.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    service = SearchService(args.config)
    service.build_indexes(force=args.force, include_dense=not args.bm25_only)
    print(
        f"Indexed {len(service.chunks)} chunks in {time.perf_counter() - start:.2f}s "
        f"(dense={'off' if args.bm25_only else 'on'})."
    )


if __name__ == "__main__":
    main()

