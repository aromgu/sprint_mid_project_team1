from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.search.evaluation import GoldenQuery, evaluate_queries
from src.search.service import PROJECT_ROOT, SearchService


def load_golden(path: Path) -> list[GoldenQuery]:
    with path.open("r", encoding="utf-8") as handle:
        return [GoldenQuery.from_dict(json.loads(line)) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against a Golden set JSONL.")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument(
        "--retriever", choices=("bm25", "dense", "hybrid", "reranked", "all"), default="all"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--scope", choices=("document", "global"), default="document",
        help="Restrict each query to its reference documents (default) or search the full corpus.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "retrieval")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "search.yaml")
    args = parser.parse_args()
    queries = load_golden(args.golden)
    service = SearchService(args.config)
    retrievers = service.available_retrievers if args.retriever == "all" else (args.retriever,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for retriever in retrievers:
        rows, summary = evaluate_queries(service, queries, retriever, args.top_k, args.scope)
        summaries.append(summary)
        path = args.output_dir / f"{retriever}_details.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
