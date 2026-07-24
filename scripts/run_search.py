from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.search.service import PROJECT_ROOT, SearchService


def print_results(query: str, results) -> None:
    print(f"\nQuery: {query}")
    for result in results:
        chunk = result.chunk
        pages = str(chunk.page_start) if chunk.page_start == chunk.page_end else f"{chunk.page_start}-{chunk.page_end}"
        requirements = ", ".join(chunk.requirement_ids) or "-"
        component = ", ".join(f"{name}#{rank}" for name, rank in result.component_ranks.items())
        preview = " ".join(chunk.text.split())[:280]
        print(
            f"{result.rank}. [{chunk.document_id}] p.{pages} {requirements} "
            f"score={result.score:.6f} ({component})\n   {preview}"
        )
    if results:
        print(f"Latency: {results[0].latency_ms:.2f} ms")


def read_queries(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if line.strip():
                payload = json.loads(line)
                records.append(
                    {
                        "question_id": str(payload.get("question_id", index)),
                        "question": payload.get("question") or payload.get("query"),
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the nine RFP documents.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query")
    source.add_argument("--queries", type=Path, help="JSONL with question or query fields")
    parser.add_argument(
        "--retriever", choices=("bm25", "dense", "hybrid", "reranked"),
        help="Override pipeline.retriever from the config"
    )
    parser.add_argument("--top-k", type=int, help="Override pipeline.top_k from the config")
    parser.add_argument("--document-id", action="append", dest="document_ids")
    parser.add_argument("--content-type", action="append", dest="content_types")
    parser.add_argument("--neighbor-window", type=int, help="Override context_expansion from config")
    parser.add_argument("--output", type=Path, help="Optional JSONL output")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "search.yaml")
    args = parser.parse_args()

    service = SearchService(args.config)
    queries = (
        [{"question_id": "interactive", "question": args.query}]
        if args.query
        else read_queries(args.queries)
    )
    output_records = []
    for item in queries:
        results = service.search(
            item["question"],
            retriever=args.retriever,
            top_k=args.top_k,
            document_ids=set(args.document_ids) if args.document_ids else None,
            content_types=set(args.content_types) if args.content_types else None,
            neighbor_window=args.neighbor_window,
        )
        print_results(item["question"], results)
        output_records.append(
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "retriever": args.retriever or service.default_retriever,
                "results": [result.to_dict() for result in results],
            }
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for record in output_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(output_records)} queries to {args.output}")


if __name__ == "__main__":
    main()
