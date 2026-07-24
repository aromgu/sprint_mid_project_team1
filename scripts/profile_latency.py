"""Profile local retrieval stages; add --live to measure one OpenAI answer call."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.search.service import SearchService


MEASUREMENTS: dict[str, float] = {}


def measure(label: str, fn):
    started = time.perf_counter()
    value = fn()
    elapsed = (time.perf_counter() - started) * 1000
    MEASUREMENTS[label] = round(elapsed, 2)
    print(f"{label:24s} {elapsed:9.1f} ms")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--query", default="SFR-007 예약 시스템 기능")
    parser.add_argument("--live", action="store_true", help="Also call OpenAI once; incurs API cost")
    parser.add_argument("--output", type=Path, default=Path("reports/latency/profile_latency.json"))
    args = parser.parse_args()

    service = measure("service init", SearchService)
    filters = {args.document_id} if args.document_id else None
    from src.search.models import SearchFilters
    search_filters = SearchFilters(document_ids=filters)
    measure("bm25 cold", lambda: service.bm25.search(args.query, 5, search_filters))
    measure("dense cold", lambda: service.dense.search(args.query, 5, search_filters))
    measure("hybrid warm", lambda: service.hybrid.search(args.query, 5, search_filters))
    measure("hybrid repeat", lambda: service.hybrid.search(args.query, 5, search_filters))
    if args.live:
        from src.generation.openai_generator import OpenAIRAGService
        generator = measure("generator init", lambda: OpenAIRAGService(search_service=service))
        result = measure("openai answer", lambda: generator.answer(args.query, document_ids=filters))
        print(f"answer model={result.model} input_tokens={result.input_tokens} output_tokens={result.output_tokens}")
        MEASUREMENTS["input_tokens"] = result.input_tokens
        MEASUREMENTS["output_tokens"] = result.output_tokens
        MEASUREMENTS["estimated_cost_usd"] = result.estimated_cost_usd
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"query": args.query, "document_id": args.document_id, "live": args.live, "measurements_ms": MEASUREMENTS}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved profile: {args.output}")


if __name__ == "__main__":
    main()
