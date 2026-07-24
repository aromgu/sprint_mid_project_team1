from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.generation.openai_generator import OpenAIRAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question from the RFP documents.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/generation.yaml"))
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid", "reranked"))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--document-id", action="append", dest="document_ids")
    parser.add_argument("--neighbor-window", type=int)
    args = parser.parse_args()
    result = OpenAIRAGService(args.config).answer(
        args.query, retriever=args.retriever, top_k=args.top_k,
        document_ids=set(args.document_ids) if args.document_ids else None,
        neighbor_window=args.neighbor_window,
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
