"""Run one question through the isolated Main Advanced RAG pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main_rag.service import MainAdvancedRAGService
from src.main_rag.settings import DEFAULT_CONFIG_PATH, load_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--document-id")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()
    service = MainAdvancedRAGService(settings=load_settings(args.config))
    result = asyncio.run(
        service.answer(
            args.question,
            document_id=args.document_id,
            top_k=args.top_k,
        )
    )
    evidence = result.get("evidence") or []
    payload = {
        "answer": result.get("answer"),
        "evidence": evidence,
        "page": [item.get("page") for item in evidence],
        "chunk_id": [item.get("chunk_id") for item in evidence],
        "confidence": result.get("confidence"),
        "latency": result.get("latency"),
        "rewritten_query": result.get("rewritten_query"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
