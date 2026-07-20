"""GCP에서 Naive 임베딩·Chroma 저장을 실행하는 CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.embeddings.build_embeddings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INPUT_PATH,
    DEFAULT_PERSIST_DIRECTORY,
    DEFAULT_REPORT_PATH,
    audit_input,
    build_embeddings,
)


def build_parser() -> argparse.ArgumentParser:
    """팀원이 경로·모델·배치 크기를 명확히 확인할 수 있는 CLI를 만든다."""

    parser = argparse.ArgumentParser(
        description="Naive 청크를 OpenAI로 임베딩하고 Chroma에 저장합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--persist-directory", type=Path, default=DEFAULT_PERSIST_DIRECTORY
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="소량 smoke test용 최대 청크 수. 전체 실행에서는 지정하지 않습니다.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="API를 호출하지 않고 입력 파일만 검사합니다.",
    )
    return parser


def main() -> None:
    """입력 검증 또는 전체 임베딩 실행 후 결과를 JSON으로 출력한다."""

    args = build_parser().parse_args()
    if args.validate_only:
        audit = audit_input(args.input, max_records=args.max_records)
        # chunk_ids 전체는 화면에 출력하지 않고 건수만 보고한다.
        payload = asdict(audit)
        payload["chunk_ids"] = len(audit.chunk_ids)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    report = build_embeddings(
        input_path=args.input,
        persist_directory=args.persist_directory,
        report_path=args.report,
        collection_name=args.collection_name,
        embedding_model=args.model,
        batch_size=args.batch_size,
        max_records=args.max_records,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
