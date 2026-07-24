"""외부 임베딩 전용 개인정보 마스킹 JSONL을 생성하는 CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.embeddings.redact_sensitive_text import redact_embedding_corpus


def build_parser() -> argparse.ArgumentParser:
    """입력·출력·보고서 경로를 명시적으로 받는다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> None:
    """마스킹 결과와 검증 보고서를 JSON으로 출력한다."""

    args = build_parser().parse_args()
    report = redact_embedding_corpus(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
