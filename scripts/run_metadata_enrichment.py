"""검증된 CSV 사업 메타데이터를 최종 청크에 보강하는 CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.chunking.enrich_metadata import enrich_chunk_metadata


def build_parser() -> argparse.ArgumentParser:
    """팀원이 모든 입력 계보를 명시하도록 CLI 인자를 정의한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--metadata-documents", type=Path, required=True)
    parser.add_argument("--corrections-csv", type=Path)
    parser.add_argument("--summary-review-csv", type=Path, required=True)
    parser.add_argument("--duplicate-aliases-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> None:
    """메타데이터 보강을 실행하고 동일한 보고서를 표준 출력에도 표시한다."""

    args = build_parser().parse_args()
    report = enrich_chunk_metadata(
        chunks_path=args.chunks,
        metadata_path=args.metadata_documents,
        corrections_path=args.corrections_csv,
        summary_review_path=args.summary_review_csv,
        duplicate_alias_path=args.duplicate_aliases_csv,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
