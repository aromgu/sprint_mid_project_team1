"""GCP에서 Advanced Dense·BM25 인덱싱을 실행하는 CLI."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from src.embeddings.build_advanced_index import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BM25_DIRECTORY,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INPUT_PATH,
    DEFAULT_PERSIST_DIRECTORY,
    DEFAULT_REPORT_PATH,
    audit_advanced_input,
    build_advanced_indexes,
)


def build_parser() -> argparse.ArgumentParser:
    """경로·모델·인덱스 유형을 명시적으로 선택하는 CLI를 만든다."""

    parser = argparse.ArgumentParser(
        description=("Advanced v2 청크를 OpenAI Dense Chroma와 Kiwi BM25로 저장합니다.")
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=DEFAULT_PERSIST_DIRECTORY,
    )
    parser.add_argument(
        "--bm25-directory",
        type=Path,
        default=DEFAULT_BM25_DIRECTORY,
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--mode",
        choices=("all", "dense", "bm25"),
        default="all",
        help="all은 Dense와 BM25 모두, dense/bm25는 해당 인덱스만 만듭니다.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="소량 smoke test용 최대 입력 청크 수입니다.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="API·DB 저장 없이 승인 SHA와 Advanced 입력 계약만 검사합니다.",
    )
    return parser


def grant_group_access(*paths: Path) -> None:
    """산출물을 teamshare 그룹이 읽고 쓸 수 있게 맞춘다.

    umask만으로는 부족하다. sqlite가 DB 파일을 0644로 만들기 때문에 umask는 비트를
    뺄 수만 있고 그룹 쓰기를 더하지 못한다. 그 결과 chroma.sqlite3이 그룹 읽기
    전용으로 남아, 다른 팀원이 컬렉션을 열 때 Chroma가 조회에도 쓰기 모드로 열기
    때문에 "attempt to write a readonly database"로 실패한다.
    """
    for path in paths:
        if not path.exists():
            continue
        targets = [path, *path.rglob("*")] if path.is_dir() else [path]
        for target in targets:
            try:
                # 디렉터리는 그룹 진입·탐색과 setgid까지, 파일은 읽기·쓰기만 준다.
                extra = 0o2070 if target.is_dir() else 0o060
                target.chmod(target.stat().st_mode | extra)
            except PermissionError:
                # 다른 사람이 만든 파일은 건드리지 않는다.
                continue


def main() -> None:
    """입력 검증 또는 Advanced 인덱싱 실행 결과를 JSON으로 출력한다."""

    args = build_parser().parse_args()
    # 산출물을 teamshare 그룹이 쓸 수 있게 만든다. 청킹 스크립트와 같은 값이다.
    # 다만 umask는 비트를 뺄 수만 있어 sqlite가 0644로 만드는 chroma.sqlite3에는
    # 그룹 쓰기를 더하지 못한다. 그건 grant_group_access가 마무리한다.
    os.umask(0o007)
    if args.validate_only:
        audit = audit_advanced_input(args.input, max_records=args.max_records)
        payload = asdict(audit)
        payload["chunk_ids"] = len(audit.chunk_ids)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    report = build_advanced_indexes(
        input_path=args.input,
        persist_directory=args.persist_directory,
        bm25_directory=args.bm25_directory,
        report_path=args.report,
        collection_name=args.collection_name,
        embedding_model=args.model,
        batch_size=args.batch_size,
        max_records=args.max_records,
        mode=args.mode,
    )
    grant_group_access(
        args.persist_directory,
        args.bm25_directory,
        args.report,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
