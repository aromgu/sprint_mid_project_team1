"""GCP에서 구조 전처리 JSONL을 Naive RCTS 청크로 저장하는 CLI.

기존 청크와 Chroma 컬렉션은 수정하지 않는다. 입력 문서·블록을 모두 읽어
품질 게이트를 통과한 경우에만 결정적인 gzip JSONL과 실행 보고서를 새
경로에 기록한다.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.chunking.split_text import (
    INDEXABLE_POLICIES,
    ChunkConfig,
    TiktokenCodec,
    build_chunk_corpus,
    build_summary,
    fallback_markdown_table,
    validate_chunk_corpus,
)

DEFAULT_DOCUMENTS_PATH = Path("/home/data/advanced/input_v1/documents_v2.jsonl")
DEFAULT_BLOCKS_PATH = Path("/home/data/advanced/input_v1/blocks_v2.jsonl")
DEFAULT_OUTPUT_PATH = Path("/home/data/advanced/chunks/chunks_naive_rcts_v2.jsonl.gz")
DEFAULT_REPORT_PATH = Path(
    "/home/data/advanced/reports/chunks_naive_rcts_v2_report.json"
)
INLINE_BREAK_TAG = re.compile(r"<br\s*/?>", re.IGNORECASE)
STRUCTURAL_HTML_TAG = re.compile(
    r"</?(?:table|caption|tr|th|td|img|p|li)\b[^>]*>",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    """큰 입력과 출력도 메모리에 올리지 않고 SHA-256을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL을 읽고 오류가 난 실제 줄 번호를 포함해 실패시킨다."""
    rows: list[dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL 파싱 실패: {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"JSON 객체가 아닌 행입니다: {path}:{line_number}")
            rows.append(value)
    return rows


def adapt_legacy_html_tables(
    blocks: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """기존 v2의 HTML 표를 검색용 평문 기반 GFM으로 안전하게 변환한다.

    ``blocks_v2``는 복합 표 복원용 ``display_content``에 HTML을 보관하고
    검색용 ``retrieval_text``를 별도로 가진다. Naive 임베딩에는 HTML을
    넣지 않으므로 검색 대상 표만 복사해 한 셀 GFM 행으로 바꾼다. 원본
    레코드와 입력 파일은 변경하지 않는다.
    """
    adapted_blocks: list[dict[str, Any]] = []
    converted_count = 0
    sanitized_break_count = 0
    for block in blocks:
        display = str(block.get("display_content") or "")
        is_indexable_table = (
            block.get("block_type") == "table"
            and block.get("index_policy") in INDEXABLE_POLICIES
        )
        if not is_indexable_table:
            adapted_blocks.append(block)
            continue

        display_without_breaks = INLINE_BREAK_TAG.sub(" / ", display)
        has_structural_html = bool(STRUCTURAL_HTML_TAG.search(display_without_breaks))
        if not has_structural_html and display_without_breaks == display:
            adapted_blocks.append(block)
            continue

        retrieval_text = str(block.get("retrieval_text") or "")
        if not retrieval_text.strip():
            raise ValueError(
                "검색 대상 HTML 표의 retrieval_text가 비었습니다: "
                f"{block.get('block_id')}"
            )
        adapted = dict(block)
        if has_structural_html:
            adapted["display_content"] = fallback_markdown_table(retrieval_text)
            quality_flag = "legacy_html_table_flattened_to_gfm"
            converted_count += 1
        else:
            # GFM 셀 안의 <br>만 평문 구분자로 바꾸면 열·행 구조는 그대로다.
            adapted["display_content"] = display_without_breaks
            quality_flag = "gfm_inline_break_sanitized"
            sanitized_break_count += 1
        adapted["render_mode"] = "gfm"
        adapted["quality_flags"] = list(
            dict.fromkeys(
                [
                    *(block.get("quality_flags") or []),
                    quality_flag,
                ]
            )
        )
        adapted_blocks.append(adapted)
    return adapted_blocks, converted_count, sanitized_break_count


def write_deterministic_jsonl_gzip(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    """gzip 시간·파일명 헤더를 고정해 같은 청크의 SHA가 같게 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as binary_stream:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=binary_stream,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                newline="\n",
            ) as text_stream:
                for row in rows:
                    text_stream.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    text_stream.write("\n")
    temporary.replace(path)
    path.chmod(0o660)


def write_report(path: Path, report: dict[str, Any]) -> None:
    """사람이 검토하기 쉬운 JSON 보고서를 원자적으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(0o660)


def build_parser() -> argparse.ArgumentParser:
    """입력과 새 출력 경로를 화면에서 명확히 확인할 수 있게 정의한다."""
    parser = argparse.ArgumentParser(
        description=(
            "구조 전처리 JSONL을 RecursiveCharacterTextSplitter 기반 "
            "512/102 청크로 변환합니다."
        )
    )
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS_PATH)
    parser.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 출력과 보고서를 명시적으로 교체합니다.",
    )
    return parser


def main() -> None:
    """전체 청킹·품질 검증에 성공한 경우에만 새 결과를 저장한다."""
    args = build_parser().parse_args()
    os.umask(0o007)
    for input_path in (args.documents, args.blocks):
        if not input_path.is_file():
            raise FileNotFoundError(f"입력 파일이 없습니다: {input_path}")
    if not args.overwrite:
        for output_path in (args.output, args.report):
            if output_path.exists():
                raise FileExistsError(
                    f"기존 결과를 보호합니다: {output_path} (교체하려면 --overwrite)"
                )

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    documents = read_jsonl(args.documents)
    raw_blocks = read_jsonl(args.blocks)
    (
        blocks,
        converted_html_table_count,
        sanitized_gfm_break_count,
    ) = adapt_legacy_html_tables(raw_blocks)
    config = ChunkConfig()
    codec = TiktokenCodec(config.model_name, config.encoding_name)
    chunks = build_chunk_corpus(documents, blocks, codec, config)
    validation = validate_chunk_corpus(
        documents,
        blocks,
        chunks,
        codec,
        config,
    )
    if not validation["overall_pass"]:
        failed = [name for name, passed in validation["gates"].items() if not passed]
        raise ValueError(f"청킹 품질 검증 실패: {', '.join(failed)}")

    write_deterministic_jsonl_gzip(args.output, chunks)
    finished_at = datetime.now(timezone.utc)
    elapsed_seconds = time.perf_counter() - started
    report = {
        "started_at_utc": started_at.isoformat(timespec="seconds"),
        "finished_at_utc": finished_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "documents_path": str(args.documents),
        "documents_sha256": sha256_file(args.documents),
        "blocks_path": str(args.blocks),
        "blocks_sha256": sha256_file(args.blocks),
        "legacy_html_table_blocks_converted": converted_html_table_count,
        "gfm_inline_break_tables_sanitized": sanitized_gfm_break_count,
        "output_path": str(args.output),
        "output_sha256": sha256_file(args.output),
        "summary": build_summary(
            documents,
            blocks,
            chunks,
            validation,
            codec,
            config,
        ),
    }
    write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
