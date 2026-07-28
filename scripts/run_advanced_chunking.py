"""Advanced 전처리 JSONL을 의미 기반 KSS 512/51 v2 청크로 저장하는 GCP CLI.

일반 텍스트는 KSS 문장 경계를 사용하고, 표는 KSS·BM25에서 제외한 채
Markdown 행 구조를 보존해 청킹한다. 파일명·위치 같은 메타데이터 prefix는
임베딩 문자열에 넣지 않는다. 모든 품질 게이트를 통과한 경우에만 결정적인
gzip JSONL과 실행 보고서를 새 경로에 원자적으로 저장한다.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.chunking.advanced_chunking import (
    CORPUS_ID,
    INPUT_SCHEMA_VERSION,
    PAGE_MARKER_DETECTOR_ID,
    SCHEMA_VERSION,
    STRATEGY_ID,
    TEXT_EMBEDDING_NORMALIZATION_ID,
    AdvancedChunkConfig,
    TiktokenCodec,
    build_advanced_chunk_corpus,
    build_advanced_summary,
    normalize_text_for_embedding,
    validate_advanced_chunks,
)

ADVANCED_PREPROCESSING_SCHEMA_VERSION = "rfp_advanced_preprocessing_v1"
RUN_REPORT_SCHEMA_VERSION = "rfp_advanced_chunking_run_v2"
PREDECESSOR_V1_SHA256 = (
    "6868170ac8777952f5fcd887bb35ce4d150c48348a736447a711575fb0314679"
)

DEFAULT_DOCUMENTS_PATH = Path(
    "/home/data/advanced/preprocessed_v1/documents_advanced_v1.jsonl"
)
DEFAULT_BLOCKS_PATH = Path(
    "/home/data/advanced/preprocessed_v1/blocks_advanced_v1.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("/home/data/advanced/chunks_kss_512_51_v2")
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "chunks_advanced_v2.jsonl.gz"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "advanced_chunking_report_v2.json"
PROTECTED_V1_OUTPUT_PATH = Path(
    "/home/data/advanced/chunks_kss_512_51_v1/chunks_advanced_v1.jsonl.gz"
)
PROTECTED_V1_REPORT_PATH = Path(
    "/home/data/advanced/chunks_kss_512_51_v1/advanced_chunking_report.json"
)


def sha256_file(path: Path) -> str:
    """대용량 입력·출력도 메모리에 올리지 않고 SHA-256을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL을 읽고 형식 오류가 난 파일과 줄 번호를 함께 알린다."""
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
    if not rows:
        raise ValueError(f"입력 JSONL이 비어 있습니다: {path}")
    return rows


def _require_advanced_schema(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    """Naive 또는 다른 버전의 레코드가 섞이면 청킹 전에 중단한다."""
    invalid = [
        index
        for index, row in enumerate(rows, start=1)
        if row.get("schema_version") != ADVANCED_PREPROCESSING_SCHEMA_VERSION
    ]
    if invalid:
        samples = ", ".join(str(index) for index in invalid[:10])
        raise ValueError(
            f"{label} schema_version이 Advanced v1이 아닙니다: 행 {samples}"
        )


def validate_advanced_inputs(
    documents: Sequence[Mapping[str, Any]],
    blocks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """API 호출 전 입력 스키마·연결·유형별 후속 처리 계약을 검사한다."""
    _require_advanced_schema(documents, label="documents")
    _require_advanced_schema(blocks, label="blocks")

    source_ids = [str(row.get("source_id") or "") for row in documents]
    if any(not source_id for source_id in source_ids):
        raise ValueError("documents에 빈 source_id가 있습니다")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("documents에 중복 source_id가 있습니다")
    known_source_ids = set(source_ids)

    for document in documents:
        if document.get("embedding_prefix_policy") != (
            "metadata_only_not_in_vector_text"
        ):
            raise ValueError(
                f"문서의 embedding prefix 정책이 다릅니다: {document.get('source_id')}"
            )
        if str(document.get("file_type") or "").casefold() not in {
            "hwp",
            "hwpx",
            "pdf",
        }:
            raise ValueError(
                f"지원하지 않는 file_type입니다: {document.get('file_type')}"
            )

    block_ids: set[str] = set()
    orders_by_source: dict[str, list[int]] = defaultdict(list)
    content_counts: dict[str, int] = defaultdict(int)
    dense_counts: dict[str, int] = defaultdict(int)
    kss_eligible_count = 0
    bm25_eligible_count = 0

    for row_number, block in enumerate(blocks, start=1):
        source_id = str(block.get("source_id") or "")
        if source_id not in known_source_ids:
            raise ValueError(
                f"blocks {row_number}행이 알 수 없는 source_id를 참조합니다: "
                f"{source_id}"
            )

        block_id = str(block.get("block_id") or "")
        if not block_id:
            raise ValueError(f"blocks {row_number}행에 block_id가 없습니다")
        if block_id in block_ids:
            raise ValueError(f"중복 block_id가 있습니다: {block_id}")
        block_ids.add(block_id)

        block_order = block.get("block_order")
        if not isinstance(block_order, int) or block_order < 1:
            raise ValueError(f"유효하지 않은 block_order입니다: {block_id}")
        orders_by_source[source_id].append(block_order)

        if block.get("embedding_prefix_included") is not False:
            raise ValueError(
                f"벡터화 본문에 metadata prefix가 포함된 블록입니다: {block_id}"
            )

        content_type = str(block.get("content_type") or "")
        if content_type not in {"text", "table", "image"}:
            raise ValueError(f"알 수 없는 content_type입니다: {block_id}")
        content_counts[content_type] += 1
        if block.get("dense_eligible"):
            dense_counts[content_type] += 1
        kss_eligible_count += bool(block.get("kss_eligible"))
        bm25_eligible_count += bool(block.get("bm25_eligible"))

        if content_type == "text":
            expected_field = "text" if block.get("dense_eligible") else None
            if block.get("vectorize_field") != expected_field:
                raise ValueError(f"텍스트 vectorize_field 오류입니다: {block_id}")
            if block.get("dense_eligible"):
                if not str(block.get("text") or "").strip():
                    raise ValueError(f"색인 대상 text가 비었습니다: {block_id}")
                if not block.get("kss_eligible") or not block.get("bm25_eligible"):
                    raise ValueError(f"텍스트 KSS/BM25 플래그 오류입니다: {block_id}")
                if not block.get("kss_boundary_type") or not block.get(
                    "kss_boundary_id"
                ):
                    raise ValueError(f"텍스트 KSS 경계가 없습니다: {block_id}")
        elif content_type == "table":
            if block.get("kss_eligible") or block.get("bm25_eligible"):
                raise ValueError(
                    f"표가 KSS 또는 BM25 대상으로 표시됐습니다: {block_id}"
                )
            expected_field = "table_markdown" if block.get("dense_eligible") else None
            if block.get("vectorize_field") != expected_field:
                raise ValueError(f"표 vectorize_field 오류입니다: {block_id}")
            if (
                block.get("dense_eligible")
                and not str(block.get("table_markdown") or "").strip()
            ):
                raise ValueError(f"색인 대상 table_markdown이 비었습니다: {block_id}")
        else:
            if any(
                block.get(field)
                for field in ("dense_eligible", "kss_eligible", "bm25_eligible")
            ):
                raise ValueError(f"이미지가 검색 대상으로 표시됐습니다: {block_id}")
            if block.get("vectorize_field") is not None:
                raise ValueError(
                    f"이미지 vectorize_field가 비어 있지 않습니다: {block_id}"
                )

    for source_id, orders in orders_by_source.items():
        if orders != list(range(1, len(orders) + 1)):
            raise ValueError(f"block_order가 1부터 연속되지 않습니다: {source_id}")

    return {
        "overall_pass": True,
        "schema_version": ADVANCED_PREPROCESSING_SCHEMA_VERSION,
        "document_count": len(documents),
        "block_count": len(blocks),
        "content_counts": dict(sorted(content_counts.items())),
        "dense_eligible_counts": dict(sorted(dense_counts.items())),
        "kss_eligible_block_count": kss_eligible_count,
        "bm25_eligible_block_count": bm25_eligible_count,
        "source_ids_unique": True,
        "block_ids_unique": True,
        "block_orders_contiguous": True,
        "embedding_prefix_included": False,
    }


def select_documents(
    documents: Sequence[dict[str, Any]],
    blocks: Sequence[dict[str, Any]],
    max_documents: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """스모크 실행은 입력 문서 순서의 앞 N개와 연결 블록만 선택한다."""
    if max_documents is None:
        return list(documents), list(blocks)
    if max_documents < 1:
        raise ValueError("--max-documents는 1 이상이어야 합니다")

    selected_documents = list(documents[:max_documents])
    selected_source_ids = {
        str(document["source_id"]) for document in selected_documents
    }
    selected_blocks = [
        block for block in blocks if str(block.get("source_id")) in selected_source_ids
    ]
    return selected_documents, selected_blocks


def validate_no_embedding_prefix(chunks: Sequence[Mapping[str, Any]]) -> None:
    """최종 본문이 유형별 raw→embedding 계약과 prefix 정책을 지키는지 본다."""
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk.get("embedding_prefix_included") is not False:
            raise ValueError(f"청크에 metadata prefix 플래그가 있습니다: {chunk_id}")
        raw_text = chunk.get("raw_text")
        embedding_text = chunk.get("embedding_text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(f"청크 raw_text가 비었습니다: {chunk_id}")
        if not isinstance(embedding_text, str) or not embedding_text.strip():
            raise ValueError(f"청크 embedding_text가 비었습니다: {chunk_id}")
        expected = (
            normalize_text_for_embedding(raw_text)
            if chunk.get("content_type") == "text"
            else raw_text
        )
        if embedding_text != expected:
            raise ValueError(
                "Advanced embedding_text가 유형별 raw 정규화 계약과 다릅니다: "
                f"{chunk_id}"
            )


def _temporary_path(path: Path) -> Path:
    """동시 실행과 미완료 파일을 구분할 같은 폴더의 임시 경로를 만든다."""
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def write_deterministic_jsonl_gzip(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """gzip 시간·파일명 헤더를 고정한 임시 JSONL 파일을 작성한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.unlink(missing_ok=True)
    try:
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
        temporary.chmod(0o660)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_report_temporary(path: Path, report: Mapping[str, Any]) -> Path:
    """사람이 검토할 보고서를 최종 교체 전 임시 파일에 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o660)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def ensure_output_policy(paths: Iterable[Path], overwrite: bool) -> None:
    """기존 v1은 항상 보호하고 v2도 명시적 승인 없이 덮어쓰지 않는다."""
    protected = {
        PROTECTED_V1_OUTPUT_PATH.expanduser().resolve(),
        PROTECTED_V1_REPORT_PATH.expanduser().resolve(),
    }
    selected = {path.expanduser().resolve() for path in paths}
    collisions = sorted(str(path) for path in selected & protected)
    if collisions:
        raise PermissionError(
            "검증 완료된 Advanced v1 경로는 --overwrite로도 변경할 수 없습니다: "
            + ", ".join(collisions)
        )
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"기존 결과를 보호합니다: {joined} (교체하려면 --overwrite)"
        )


def build_parser() -> argparse.ArgumentParser:
    """GCP 기본 경로와 검증·스모크·덮어쓰기 옵션을 정의한다."""
    parser = argparse.ArgumentParser(
        description=(
            "Advanced 전처리 JSONL을 의미 기반 KSS 512/51 v2 청크와 "
            "Markdown 표 청크로 변환하고 PDF 페이지 표식·일반 텍스트 "
            "줄바꿈을 임베딩 본문에서 제거합니다."
        )
    )
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS_PATH)
    parser.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="입력 순서의 앞 N개 문서만 청킹하는 스모크 옵션입니다.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="입력 스키마와 연결만 검사하고 KSS 청킹·파일 저장은 생략합니다.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 출력과 보고서를 명시적으로 교체합니다.",
    )
    return parser


def _failed_validation_gates(validation: Mapping[str, Any]) -> list[str]:
    """모듈 검증 결과에서 실패한 품질 게이트 이름을 추출한다."""
    gates = validation.get("gates")
    if not isinstance(gates, Mapping):
        return ["overall_pass"]
    return [str(name) for name, passed in gates.items() if not passed]


def main() -> None:
    """입력을 감사하고 검증된 Advanced 청크와 보고서만 저장한다."""
    args = build_parser().parse_args()
    os.umask(0o007)

    for input_path in (args.documents, args.blocks):
        if not input_path.is_file():
            raise FileNotFoundError(f"입력 파일이 없습니다: {input_path}")

    documents_sha256 = sha256_file(args.documents)
    blocks_sha256 = sha256_file(args.blocks)
    documents = read_jsonl(args.documents)
    blocks = read_jsonl(args.blocks)
    input_validation = validate_advanced_inputs(documents, blocks)
    selected_documents, selected_blocks = select_documents(
        documents,
        blocks,
        args.max_documents,
    )
    selection_validation = validate_advanced_inputs(
        selected_documents,
        selected_blocks,
    )

    config = AdvancedChunkConfig()
    codec = TiktokenCodec(config.model_name, config.encoding_name)
    validation_report = {
        "schema_version": RUN_REPORT_SCHEMA_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "chunk_schema_version": SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "strategy_id": STRATEGY_ID,
        "page_marker_detector_id": PAGE_MARKER_DETECTOR_ID,
        "embedding_text_field": "embedding_text",
        "text_embedding_normalization": TEXT_EMBEDDING_NORMALIZATION_ID,
        "table_embedding_normalization": "preserve_markdown_newlines",
        "token_count_basis": "embedding_text",
        "overlap_token_basis": "normalized_embedding_text",
        "bm25_source_field": "embedding_text",
        "mode": "validate_only",
        "documents_path": str(args.documents.expanduser().resolve()),
        "documents_sha256": documents_sha256,
        "blocks_path": str(args.blocks.expanduser().resolve()),
        "blocks_sha256": blocks_sha256,
        "full_input_validation": input_validation,
        "selected_input_validation": selection_validation,
        "max_documents": args.max_documents,
        "tokenizer_model": codec.model_name,
        "tokenizer_encoding": codec.encoding_name,
        "tokenizer_version": codec.version,
        "max_tokens": config.max_tokens,
        "overlap_tokens": config.overlap_tokens,
        "min_tail_tokens": config.min_tail_tokens,
        "embedding_prefix_included": False,
    }
    if args.validate_only:
        print(
            json.dumps(
                validation_report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    ensure_output_policy((args.output, args.report), args.overwrite)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()

    chunks = build_advanced_chunk_corpus(
        selected_documents,
        selected_blocks,
        codec,
        config,
    )
    validation = validate_advanced_chunks(
        selected_documents,
        selected_blocks,
        chunks,
        codec,
        config,
    )
    if not validation.get("overall_pass"):
        failed = _failed_validation_gates(validation)
        raise ValueError(f"Advanced 청킹 품질 검증 실패: {', '.join(failed)}")
    validate_no_embedding_prefix(chunks)

    temporary_output: Path | None = None
    temporary_report: Path | None = None
    try:
        temporary_output = write_deterministic_jsonl_gzip(args.output, chunks)
        output_sha256 = sha256_file(temporary_output)
        finished_at = datetime.now(timezone.utc)
        report = {
            "schema_version": RUN_REPORT_SCHEMA_VERSION,
            "input_schema_version": INPUT_SCHEMA_VERSION,
            "chunk_schema_version": SCHEMA_VERSION,
            "corpus_id": CORPUS_ID,
            "strategy_id": STRATEGY_ID,
            "page_marker_detector_id": PAGE_MARKER_DETECTOR_ID,
            "embedding_text_field": "embedding_text",
            "text_embedding_normalization": TEXT_EMBEDDING_NORMALIZATION_ID,
            "table_embedding_normalization": "preserve_markdown_newlines",
            "token_count_basis": "embedding_text",
            "overlap_token_basis": "normalized_embedding_text",
            "bm25_source_field": "embedding_text",
            "predecessor_v1_path": str(PROTECTED_V1_OUTPUT_PATH),
            "predecessor_v1_sha256": PREDECESSOR_V1_SHA256,
            "started_at_utc": started_at.isoformat(timespec="seconds"),
            "finished_at_utc": finished_at.isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "documents_path": str(args.documents.expanduser().resolve()),
            "documents_sha256": documents_sha256,
            "blocks_path": str(args.blocks.expanduser().resolve()),
            "blocks_sha256": blocks_sha256,
            "full_input_validation": input_validation,
            "selected_input_validation": selection_validation,
            "max_documents": args.max_documents,
            "output_path": str(args.output.expanduser().resolve()),
            "output_sha256": output_sha256,
            "output_chunk_count": len(chunks),
            "report_path": str(args.report.expanduser().resolve()),
            "embedding_prefix_included": False,
            "summary": build_advanced_summary(
                selected_documents,
                selected_blocks,
                chunks,
                validation,
                codec,
                config,
            ),
        }
        temporary_report = write_report_temporary(args.report, report)

        # 두 최종 파일 모두 준비된 뒤에만 기존 결과를 원자적으로 교체한다.
        temporary_output.replace(args.output)
        temporary_output = None
        temporary_report.replace(args.report)
        temporary_report = None
        args.output.chmod(0o660)
        args.report.chmod(0o660)
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
