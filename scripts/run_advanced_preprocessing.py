"""Advanced RAG 구조 전처리 결과를 별도 JSONL 파일로 생성한다."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from src.loader.load_documents import load_documents, sha256_file
from src.preprocessing.prepare_advanced import (
    ADVANCED_SCHEMA_VERSION,
    AdvancedPreprocessingResult,
    build_sources_from_manifest,
    load_document_manifest,
    preflight_required_parsers,
    prepare_advanced_document,
)

DEFAULT_SOURCE_DIR = Path("/home/data/advanced/input_v1/raw/files")
DEFAULT_MANIFEST = Path("/home/data/advanced/input_v1/documents_v2.jsonl")
DEFAULT_RECOVERY_DIR = Path("/home/data/advanced/input_v1/hwpx")
DEFAULT_OUTPUT_DIR = Path("/home/data/advanced/preprocessed_v1")

OUTPUT_FILENAMES = {
    "documents": "documents_advanced_v1.jsonl",
    "blocks": "blocks_advanced_v1.jsonl",
    "tables": "tables_advanced_v1.jsonl",
    "images": "images_advanced_v1.jsonl",
}
REPORT_FILENAME = "advanced_preprocessing_report.json"


def parse_args() -> argparse.Namespace:
    """GCP 기본 경로를 제공하되 테스트·재현용 경로 재지정도 허용한다."""
    parser = argparse.ArgumentParser(
        description="HWP 문단/PDF 페이지 경계와 표 이중 표현을 생성합니다.",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--recovery-dir", type=Path, default=DEFAULT_RECOVERY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="특정 source_id만 처리합니다. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="앞에서부터 지정한 문서 수만 처리하는 스모크 옵션입니다.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="입력 경로와 manifest 연결만 확인하고 원본은 파싱하지 않습니다.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="같은 출력 파일이 있으면 성공한 새 결과로 교체합니다.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    """보고서 시각을 timezone이 명확한 초 단위 ISO 문자열로 만든다."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_jsonl(file: TextIO, records: Iterable[dict[str, Any]]) -> int:
    """레코드를 UTF-8 JSONL로 기록하고 쓴 행 수를 반환한다."""
    count = 0
    for record in records:
        file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        file.write("\n")
        count += 1
    return count


def _selected_sources(
    sources: tuple[Any, ...],
    requested_source_ids: list[str],
    max_documents: int | None,
) -> tuple[Any, ...]:
    """source_id 필터와 스모크 처리 수를 결정적인 manifest 순서로 적용한다."""
    if max_documents is not None and max_documents < 1:
        raise ValueError("--max-documents는 1 이상이어야 합니다")

    requested = set(requested_source_ids)
    known = {source.source_id for source in sources}
    unknown = requested - known
    if unknown:
        raise ValueError(f"manifest에 없는 source_id입니다: {sorted(unknown)}")

    selected = tuple(
        source for source in sources if not requested or source.source_id in requested
    )
    if max_documents is not None:
        selected = selected[:max_documents]
    if not selected:
        raise ValueError("처리할 문서가 없습니다")
    return selected


def _output_paths(output_dir: Path) -> dict[str, Path]:
    """유형별 JSONL과 보고서의 최종 경로를 만든다."""
    paths = {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}
    paths["report"] = output_dir / REPORT_FILENAME
    return paths


def _ensure_output_policy(paths: dict[str, Path], overwrite: bool) -> None:
    """사용자 승인 없이 기존 결과를 조용히 덮어쓰지 않는다."""
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"출력 파일이 이미 있습니다. 교체하려면 --overwrite를 지정하세요: {joined}"
        )


def _temporary_paths(paths: dict[str, Path]) -> dict[str, Path]:
    """완료 전 결과가 최종 파일처럼 보이지 않게 임시 경로를 사용한다."""
    return {name: path.with_name(f".{path.name}.tmp") for name, path in paths.items()}


def _cleanup_temporary(paths: dict[str, Path]) -> None:
    """실패한 실행의 임시 파일만 지우고 이전 최종 결과는 보존한다."""
    for path in paths.values():
        path.unlink(missing_ok=True)


def _validation_summary(
    sources: tuple[Any, ...],
    analysis_sources: dict[str, Any],
    manifest_path: Path,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    """원본 파싱 전에 확인 가능한 입력 계약을 요약한다."""
    return {
        "schema_version": ADVANCED_SCHEMA_VERSION,
        "manifest_path": str(manifest_path.expanduser().resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "document_count": len(sources),
        "hwp_document_count": sum(
            source.file_type in {"hwp", "hwpx"} for source in sources
        ),
        "pdf_document_count": sum(source.file_type == "pdf" for source in sources),
        "recovery_document_count": len(analysis_sources),
        "source_ids_unique": len({source.source_id for source in sources})
        == len(sources),
        **inventory,
    }


def _source_inventory_summary(
    source_dir: Path,
    manifest_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """원본 100개가 manifest의 고유 문서 98개와 같은 SHA 집합인지 확인한다."""
    loaded_sources = load_documents(source_dir)
    raw_source_ids = {source.source_id for source in loaded_sources}
    manifest_source_ids = {str(record["source_id"]) for record in manifest_records}
    if raw_source_ids != manifest_source_ids:
        missing = sorted(raw_source_ids - manifest_source_ids)
        unknown = sorted(manifest_source_ids - raw_source_ids)
        raise ValueError(
            "원본과 documents_v2의 source_id 집합이 다릅니다: "
            f"manifest 누락={missing}, 원본에 없음={unknown}"
        )
    return {
        "raw_file_count": len(loaded_sources),
        "raw_unique_document_count": len(raw_source_ids),
        "raw_duplicate_file_count": len(loaded_sources) - len(raw_source_ids),
        "manifest_matches_raw_source_ids": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """입력을 검증하고 문서별 결과를 스트리밍해 원자적으로 저장한다."""
    manifest_records = load_document_manifest(args.manifest)
    inventory = _source_inventory_summary(args.source_dir, manifest_records)
    sources, analysis_sources, manifest_by_source_id = build_sources_from_manifest(
        manifest_records,
        source_dir=args.source_dir,
        recovery_dir=args.recovery_dir,
    )
    selected_sources = _selected_sources(
        sources,
        args.source_id,
        args.max_documents,
    )
    selected_analysis_sources = {
        source_id: analysis_source
        for source_id, analysis_source in analysis_sources.items()
        if source_id in {source.source_id for source in selected_sources}
    }
    summary = _validation_summary(
        selected_sources,
        selected_analysis_sources,
        args.manifest,
        inventory,
    )
    if args.validate_only:
        return summary

    # rhwp ABI 오류 등은 출력 폴더나 임시 파일을 만들기 전에 중단한다.
    preflight_required_parsers(selected_sources)

    output_dir = args.output_dir.expanduser().resolve()
    paths = _output_paths(output_dir)
    _ensure_output_policy(paths, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_paths(paths)
    _cleanup_temporary(temporary)

    started_at = _utc_now()
    started = time.perf_counter()
    counts = {
        "documents": 0,
        "blocks": 0,
        "tables": 0,
        "images": 0,
        "dense_eligible_blocks": 0,
        "kss_eligible_blocks": 0,
        "bm25_eligible_blocks": 0,
        "table_blocks_with_dual_formats": 0,
        "pdf_table_text_fallback_count": 0,
        "pdf_table_text_fallback_page_count": 0,
        "pdf_table_geometry_recovered_count": 0,
        "pdf_table_geometry_recovered_page_count": 0,
        "pdf_table_one_column_fallback_count": 0,
        "pdf_table_fallback_markdown_blocks": 0,
    }

    handles: dict[str, TextIO] = {}
    try:
        for name in OUTPUT_FILENAMES:
            handles[name] = temporary[name].open("w", encoding="utf-8", newline="\n")

        for index, source in enumerate(selected_sources, start=1):
            result: AdvancedPreprocessingResult = prepare_advanced_document(
                source,
                manifest_by_source_id[source.source_id],
                analysis_source=selected_analysis_sources.get(source.source_id),
            )
            counts["documents"] += _write_jsonl(
                handles["documents"],
                [result.document],
            )
            counts["blocks"] += _write_jsonl(handles["blocks"], result.blocks)
            counts["tables"] += _write_jsonl(handles["tables"], result.tables)
            counts["images"] += _write_jsonl(handles["images"], result.images)
            counts["dense_eligible_blocks"] += sum(
                bool(block["dense_eligible"]) for block in result.blocks
            )
            counts["kss_eligible_blocks"] += sum(
                bool(block["kss_eligible"]) for block in result.blocks
            )
            counts["bm25_eligible_blocks"] += sum(
                bool(block["bm25_eligible"]) for block in result.blocks
            )
            counts["table_blocks_with_dual_formats"] += sum(
                block["content_type"] == "table"
                and bool(block["table_html"])
                and bool(block["table_markdown"])
                for block in result.blocks
            )
            counts["pdf_table_text_fallback_count"] += int(
                result.document.get("pdf_table_text_fallback_count", 0)
            )
            counts["pdf_table_text_fallback_page_count"] += int(
                result.document.get("pdf_table_text_fallback_page_count", 0)
            )
            counts["pdf_table_geometry_recovered_count"] += int(
                result.document.get("pdf_table_geometry_recovered_count", 0)
            )
            counts["pdf_table_geometry_recovered_page_count"] += int(
                result.document.get("pdf_table_geometry_recovered_page_count", 0)
            )
            counts["pdf_table_one_column_fallback_count"] += int(
                result.document.get("pdf_table_one_column_fallback_count", 0)
            )
            counts["pdf_table_fallback_markdown_blocks"] += sum(
                block["content_type"] == "table"
                and block.get("index_reason") == "incomplete_pdf_table_bbox_text"
                and block.get("vectorize_field") == "table_markdown"
                and bool(block.get("dense_eligible"))
                and not bool(block.get("kss_eligible"))
                and not bool(block.get("bm25_eligible"))
                for block in result.blocks
            )
            print(
                f"진행: {index}/{len(selected_sources)} "
                f"({source.source_id}, {source.source_filename})",
                flush=True,
            )

        for handle in handles.values():
            handle.flush()
            handle.close()
        handles.clear()

        for name in OUTPUT_FILENAMES:
            temporary[name].replace(paths[name])

        finished_at = _utc_now()
        report = {
            **summary,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "total_seconds": round(time.perf_counter() - started, 6),
            "source_dir": str(args.source_dir.expanduser().resolve()),
            "recovery_dir": str(args.recovery_dir.expanduser().resolve()),
            "output_dir": str(output_dir),
            "output_counts": counts,
            "output_files": {name: str(paths[name]) for name in OUTPUT_FILENAMES},
            "output_sha256": {
                name: sha256_file(paths[name]) for name in OUTPUT_FILENAMES
            },
            "table_storage": "html_and_markdown_in_blocks_only",
            "embedding_policy": "text_or_table_markdown_without_metadata_prefix",
            "kss_status": "not_applied_next_stage",
            "kiwi_status": "not_applied_next_stage",
        }
        temporary["report"].write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary["report"].replace(paths["report"])
        return report
    except Exception:
        for handle in handles.values():
            handle.close()
        _cleanup_temporary(temporary)
        raise


def main() -> None:
    """CLI 결과를 사람이 확인하기 쉬운 JSON으로 출력한다."""
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
