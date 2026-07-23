"""Advanced RAG 청킹 전에 사용할 구조 전처리 레코드를 만든다.

기존 Naive 전처리 함수로 검증된 문서 순서와 위치 정보를 재사용하되,
Advanced 전용 출력에서는 일반 텍스트·표·이미지를 명확히 분리한다.

* HWP/HWPX 일반 텍스트는 section/paragraph 위치를 유지한다.
* PDF 일반 텍스트는 실제 1-based page 위치를 유지한다.
* 표는 HTML과 Markdown을 함께 저장하고 Markdown만 벡터화 대상으로 표시한다.
* 이미지는 바이트나 Base64 없이 ``image://`` 참조만 저장한다.
* 파일명·위치·유형은 메타데이터로만 두며 벡터화 문자열에 붙이지 않는다.

KSS 문장 분리, 512-token packing, Kiwi 형태소 분석은 다음 단계가 담당한다.
이 모듈은 그 단계가 안전하게 경계를 지킬 수 있도록 ``kss_boundary_*``와
``vectorize_field`` 계약을 제공한다.
"""

from __future__ import annotations

import importlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.loader.load_documents import SourceDocument
from src.preprocessing.clean_text import (
    HWP_FILE_TYPES,
    PDF_FILE_TYPE,
    PreprocessingDependencyError,
    PreprocessingResult,
    VerifiedAnalysisSource,
    build_hwp_structure_maps,
    compact_text,
    deduplicate_pdf_tables,
    furniture_roots_with_type,
    has_forbidden_image_payload,
    kind_name,
    preprocess_document,
    walk_blocks_with_depth,
)
from src.preprocessing.table_formats import (
    build_hwp_table_formats,
    build_pdf_table_formats,
)

ADVANCED_SCHEMA_VERSION = "rfp_advanced_preprocessing_v1"
TABLE_FORMAT_VERSION = "html_gfm_dual_v1"
INDEXABLE_POLICIES = frozenset({"index", "flatten"})
PDF_TABLE_TEXT_FALLBACK_REASON = "incomplete_pdf_table_bbox_text"


@dataclass(frozen=True, slots=True)
class AdvancedPreprocessingResult:
    """문서 하나의 Advanced 구조 전처리 결과다."""

    document: dict[str, Any]
    blocks: tuple[dict[str, Any], ...]
    tables: tuple[dict[str, Any], ...]
    images: tuple[dict[str, Any], ...]


def load_document_manifest(path: str | Path) -> list[dict[str, Any]]:
    """검증된 documents_v2 JSONL을 읽고 source_id 중복을 차단한다."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"문서 manifest를 찾을 수 없습니다: {manifest_path}")

    records: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    with manifest_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"문서 manifest {line_number}행이 올바른 JSON이 아닙니다"
                ) from error
            source_id = str(record.get("source_id") or "")
            if not source_id:
                raise ValueError(
                    f"문서 manifest {line_number}행에 source_id가 없습니다"
                )
            if source_id in source_ids:
                raise ValueError(
                    f"문서 manifest에 source_id가 중복됐습니다: {source_id}"
                )
            source_ids.add(source_id)
            records.append(record)

    if not records:
        raise ValueError("문서 manifest가 비어 있습니다")
    return records


def _normalized_path_index(root: Path) -> dict[str, Path]:
    """macOS NFD와 Linux NFC 파일명이 달라도 같은 원본을 찾게 한다."""
    if not root.is_dir():
        raise NotADirectoryError(f"입력 폴더를 찾을 수 없습니다: {root}")

    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = unicodedata.normalize("NFC", str(path.relative_to(root)))
        previous = index.setdefault(relative, path)
        if previous != path:
            raise ValueError(f"정규화 후 같은 경로가 두 개입니다: {relative}")
    return index


def _resolve_manifest_path(
    path_index: dict[str, Path],
    relative_path: str,
    *,
    label: str,
) -> Path:
    """manifest 상대경로를 실제 파일로 바꾸고 누락을 명확히 알린다."""
    normalized = unicodedata.normalize("NFC", relative_path)
    try:
        return path_index[normalized]
    except KeyError as error:
        raise FileNotFoundError(
            f"{label} 파일을 찾을 수 없습니다: {relative_path}"
        ) from error


def build_sources_from_manifest(
    records: list[dict[str, Any]],
    *,
    source_dir: str | Path,
    recovery_dir: str | Path | None = None,
) -> tuple[
    tuple[SourceDocument, ...],
    dict[str, VerifiedAnalysisSource],
    dict[str, dict[str, Any]],
]:
    """documents_v2를 SourceDocument와 검증된 복구 파일 연결로 변환한다."""
    source_root = Path(source_dir).expanduser().resolve()
    source_paths = _normalized_path_index(source_root)
    recovery_paths: dict[str, Path] = {}
    if recovery_dir is not None:
        recovery_root = Path(recovery_dir).expanduser().resolve()
        if recovery_root.exists():
            recovery_paths = _normalized_path_index(recovery_root)

    sources: list[SourceDocument] = []
    analysis_sources: dict[str, VerifiedAnalysisSource] = {}
    manifest_by_source_id: dict[str, dict[str, Any]] = {}

    for record in records:
        source_id = str(record["source_id"])
        source_sha256 = str(record.get("source_sha256") or "")
        if len(source_sha256) != 64 or source_id != source_sha256[:16]:
            raise ValueError(f"manifest의 source_id/SHA 규칙이 다릅니다: {source_id}")

        relative_path = str(record.get("source_relative_path") or "")
        source_path = _resolve_manifest_path(
            source_paths,
            relative_path,
            label="원본",
        )
        filename_aliases = tuple(record.get("filename_aliases") or [])
        all_filenames = tuple(record.get("all_source_filenames") or [])
        if not all_filenames:
            all_filenames = (str(record.get("source_filename") or source_path.name),)
        duplicate_group_size = max(
            int(record.get("duplicate_alias_count") or len(filename_aliases)) + 1,
            1,
        )

        source = SourceDocument(
            source_id=source_id,
            document_id=str(record.get("document_id") or source_id),
            source_path=source_path,
            source_relative_path=relative_path,
            source_filename=str(record.get("source_filename") or source_path.name),
            source_sha256=source_sha256,
            file_type=str(record.get("file_type") or source_path.suffix[1:]).casefold(),
            source_file_size_bytes=int(
                record.get("source_file_size_bytes") or source_path.stat().st_size
            ),
            duplicate_group_size=duplicate_group_size,
            is_default_canonical=True,
            default_canonical_filename=str(
                record.get("source_filename") or source_path.name
            ),
            filename_aliases=filename_aliases,
            source_relative_path_aliases=(),
            all_source_filenames=all_filenames,
            # documents_v2는 팀이 EDA에서 확정한 대표 파일 manifest다.
            canonical_selection_source="team_policy",
            canonical_selection_reason=str(
                record.get("canonical_selection_reason")
                or "documents_v2_verified_canonical"
            ),
        )
        sources.append(source)
        manifest_by_source_id[source_id] = record

        analysis_sha256 = str(record.get("analysis_source_sha256") or source_sha256)
        analysis_relative_path = str(
            record.get("analysis_source_relative_path") or relative_path
        )
        uses_alternate = (
            analysis_sha256 != source_sha256
            or str(record.get("analysis_file_type") or source.file_type).casefold()
            != source.file_type
        )
        if uses_alternate:
            if not recovery_paths:
                raise FileNotFoundError(
                    f"{source_id}는 복구 분석본이 필요하지만 recovery_dir가 없습니다"
                )
            analysis_path = _resolve_manifest_path(
                recovery_paths,
                analysis_relative_path,
                label="복구 분석본",
            )
            analysis_sources[source_id] = VerifiedAnalysisSource(
                path=analysis_path,
                sha256=analysis_sha256,
                original_source_id=source_id,
                original_source_sha256=source_sha256,
                verification_source=(
                    "documents_v2:"
                    f"{record.get('recovery_status') or 'verified_alternate'}"
                ),
            )

    return tuple(sources), analysis_sources, manifest_by_source_id


def _load_rhwp_module() -> Any:
    """rhwp를 지연 import하고 GCP FreeType ABI 오류를 정확히 설명한다."""
    try:
        return importlib.import_module("rhwp")
    except ImportError as error:
        message = str(error)
        if "FT_Palette_Data_Get" in message:
            raise PreprocessingDependencyError(
                "rhwp 로드 실패: FreeType ABI 불일치(FT_Palette_Data_Get). "
                "GCP에서는 scripts/gcp_run_advanced_preprocessing.sh를 사용하세요. "
                "LD_PRELOAD를 .env나 shell profile에 영구 저장하면 안 됩니다. "
                f"원래 오류: {message}"
            ) from error
        raise PreprocessingDependencyError(
            f"rhwp-python==0.8.1 로드에 실패했습니다: {message}"
        ) from error


def preflight_required_parsers(
    sources: tuple[SourceDocument, ...],
) -> None:
    """산출물 파일을 만들기 전에 필요한 네이티브 파서를 모두 확인한다."""
    file_types = {source.file_type.casefold() for source in sources}
    if file_types & HWP_FILE_TYPES:
        _load_rhwp_module()
    if PDF_FILE_TYPE in file_types:
        try:
            importlib.import_module("pdfplumber")
        except ImportError as error:
            raise PreprocessingDependencyError(
                f"pdfplumber==0.11.10 로드에 실패했습니다: {error}"
            ) from error


def _analysis_path(
    source: SourceDocument,
    analysis_source: VerifiedAnalysisSource | None,
) -> Path:
    """기존 전처리 검증 전에 파서가 열 경로만 결정한다."""
    if analysis_source is None:
        return source.source_path.expanduser().resolve()
    return Path(analysis_source.path).expanduser().resolve()


def _hwp_base_and_formats(
    source: SourceDocument,
    *,
    analysis_source: VerifiedAnalysisSource | None,
    rhwp_module: Any | None,
) -> tuple[PreprocessingResult, dict[str, dict[str, str]]]:
    """HWP를 한 번만 to_ir한 뒤 기존 구조와 이중 표 표현을 함께 만든다."""
    parser = rhwp_module or _load_rhwp_module()
    selected_path = _analysis_path(source, analysis_source)
    parsed_document = parser.parse(str(selected_path))
    ir = parsed_document.to_ir()

    # 기존 preprocess_document가 동일한 IR을 다시 생성하지 않도록 작은 adapter를
    # 주입한다. 파일/SHA/복구 연결 검증은 기존 공개 API가 그대로 수행한다.
    cached_document = SimpleNamespace(
        to_ir=lambda: ir,
        page_count=getattr(parsed_document, "page_count", None),
        section_count=getattr(parsed_document, "section_count", None),
        paragraph_count=getattr(parsed_document, "paragraph_count", None),
    )
    cached_module = SimpleNamespace(parse=lambda _path: cached_document)
    base_result = preprocess_document(
        source,
        analysis_source=analysis_source,
        rhwp_module=cached_module,
    )

    table_ids, picture_ids, _, _ = build_hwp_structure_maps(ir, source.source_id)
    roots = list(getattr(ir, "body", []) or [])
    roots.extend(block for block, _ in furniture_roots_with_type(ir))
    formats: dict[str, dict[str, str]] = {}
    for block, _ in walk_blocks_with_depth(roots):
        if kind_name(block) != "table":
            continue
        table_id = table_ids[id(block)]
        formats[table_id] = build_hwp_table_formats(
            block,
            table_ids,
            picture_ids,
        )
    return base_result, formats


def _pdf_table_formats(
    source: SourceDocument,
    parser: Any,
    analysis_source: VerifiedAnalysisSource | None,
) -> dict[str, dict[str, str]]:
    """기존 PDF 표 탐지 순서와 같은 ID로 HTML·Markdown을 다시 만든다."""
    selected_path = _analysis_path(source, analysis_source)
    formats: dict[str, dict[str, str]] = {}
    table_number = 0
    with parser.open(selected_path) as pdf:
        for page in pdf.pages:
            try:
                detected_tables = deduplicate_pdf_tables(page.find_tables() or [])
            except Exception:
                detected_tables = []
            for detected_table in detected_tables:
                try:
                    matrix = detected_table.extract() or []
                except Exception:
                    continue
                table_number += 1
                table_id = f"{source.source_id}:pdf:T{table_number:06d}"
                formats[table_id] = build_pdf_table_formats(matrix, table_id)
    return formats


def _pdf_base_and_formats(
    source: SourceDocument,
    *,
    analysis_source: VerifiedAnalysisSource | None,
    pdfplumber_module: Any | None,
) -> tuple[PreprocessingResult, dict[str, dict[str, str]]]:
    """PDF 기본 구조를 만든 뒤 같은 탐지 순서로 표 HTML을 생성한다."""
    parser = pdfplumber_module or importlib.import_module("pdfplumber")
    base_result = preprocess_document(
        source,
        analysis_source=analysis_source,
        pdfplumber_module=parser,
    )
    formats = _pdf_table_formats(source, parser, analysis_source)
    expected_ids = {str(table["table_id"]) for table in base_result.tables}
    if set(formats) != expected_ids:
        raise ValueError(
            "PDF를 두 번 열었을 때 표 탐지 순서가 달라졌습니다: "
            f"기본={len(expected_ids)}, 이중표현={len(formats)}"
        )
    return base_result, formats


def _content_type(block: dict[str, Any]) -> str:
    """원본 블록 유형을 Advanced의 text/table/image 세 유형으로 통일한다."""
    block_type = str(block.get("block_type") or "").casefold()
    if (
        block_type == "table"
        or block.get("index_reason") == PDF_TABLE_TEXT_FALLBACK_REASON
    ):
        return "table"
    if block_type == "picture":
        return "image"
    return "text"


def _pdf_fallback_table_formats(block: dict[str, Any]) -> dict[str, str]:
    """불완전 PDF 표의 bbox 원문을 안전한 1열 HTML·Markdown 표로 만든다.

    셀 좌표를 복원할 수 없는 상황에서 행 관계를 추측하지 않고, 읽기 순서로
    보존한 각 줄을 1열 표의 한 행으로 저장한다. 이 결과는 표 Markdown만
    벡터화되며 KSS·Kiwi의 일반 문단 처리에는 들어가지 않는다.
    """
    table_id = str(block.get("table_id") or "")
    if not table_id:
        raise ValueError("PDF 표 fallback 블록에 table_id가 없습니다")
    lines = [
        compact_text(line)
        for line in str(block.get("retrieval_text") or "").splitlines()
        if compact_text(line)
    ]
    if not lines:
        raise ValueError("PDF 표 fallback 블록의 복구 텍스트가 비어 있습니다")
    return build_pdf_table_formats([[line] for line in lines], table_id)


def _text_boundary(
    block: dict[str, Any],
    file_type: str,
) -> tuple[str, str, list[str]]:
    """다음 KSS 단계가 절대 넘으면 안 되는 PDF page/HWP paragraph 경계를 만든다."""
    source_id = str(block["source_id"])
    quality_flags = list(block.get("quality_flags") or [])
    if file_type == PDF_FILE_TYPE:
        page = block.get("page")
        if not isinstance(page, int) or page < 1:
            raise ValueError("PDF 텍스트 블록에 1-based page가 없습니다")
        return "pdf_page", f"{source_id}:page:{page:04d}", quality_flags

    section_idx = block.get("section_idx")
    para_idx = block.get("para_idx")
    if para_idx is None:
        quality_flags.append("missing_hwp_para_idx_fallback_block_id")
        return "hwp_paragraph", str(block["block_id"]), quality_flags
    section_label = "none" if section_idx is None else str(section_idx)
    return (
        "hwp_paragraph",
        f"{source_id}:section:{section_label}:paragraph:{para_idx}",
        quality_flags,
    )


def _build_advanced_block(
    block: dict[str, Any],
    *,
    file_type: str,
    table_formats: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """중복 표시 문자열을 제거하고 유형별 전용 내용 필드를 만든다."""
    source_schema_version = str(block.get("schema_version") or "")
    record = {
        key: value
        for key, value in block.items()
        if key
        not in {
            "schema_version",
            "display_content",
            "retrieval_text",
            "render_mode",
        }
    }
    content_type = _content_type(block)
    indexable = str(block.get("index_policy")) in INDEXABLE_POLICIES
    text: str | None = None
    table_html: str | None = None
    table_markdown: str | None = None
    image_uri: str | None = None
    vectorize_field: str | None = None
    kss_eligible = False
    bm25_eligible = False
    dense_eligible = False
    boundary_type: str | None = None
    boundary_id: str | None = None
    quality_flags = list(block.get("quality_flags") or [])

    if content_type == "table":
        table_id = str(block.get("table_id") or "")
        if table_id not in table_formats:
            raise ValueError(f"표 이중 표현이 없습니다: {table_id}")
        if block.get("index_reason") == PDF_TABLE_TEXT_FALLBACK_REASON:
            formats = _pdf_fallback_table_formats(block)
        else:
            formats = table_formats[table_id]
        table_html = formats["table_html"]
        table_markdown = formats["table_markdown"]
        dense_eligible = indexable and bool(compact_text(table_markdown))
        vectorize_field = "table_markdown" if dense_eligible else None
    elif content_type == "image":
        picture_id = str(block.get("picture_id") or "")
        if not picture_id:
            raise ValueError("이미지 블록에 picture_id가 없습니다")
        image_uri = f"image://{picture_id}"
    else:
        text = str(block.get("retrieval_text") or "")
        dense_eligible = indexable and bool(compact_text(text))
        kss_eligible = dense_eligible
        bm25_eligible = dense_eligible
        vectorize_field = "text" if dense_eligible else None
        if dense_eligible:
            boundary_type, boundary_id, quality_flags = _text_boundary(
                block,
                file_type,
            )

    return {
        "schema_version": ADVANCED_SCHEMA_VERSION,
        "source_schema_version": source_schema_version,
        **record,
        "content_type": content_type,
        "source_render_mode": block.get("render_mode"),
        "render_mode": "dual_html_gfm" if content_type == "table" else None,
        "format_version": TABLE_FORMAT_VERSION if content_type == "table" else None,
        "text": text,
        "table_html": table_html,
        "table_markdown": table_markdown,
        "image_uri": image_uri,
        "vectorize_field": vectorize_field,
        "dense_eligible": dense_eligible,
        "kss_eligible": kss_eligible,
        "bm25_eligible": bm25_eligible,
        "kss_boundary_type": boundary_type,
        "kss_boundary_id": boundary_id,
        "embedding_prefix_included": False,
        "quality_flags": quality_flags,
    }


def build_advanced_result(
    manifest_document: dict[str, Any],
    base_result: PreprocessingResult,
    table_formats: dict[str, dict[str, str]],
) -> AdvancedPreprocessingResult:
    """기존 구조 결과를 Advanced 전용 content/metadata 계약으로 변환한다."""
    source_id = str(manifest_document.get("source_id") or "")
    if source_id != str(base_result.document.get("source_id") or ""):
        raise ValueError("manifest와 전처리 결과의 source_id가 다릅니다")
    file_type = str(manifest_document.get("file_type") or "").casefold()

    blocks = tuple(
        _build_advanced_block(
            block,
            file_type=file_type,
            table_formats=table_formats,
        )
        for block in base_result.blocks
    )
    blocks_by_table_id = {
        str(block["table_id"]): block
        for block in blocks
        if block.get("content_type") == "table"
    }

    # 표 manifest는 구조 메타데이터만 유지한다. HTML·Markdown 원문은
    # blocks 파일에 한 번만 저장해 같은 대용량 문자열이 중복되지 않게 한다.
    tables: list[dict[str, Any]] = []
    for table in base_result.tables:
        table_id = str(table["table_id"])
        storage_block = blocks_by_table_id.get(table_id)
        is_vectorized = bool(storage_block and storage_block["dense_eligible"])
        table_record = {
            key: value
            for key, value in table.items()
            if key not in {"schema_version", "render_mode"}
        }
        tables.append(
            {
                "schema_version": ADVANCED_SCHEMA_VERSION,
                "source_schema_version": table.get("schema_version"),
                **table_record,
                "source_render_mode": table.get("render_mode"),
                "render_mode": "dual_html_gfm",
                "format_version": TABLE_FORMAT_VERSION,
                "format_storage_block_id": (
                    storage_block["block_id"]
                    if storage_block
                    else table.get("parent_block_id")
                ),
                "vectorize_field": "table_markdown" if is_vectorized else None,
                "dense_eligible": is_vectorized,
                "kss_eligible": False,
                "bm25_eligible": False,
                "embedding_prefix_included": False,
            }
        )

    images: list[dict[str, Any]] = []
    for image in base_result.images:
        picture_id = str(image["picture_id"])
        image_record = {
            key: value
            for key, value in image.items()
            if key
            not in {
                "schema_version",
                "original_ir_uri",
                "data",
                "base64",
                "payload",
                "image_bytes",
                "binary_payload",
            }
        }
        images.append(
            {
                "schema_version": ADVANCED_SCHEMA_VERSION,
                "source_schema_version": image.get("schema_version"),
                **image_record,
                "image_uri": f"image://{picture_id}",
                "dense_eligible": False,
                "kss_eligible": False,
                "bm25_eligible": False,
                "embedding_prefix_included": False,
            }
        )

    manifest_quality_flags = set(manifest_document.get("quality_flags") or [])
    base_quality_flags = set(base_result.document.get("quality_flags") or [])
    document = {
        **manifest_document,
        "schema_version": ADVANCED_SCHEMA_VERSION,
        "source_schema_version": manifest_document.get("schema_version"),
        "base_preprocessing_schema_version": base_result.document.get("schema_version"),
        "parser": base_result.document.get("parser"),
        "page_count": base_result.document.get("page_count"),
        "section_count": base_result.document.get("section_count"),
        "paragraph_count": base_result.document.get("paragraph_count"),
        "pdf_table_error_count": base_result.document.get("pdf_table_error_count", 0),
        "pdf_word_extraction_error_count": base_result.document.get(
            "pdf_word_extraction_error_count", 0
        ),
        "pdf_fallback_text_error_count": base_result.document.get(
            "pdf_fallback_text_error_count", 0
        ),
        "pdf_table_text_fallback_count": base_result.document.get(
            "pdf_table_text_fallback_count", 0
        ),
        "pdf_table_text_fallback_page_count": base_result.document.get(
            "pdf_table_text_fallback_page_count", 0
        ),
        "block_count": len(blocks),
        "text_block_count": sum(block["content_type"] == "text" for block in blocks),
        "table_block_count": sum(block["content_type"] == "table" for block in blocks),
        "image_block_count": sum(block["content_type"] == "image" for block in blocks),
        "dense_eligible_block_count": sum(
            bool(block["dense_eligible"]) for block in blocks
        ),
        "kss_eligible_block_count": sum(
            bool(block["kss_eligible"]) for block in blocks
        ),
        "bm25_eligible_block_count": sum(
            bool(block["bm25_eligible"]) for block in blocks
        ),
        "table_count": len(tables),
        "picture_count": len(images),
        "table_format_version": TABLE_FORMAT_VERSION,
        "table_format_storage": "blocks_only_no_duplicate_content",
        "embedding_prefix_policy": "metadata_only_not_in_vector_text",
        "kss_status": "not_applied_pre_chunking_contract_only",
        "kiwi_status": "not_applied_pre_chunking_contract_only",
        "image_storage": "image_uri_only_no_payload",
        "quality_flags": sorted(manifest_quality_flags | base_quality_flags),
    }

    result = AdvancedPreprocessingResult(
        document=document,
        blocks=blocks,
        tables=tuple(tables),
        images=tuple(images),
    )
    validate_advanced_result(result)
    return result


def validate_advanced_result(result: AdvancedPreprocessingResult) -> None:
    """Advanced 구조 출력의 위치·표·이미지·벡터화 불변식을 검사한다."""
    document = result.document
    source_id = str(document["source_id"])
    file_type = str(document["file_type"]).casefold()
    block_ids = [str(block["block_id"]) for block in result.blocks]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("Advanced block_id가 중복됐습니다")
    if [block["block_order"] for block in result.blocks] != list(
        range(1, len(result.blocks) + 1)
    ):
        raise ValueError("Advanced block_order가 1부터 연속되지 않습니다")

    fallback_blocks = [
        block
        for block in result.blocks
        if block.get("index_reason") == PDF_TABLE_TEXT_FALLBACK_REASON
    ]
    if document.get("pdf_table_text_fallback_count", 0) != len(fallback_blocks):
        raise ValueError("PDF 표 fallback 문서 통계와 복구 블록 수가 다릅니다")
    fallback_pages = {
        block.get("page") for block in fallback_blocks if block.get("page") is not None
    }
    if document.get("pdf_table_text_fallback_page_count", 0) != len(fallback_pages):
        raise ValueError("PDF 표 fallback 페이지 통계와 복구 블록 위치가 다릅니다")

    for block in result.blocks:
        if block.get("source_id") != source_id:
            raise ValueError("Advanced 블록 source_id가 문서와 다릅니다")
        if block.get("embedding_prefix_included") is not False:
            raise ValueError("벡터화 본문에 메타데이터 prefix가 포함됐습니다")
        content_type = block.get("content_type")
        if file_type in HWP_FILE_TYPES and block.get("page") is not None:
            raise ValueError("HWP/HWPX에 추측한 page가 포함됐습니다")
        if file_type == PDF_FILE_TYPE:
            page = block.get("page")
            if not isinstance(page, int) or page < 1:
                raise ValueError("PDF 블록에 유효한 1-based page가 없습니다")

        if content_type == "table":
            if not str(block.get("table_html") or "").lstrip().startswith("<table"):
                raise ValueError("표 HTML이 없거나 <table>로 시작하지 않습니다")
            if not compact_text(str(block.get("table_markdown") or "")):
                raise ValueError("표 Markdown이 비어 있습니다")
            if block.get("kss_eligible") or block.get("bm25_eligible"):
                raise ValueError("표는 KSS와 1차 BM25 대상이 아니어야 합니다")
            expected_field = "table_markdown" if block.get("dense_eligible") else None
            if block.get("vectorize_field") != expected_field:
                raise ValueError("표 벡터화 필드는 table_markdown이어야 합니다")
            if block.get("text") is not None or block.get("image_uri") is not None:
                raise ValueError("표 블록에 다른 유형의 내용 필드가 채워졌습니다")
        elif content_type == "image":
            expected_uri = f"image://{block['picture_id']}"
            if block.get("image_uri") != expected_uri:
                raise ValueError("이미지 URI가 picture_id와 다릅니다")
            if any(
                block.get(field)
                for field in ("dense_eligible", "kss_eligible", "bm25_eligible")
            ):
                raise ValueError("이미지는 색인 대상이 아니어야 합니다")
        elif content_type == "text":
            if block.get("dense_eligible"):
                if block.get("vectorize_field") != "text":
                    raise ValueError("일반 텍스트 벡터화 필드는 text여야 합니다")
                if not block.get("kss_boundary_type") or not block.get(
                    "kss_boundary_id"
                ):
                    raise ValueError("KSS 대상 텍스트에 위치 경계가 없습니다")
                if not block.get("kss_eligible") or not block.get("bm25_eligible"):
                    raise ValueError("검색용 일반 텍스트의 후속 처리 플래그가 없습니다")
        else:
            raise ValueError(f"알 수 없는 Advanced content_type입니다: {content_type}")

    if has_forbidden_image_payload(
        [result.document, result.blocks, result.tables, result.images]
    ):
        raise ValueError("Advanced 결과에 이미지 payload 또는 data URI가 있습니다")


def prepare_advanced_document(
    source: SourceDocument,
    manifest_document: dict[str, Any],
    *,
    analysis_source: VerifiedAnalysisSource | None = None,
    rhwp_module: Any | None = None,
    pdfplumber_module: Any | None = None,
) -> AdvancedPreprocessingResult:
    """실제 원본 하나를 파싱해 Advanced 구조 결과를 반환한다."""
    file_type = source.file_type.casefold()
    if file_type in HWP_FILE_TYPES:
        base_result, table_formats = _hwp_base_and_formats(
            source,
            analysis_source=analysis_source,
            rhwp_module=rhwp_module,
        )
    elif file_type == PDF_FILE_TYPE:
        base_result, table_formats = _pdf_base_and_formats(
            source,
            analysis_source=analysis_source,
            pdfplumber_module=pdfplumber_module,
        )
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {source.file_type}")
    return build_advanced_result(
        manifest_document,
        base_result,
        table_formats,
    )


__all__ = [
    "ADVANCED_SCHEMA_VERSION",
    "AdvancedPreprocessingResult",
    "build_advanced_result",
    "build_sources_from_manifest",
    "load_document_manifest",
    "preflight_required_parsers",
    "prepare_advanced_document",
    "validate_advanced_result",
]
