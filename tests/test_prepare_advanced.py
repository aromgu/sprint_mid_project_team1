"""Advanced 구조 전처리의 content/metadata 계약을 검증한다."""

from __future__ import annotations

import hashlib
import json

import pytest

from src.preprocessing.clean_text import PreprocessingResult
from src.preprocessing.prepare_advanced import (
    ADVANCED_SCHEMA_VERSION,
    build_advanced_result,
    build_sources_from_manifest,
)

SOURCE_ID = "0123456789abcdef"
SOURCE_SHA = SOURCE_ID + "0" * 48


def _manifest(file_type: str = "hwp") -> dict[str, object]:
    """사업 메타데이터가 본문에 섞이지 않는지 확인할 최소 문서를 만든다."""
    return {
        "schema_version": "rfp_structured_preprocessing_v2",
        "source_id": SOURCE_ID,
        "document_id": SOURCE_ID,
        "source_sha256": SOURCE_SHA,
        "source_filename": "META_ONLY_SENTINEL_9X.hwp",
        "file_type": file_type,
        "business_metadata": {
            "project_name": "META_ONLY_SENTINEL_9X 사업",
            "notice_number": "20260001",
        },
    }


def _base_result(
    *,
    file_type: str = "hwp",
    para_idx: int | None = 7,
) -> PreprocessingResult:
    """일반 텍스트·표·이미지를 한 개씩 가진 기존 구조 결과를 만든다."""
    page = 2 if file_type == "pdf" else None
    text_block = {
        "schema_version": "rfp_structured_preprocessing_v2",
        "source_id": SOURCE_ID,
        "document_id": SOURCE_ID,
        "block_id": f"{SOURCE_ID}:B000001",
        "block_order": 1,
        "block_type": "text" if file_type == "pdf" else "paragraph",
        "display_content": "과업의 목적입니다.",
        "retrieval_text": "과업의 목적입니다.",
        "index_policy": "index",
        "index_reason": "body_text",
        "scope": "body",
        "section_path": "Ⅰ. 사업 개요",
        "section_idx": None if file_type == "pdf" else 0,
        "para_idx": None if file_type == "pdf" else para_idx,
        "page": page,
        "table_id": None,
        "picture_id": None,
        "quality_flags": [],
    }
    table_block = {
        "schema_version": "rfp_structured_preprocessing_v2",
        "source_id": SOURCE_ID,
        "document_id": SOURCE_ID,
        "block_id": f"{SOURCE_ID}:B000002",
        "block_order": 2,
        "block_type": "table",
        "display_content": "| 구분 | 내용 |",
        "retrieval_text": "구분 | 내용",
        "index_policy": "index",
        "index_reason": "content_table",
        "scope": "body",
        "section_path": "Ⅰ. 사업 개요",
        "section_idx": None if file_type == "pdf" else 0,
        "para_idx": None if file_type == "pdf" else 8,
        "page": page,
        "table_id": f"{SOURCE_ID}:T000001",
        "picture_id": None,
        "quality_flags": [],
    }
    picture_block = {
        "schema_version": "rfp_structured_preprocessing_v2",
        "source_id": SOURCE_ID,
        "document_id": SOURCE_ID,
        "block_id": f"{SOURCE_ID}:B000003",
        "block_order": 3,
        "block_type": "picture",
        "display_content": f"![도식](image://{SOURCE_ID}:I000001)",
        "retrieval_text": "",
        "index_policy": "exclude",
        "index_reason": "image_metadata_only",
        "scope": "body",
        "section_path": "Ⅰ. 사업 개요",
        "section_idx": None if file_type == "pdf" else 0,
        "para_idx": None if file_type == "pdf" else 9,
        "page": page,
        "table_id": None,
        "picture_id": f"{SOURCE_ID}:I000001",
        "quality_flags": [],
    }
    table = {
        "schema_version": "rfp_structured_preprocessing_v2",
        "source_id": SOURCE_ID,
        "document_id": SOURCE_ID,
        "table_id": table_block["table_id"],
        "parent_block_id": table_block["block_id"],
        "is_top_level": True,
        "index_policy": "index",
        "render_mode": "gfm",
        "page": page,
    }
    image = {
        "schema_version": "rfp_structured_preprocessing_v2",
        "source_id": SOURCE_ID,
        "document_id": SOURCE_ID,
        "picture_id": picture_block["picture_id"],
        "parent_block_id": picture_block["block_id"],
        "original_ir_uri": "bin://BIN0001",
        "index_policy": "exclude",
        "page": page,
    }
    return PreprocessingResult(
        document={
            "schema_version": "rfp_structured_preprocessing_v2",
            "source_id": SOURCE_ID,
            "parser": "fixture",
            "page_count": 2 if file_type == "pdf" else 10,
            "section_count": None if file_type == "pdf" else 1,
            "paragraph_count": None if file_type == "pdf" else 10,
        },
        blocks=(text_block, table_block, picture_block),
        tables=(table,),
        images=(image,),
    )


def _formats() -> dict[str, dict[str, str]]:
    """표 HTML과 벡터화용 Markdown fixture를 반환한다."""
    return {
        f"{SOURCE_ID}:T000001": {
            "table_html": (
                f'<table data-table-id="{SOURCE_ID}:T000001">'
                "<tr><th>구분</th><th>내용</th></tr></table>"
            ),
            "table_markdown": "| 구분 | 내용 |\n| --- | --- |",
            "vectorize_field": "table_markdown",
        }
    }


def test_hwp_result_separates_text_table_image_and_metadata() -> None:
    """HWP 본문·표·이미지는 서로 다른 필드와 후속 처리 정책을 가진다."""
    result = build_advanced_result(_manifest(), _base_result(), _formats())
    text, table, image = result.blocks

    assert result.document["schema_version"] == ADVANCED_SCHEMA_VERSION
    assert text["content_type"] == "text"
    assert text["text"] == "과업의 목적입니다."
    assert text["vectorize_field"] == "text"
    assert text["kss_boundary_type"] == "hwp_paragraph"
    assert text["kss_boundary_id"].endswith("section:0:paragraph:7")
    assert text["kss_eligible"] is True
    assert text["bm25_eligible"] is True

    assert table["content_type"] == "table"
    assert table["vectorize_field"] == "table_markdown"
    assert table["source_render_mode"] is None
    assert table["render_mode"] == "dual_html_gfm"
    assert table["format_version"] == "html_gfm_dual_v1"
    assert table["table_html"].startswith("<table")
    assert table["table_markdown"].startswith("| 구분")
    assert table["kss_eligible"] is False
    assert table["bm25_eligible"] is False
    assert "display_content" not in table
    assert "retrieval_text" not in table

    assert image["content_type"] == "image"
    assert image["image_uri"] == f"image://{SOURCE_ID}:I000001"
    assert image["dense_eligible"] is False
    assert "original_ir_uri" not in result.images[0]

    # 파일명과 사업명 sentinel은 document metadata에만 있고 내용 블록에는 없다.
    serialized_blocks = json.dumps(result.blocks, ensure_ascii=False)
    assert "META_ONLY_SENTINEL_9X" not in serialized_blocks
    assert result.document["source_filename"] == "META_ONLY_SENTINEL_9X.hwp"


def test_pdf_text_uses_page_boundary_and_only_pdf_has_page() -> None:
    """PDF KSS 경계는 실제 페이지이며 페이지 번호를 1부터 유지한다."""
    result = build_advanced_result(
        _manifest("pdf"),
        _base_result(file_type="pdf"),
        _formats(),
    )

    text = result.blocks[0]
    assert text["kss_boundary_type"] == "pdf_page"
    assert text["kss_boundary_id"] == f"{SOURCE_ID}:page:0002"
    assert all(block["page"] == 2 for block in result.blocks)


def test_missing_hwp_para_uses_block_id_without_merging() -> None:
    """rhwp가 para_idx를 주지 않으면 그 블록 자체를 독립 경계로 사용한다."""
    result = build_advanced_result(
        _manifest(),
        _base_result(para_idx=None),
        _formats(),
    )

    text = result.blocks[0]
    assert text["kss_boundary_id"] == text["block_id"]
    assert "missing_hwp_para_idx_fallback_block_id" in text["quality_flags"]


def test_missing_table_formats_fails_before_chunking() -> None:
    """HTML이나 Markdown을 만들지 못한 표를 조용히 다음 단계로 넘기지 않는다."""
    with pytest.raises(ValueError, match="표 이중 표현이 없습니다"):
        build_advanced_result(_manifest(), _base_result(), {})


def test_manifest_builds_verified_recovery_source(tmp_path) -> None:
    """원본 HWP와 검증된 복구 HWPX의 ID·SHA 연결을 보존한다."""
    source_dir = tmp_path / "raw"
    recovery_dir = tmp_path / "hwpx"
    source_dir.mkdir()
    recovery_dir.mkdir()
    source_path = source_dir / "원본.hwp"
    recovery_path = recovery_dir / "recovered.hwpx"
    source_path.write_bytes(b"original-hwp")
    recovery_path.write_bytes(b"verified-hwpx")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    recovery_sha = hashlib.sha256(recovery_path.read_bytes()).hexdigest()
    record = {
        "source_id": source_sha[:16],
        "document_id": source_sha[:16],
        "source_sha256": source_sha,
        "source_relative_path": "원본.hwp",
        "source_filename": "원본.hwp",
        "source_file_size_bytes": source_path.stat().st_size,
        "file_type": "hwp",
        "filename_aliases": [],
        "all_source_filenames": ["원본.hwp"],
        "duplicate_alias_count": 0,
        "analysis_source_relative_path": "recovered.hwpx",
        "analysis_source_sha256": recovery_sha,
        "analysis_file_type": "hwpx",
        "recovery_status": "recovered_hwpx",
    }

    sources, analysis_sources, manifests = build_sources_from_manifest(
        [record],
        source_dir=source_dir,
        recovery_dir=recovery_dir,
    )

    assert len(sources) == 1
    assert sources[0].source_path == source_path
    assert analysis_sources[source_sha[:16]].path == recovery_path
    assert analysis_sources[source_sha[:16]].sha256 == recovery_sha
    assert manifests[source_sha[:16]] == record
