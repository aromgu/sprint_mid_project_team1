"""임베딩 전 사업 메타데이터 보강 계약을 검증한다."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from src.chunking.enrich_metadata import (
    apply_correction,
    enrich_chunk_metadata,
    sha256_file,
)


def write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    """테스트용 gzip JSONL을 저장한다."""

    with gzip.open(path, "wt", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """테스트용 JSONL을 저장한다."""

    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """테스트용 CSV를 저장한다."""

    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_inputs(tmp_path: Path) -> dict[str, Path]:
    """교정값·수정 요약·중복 별칭이 모두 있는 최소 입력을 만든다."""

    paths = {
        "chunks": tmp_path / "chunks.jsonl.gz",
        "metadata": tmp_path / "metadata.jsonl",
        "corrections": tmp_path / "corrections.csv",
        "reviews": tmp_path / "reviews.csv",
        "aliases": tmp_path / "aliases.csv",
        "output": tmp_path / "enriched.jsonl.gz",
        "report": tmp_path / "report.json",
    }
    write_gzip_jsonl(
        paths["chunks"],
        [
            {
                "chunk_id": "source-1:C000001",
                "source_id": "source-1",
                "document_id": "source-1",
                "source_filename": "대표.hwp",
                "retrieval_text": "[문서] 대표.hwp\n\n본문",
                "token_count": 10,
                "filename_aliases": [],
            },
            {
                "chunk_id": "source-1:C000002",
                "source_id": "source-1",
                "document_id": "source-1",
                "source_filename": "대표.hwp",
                "retrieval_text": "[문서] 대표.hwp\n\n다음 본문",
                "token_count": 12,
                "filename_aliases": [],
            },
        ],
    )
    write_jsonl(
        paths["metadata"],
        [
            {
                "source_filename": "대표.hwp",
                "source_row": 1,
                "project_name": "원래 사업명",
                "issuer": "발주기관",
                "notice_number": None,
                "project_amount_won": 100,
                "project_amount_status": "usable",
                "project_summary": "- 잘못된 요약",
                "metadata_quality_flags": ["missing_notice_number"],
                "validation_status": "original_selected",
            }
        ],
    )
    write_csv(
        paths["corrections"],
        [
            "source_filename",
            "notice_number",
            "notice_round",
            "published_at",
            "bid_start_at",
            "bid_end_at",
            "project_amount_won",
            "correction_source",
            "verification_note",
            "verified_by",
            "project_amount_status",
            "add_quality_flag",
        ],
        [
            {
                "source_filename": "대표.hwp",
                "notice_number": "20260001",
                "correction_source": "원문",
                "verification_note": "공고번호 확인",
                "verified_by": "검토자",
            }
        ],
    )
    write_csv(
        paths["reviews"],
        [
            "source_filename",
            "project_summary",
            "manual_decision",
            "corrected_summary",
            "reviewer",
            "review_note",
        ],
        [
            {
                "source_filename": "대표.hwp",
                "project_summary": "- 잘못된 요약",
                "manual_decision": "revise",
                "corrected_summary": "- 원문 근거 수정 요약",
                "reviewer": "AI 원문대조",
                "review_note": "범위 수정",
            }
        ],
    )
    write_csv(
        paths["aliases"],
        [
            "source_id",
            "canonical_filename",
            "alias_filename",
            "canonical_selection_source",
            "canonical_selection_reason",
        ],
        [
            {
                "source_id": "source-1",
                "canonical_filename": "대표.hwp",
                "alias_filename": "중복본.hwp",
                "canonical_selection_source": "team_policy",
                "canonical_selection_reason": "preferred",
            }
        ],
    )
    return paths


def test_enrichment_adds_filters_reviewed_summary_and_alias(tmp_path: Path) -> None:
    """모든 청크에 검색 필드·검토 요약·중복 별칭을 동일하게 붙인다."""

    paths = make_inputs(tmp_path)
    report = enrich_chunk_metadata(
        chunks_path=paths["chunks"],
        metadata_path=paths["metadata"],
        corrections_path=paths["corrections"],
        summary_review_path=paths["reviews"],
        duplicate_alias_path=paths["aliases"],
        output_path=paths["output"],
        report_path=paths["report"],
    )

    with gzip.open(paths["output"], "rt", encoding="utf-8") as file_obj:
        rows = [json.loads(line) for line in file_obj]

    assert len(rows) == 2
    assert rows[0]["notice_number"] == "20260001"
    assert rows[0]["project_summary"] == "- 원문 근거 수정 요약"
    assert rows[0]["project_summary_review_status"] == "ai_source_review_revise"
    assert rows[0]["filename_aliases"] == ["중복본.hwp"]
    assert report.chunk_count == 2
    assert report.corrected_business_document_count == 1
    assert report.project_summary_revised_document_count == 1


def test_enrichment_output_is_deterministic(tmp_path: Path) -> None:
    """같은 입력의 gzip 결과가 실행 시각과 무관하게 같은 SHA를 갖는다."""

    paths = make_inputs(tmp_path)
    second_output = tmp_path / "enriched_second.jsonl.gz"
    second_report = tmp_path / "report_second.json"
    kwargs = {
        "chunks_path": paths["chunks"],
        "metadata_path": paths["metadata"],
        "corrections_path": paths["corrections"],
        "summary_review_path": paths["reviews"],
        "duplicate_alias_path": paths["aliases"],
    }
    enrich_chunk_metadata(
        **kwargs,
        output_path=paths["output"],
        report_path=paths["report"],
    )
    enrich_chunk_metadata(
        **kwargs,
        output_path=second_output,
        report_path=second_report,
    )

    assert sha256_file(paths["output"]) == sha256_file(second_output)


def test_enrichment_rejects_unmatched_document(tmp_path: Path) -> None:
    """메타데이터가 없는 문서는 조용히 누락시키지 않고 즉시 실패한다."""

    paths = make_inputs(tmp_path)
    with gzip.open(paths["chunks"], "wt", encoding="utf-8") as file_obj:
        file_obj.write(
            json.dumps(
                {
                    "chunk_id": "missing:C000001",
                    "source_id": "missing",
                    "document_id": "missing",
                    "source_filename": "없음.hwp",
                    "retrieval_text": "본문",
                    "token_count": 1,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="사업 메타데이터를 찾지 못했습니다"):
        enrich_chunk_metadata(
            chunks_path=paths["chunks"],
            metadata_path=paths["metadata"],
            corrections_path=paths["corrections"],
            summary_review_path=paths["reviews"],
            duplicate_alias_path=paths["aliases"],
            output_path=paths["output"],
            report_path=paths["report"],
        )


def test_correction_accepts_none_from_short_csv_row() -> None:
    """실제 CSV의 생략된 마지막 셀이 None이어도 빈 교정값으로 처리한다."""

    result = apply_correction(
        {
            "source_filename": "대표.hwp",
            "metadata_quality_flags": [],
        },
        {
            "notice_number": "20260001",
            "project_amount_status": None,
            "add_quality_flag": None,
        },
    )

    assert result["notice_number"] == "20260001"
    assert result["metadata_quality_flags"] == []
