"""검증된 사업 메타데이터를 임베딩 직전 청크에 결합한다."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import unicodedata
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, TextIO


BUSINESS_FIELDS = (
    "source_row",
    "notice_number",
    "notice_round",
    "project_name",
    "project_amount_won",
    "project_amount_status",
    "issuer",
    "published_at",
    "bid_start_at",
    "bid_end_at",
    "bid_period_status",
    "project_summary",
    "metadata_quality_flags",
    "validation_status",
    "metadata_correction_fields",
    "metadata_correction_source",
    "metadata_correction_note",
    "metadata_verified_by",
)

CORRECTABLE_FIELDS = (
    "notice_number",
    "notice_round",
    "published_at",
    "bid_start_at",
    "bid_end_at",
    "project_amount_won",
)

MISSING_FLAGS = {
    "notice_number": {"missing_notice_number"},
    "notice_round": {"missing_notice_round"},
    "published_at": {"missing_published_at"},
    "bid_start_at": {"missing_bid_start_at"},
    "bid_end_at": {"missing_bid_end_at"},
    "project_amount_won": {
        "missing_or_unparsed_amount",
        "zero_or_unset_amount",
        "review_low_value_amount",
    },
}


@dataclass(frozen=True)
class EnrichmentReport:
    """메타데이터 보강의 입력 계보와 품질 검사를 기록한다."""

    input_path: str
    input_sha256: str
    output_path: str
    output_sha256: str
    chunk_count: int
    document_count: int
    metadata_matched_document_count: int
    metadata_unmatched_document_count: int
    corrected_business_document_count: int
    project_summary_pass_document_count: int
    project_summary_revised_document_count: int
    documents_with_filename_aliases: int
    filename_alias_count: int
    metadata_source: str
    metadata_source_sha256: str
    corrections_source: str | None
    corrections_source_sha256: str | None
    project_summary_review_source: str
    project_summary_review_source_sha256: str
    duplicate_alias_source: str
    duplicate_alias_source_sha256: str
    metadata_schema_version: str


def normalize_filename(value: str) -> str:
    """macOS NFD 파일명과 CSV NFC 파일명을 같은 값으로 비교한다."""

    return unicodedata.normalize("NFC", value).strip()


def sha256_file(path: Path) -> str:
    """대용량 입력도 메모리에 올리지 않고 SHA-256을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_jsonl(path: Path) -> TextIO:
    """일반 JSONL과 gzip JSONL을 같은 방식으로 연다."""

    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


@contextmanager
def deterministic_gzip_writer(path: Path) -> Iterator[TextIO]:
    """같은 입력에서 같은 SHA가 나오도록 gzip mtime을 0으로 고정한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_file:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_file, mtime=0
        ) as gzip_file:
            with io.TextIOWrapper(
                gzip_file, encoding="utf-8", newline="\n"
            ) as text_file:
                yield text_file


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """UTF-8 BOM을 허용하며 CSV 행을 문자열 사전으로 읽는다."""

    with path.open(encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def load_corrections(path: Path | None) -> dict[str, dict[str, str]]:
    """팀 검토가 끝난 비어 있거나 잘못된 사업 필드의 교정값을 읽는다."""

    if path is None:
        return {}
    corrections: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(path):
        filename = normalize_filename(row["source_filename"])
        if filename in corrections:
            raise ValueError(f"중복 메타데이터 교정 행: {filename}")
        corrections[filename] = row
    return corrections


def apply_correction(
    record: dict[str, Any], correction: dict[str, str]
) -> dict[str, Any]:
    """검토 CSV의 비어 있지 않은 교정값만 원본 메타데이터에 덮어쓴다."""

    result = dict(record)
    corrected_fields: list[str] = []
    flags = set(result.get("metadata_quality_flags") or [])

    for field in CORRECTABLE_FIELDS:
        value = str(correction.get(field) or "").strip()
        if not value:
            continue
        if field == "project_amount_won":
            result[field] = None if value == "__NULL__" else int(value)
        else:
            result[field] = value
        corrected_fields.append(field)
        flags -= MISSING_FLAGS[field]

    if "project_amount_won" in corrected_fields:
        result["project_amount_status"] = str(
            correction.get("project_amount_status") or "usable"
        ).strip()
    if {"bid_start_at", "bid_end_at"} & set(corrected_fields):
        start = result.get("bid_start_at")
        end = result.get("bid_end_at")
        result["bid_period_status"] = (
            "complete" if start and end else "partial" if start or end else "missing"
        )

    added_flag = str(correction.get("add_quality_flag") or "").strip()
    if added_flag:
        flags.add(added_flag)
    result["metadata_quality_flags"] = sorted(flags)
    result["validation_status"] = "team_validated_correction"
    result["metadata_correction_fields"] = corrected_fields
    result["metadata_correction_source"] = str(
        correction.get("correction_source") or ""
    ).strip()
    result["metadata_correction_note"] = str(
        correction.get("verification_note") or ""
    ).strip()
    result["metadata_verified_by"] = str(correction.get("verified_by") or "").strip()
    return result


def load_summary_reviews(path: Path) -> dict[str, dict[str, str]]:
    """원문 대조 결과를 읽고 pass 또는 수정 요약만 허용한다."""

    reviews: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(path):
        filename = normalize_filename(row["source_filename"])
        decision = row.get("manual_decision", "").strip()
        if decision not in {"pass", "revise"}:
            raise ValueError(f"미완료 사업요약 검토: {filename}={decision!r}")
        if decision == "revise" and not row.get("corrected_summary", "").strip():
            raise ValueError(f"수정 사업요약이 비어 있습니다: {filename}")
        if filename in reviews:
            raise ValueError(f"중복 사업요약 검토 행: {filename}")
        reviews[filename] = row
    return reviews


def apply_summary_review(
    record: dict[str, Any], review: dict[str, str]
) -> dict[str, Any]:
    """원문 대조 판정에 따라 기존 요약 또는 수정 요약을 선택한다."""

    reviewed_original = review.get("project_summary", "").strip()
    metadata_original = str(record.get("project_summary") or "").strip()
    if reviewed_original != metadata_original:
        raise ValueError(
            f"검토 대상 요약과 메타데이터 요약이 다릅니다: {record['source_filename']}"
        )

    result = dict(record)
    decision = review["manual_decision"].strip()
    if decision == "revise":
        result["project_summary"] = review["corrected_summary"].strip()
    result["project_summary_review_status"] = f"ai_source_review_{decision}"
    result["project_summary_reviewer"] = review.get("reviewer", "").strip()
    result["project_summary_review_note"] = review.get("review_note", "").strip()
    return result


def load_business_metadata(
    metadata_path: Path,
    corrections_path: Path | None,
    summary_review_path: Path,
) -> tuple[dict[str, dict[str, Any]], set[str], Counter[str]]:
    """파일명별 사업 메타데이터에 교정값과 원문 대조 요약을 적용한다."""

    corrections = load_corrections(corrections_path)
    reviews = load_summary_reviews(summary_review_path)
    records: dict[str, dict[str, Any]] = {}
    applied_corrections: set[str] = set()
    review_decisions: Counter[str] = Counter()

    with open_jsonl(metadata_path) as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            filename = normalize_filename(record["source_filename"])
            if filename in records:
                raise ValueError(f"중복 사업 메타데이터: {filename}")
            if filename in corrections:
                record = apply_correction(record, corrections[filename])
                applied_corrections.add(filename)
            if filename in reviews:
                record = apply_summary_review(record, reviews[filename])
                review_decisions[reviews[filename]["manual_decision"].strip()] += 1
            records[filename] = record

    missing_corrections = set(corrections) - applied_corrections
    if missing_corrections:
        raise ValueError(
            f"적용되지 않은 메타데이터 교정: {sorted(missing_corrections)}"
        )
    missing_reviews = set(reviews) - set(records)
    if missing_reviews:
        raise ValueError(
            f"메타데이터와 연결되지 않은 요약 검토: {sorted(missing_reviews)}"
        )
    return records, applied_corrections, review_decisions


def load_duplicate_aliases(
    path: Path,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """대표 문서 ID별 대표 파일명과 중복 별칭 파일명을 읽는다."""

    canonical_by_source_id: dict[str, str] = {}
    aliases_by_source_id: defaultdict[str, list[str]] = defaultdict(list)
    for row in read_csv_rows(path):
        source_id = row["source_id"].strip()
        canonical = normalize_filename(row["canonical_filename"])
        alias = normalize_filename(row["alias_filename"])
        previous = canonical_by_source_id.setdefault(source_id, canonical)
        if previous != canonical:
            raise ValueError(f"대표 파일명이 충돌합니다: {source_id}")
        if alias not in aliases_by_source_id[source_id]:
            aliases_by_source_id[source_id].append(alias)
    return canonical_by_source_id, dict(aliases_by_source_id)


def enrich_chunk_metadata(
    *,
    chunks_path: Path,
    metadata_path: Path,
    corrections_path: Path | None,
    summary_review_path: Path,
    duplicate_alias_path: Path,
    output_path: Path,
    report_path: Path,
) -> EnrichmentReport:
    """전체 청크에 문서 단위 사업 메타데이터를 붙이고 품질 보고서를 만든다."""

    records, applied_corrections, review_decisions = load_business_metadata(
        metadata_path,
        corrections_path,
        summary_review_path,
    )
    canonical_by_source_id, aliases_by_source_id = load_duplicate_aliases(
        duplicate_alias_path
    )

    matched_documents: set[str] = set()
    unmatched_documents: set[str] = set()
    seen_documents: set[str] = set()
    aliased_documents: set[str] = set()
    chunk_count = 0

    with open_jsonl(chunks_path) as input_file:
        with deterministic_gzip_writer(output_path) as output_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                chunk = json.loads(line)
                source_id = str(chunk.get("source_id") or "")
                source_filename = normalize_filename(
                    str(chunk.get("source_filename") or "")
                )
                if not source_id or not source_filename:
                    raise ValueError(f"출처 필드가 비어 있습니다: line={line_number}")

                seen_documents.add(source_id)
                record = records.get(source_filename)
                if record is None:
                    unmatched_documents.add(source_id)
                    raise ValueError(
                        f"사업 메타데이터를 찾지 못했습니다: {source_filename}"
                    )

                matched_documents.add(source_id)
                aliases = aliases_by_source_id.get(source_id, [])
                expected_canonical = canonical_by_source_id.get(source_id)
                if expected_canonical and expected_canonical != source_filename:
                    raise ValueError(
                        f"중복 대표 파일명이 청크와 다릅니다: {source_filename}"
                    )
                if aliases:
                    aliased_documents.add(source_id)

                for field in BUSINESS_FIELDS:
                    chunk[field] = record.get(field)
                chunk["project_summary_review_status"] = record.get(
                    "project_summary_review_status"
                )
                chunk["project_summary_reviewer"] = record.get(
                    "project_summary_reviewer"
                )
                chunk["project_summary_review_note"] = record.get(
                    "project_summary_review_note"
                )
                chunk["business_metadata_match_rule"] = "canonical_filename"
                chunk["metadata_schema_version"] = "business_metadata_v1"
                chunk["filename_aliases"] = aliases

                output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunk_count += 1

    report = EnrichmentReport(
        input_path=str(chunks_path),
        input_sha256=sha256_file(chunks_path),
        output_path=str(output_path),
        output_sha256=sha256_file(output_path),
        chunk_count=chunk_count,
        document_count=len(seen_documents),
        metadata_matched_document_count=len(matched_documents),
        metadata_unmatched_document_count=len(unmatched_documents),
        corrected_business_document_count=len(applied_corrections),
        project_summary_pass_document_count=review_decisions["pass"],
        project_summary_revised_document_count=review_decisions["revise"],
        documents_with_filename_aliases=len(aliased_documents),
        filename_alias_count=sum(len(value) for value in aliases_by_source_id.values()),
        metadata_source=str(metadata_path),
        metadata_source_sha256=sha256_file(metadata_path),
        corrections_source=str(corrections_path) if corrections_path else None,
        corrections_source_sha256=(
            sha256_file(corrections_path) if corrections_path else None
        ),
        project_summary_review_source=str(summary_review_path),
        project_summary_review_source_sha256=sha256_file(summary_review_path),
        duplicate_alias_source=str(duplicate_alias_path),
        duplicate_alias_source_sha256=sha256_file(duplicate_alias_path),
        metadata_schema_version="business_metadata_v1",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
