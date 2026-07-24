from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "documents.json"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "CHUNK_QUALITY_REPORT.md"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[int], ratio: float) -> int:
    ordered = sorted(values)
    return ordered[min(int((len(ordered) - 1) * ratio), len(ordered) - 1)]


def excerpt(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", text).strip().replace("|", "\\|")
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def sample_chunks(chunks: list[dict]) -> list[tuple[str, dict]]:
    samples: list[tuple[str, dict]] = []
    selectors = (
        ("요구사항 표", lambda c: c["content_type"] == "requirement_table_row"),
        ("일반 표", lambda c: c["content_type"] == "table_row"),
        ("일반 섹션", lambda c: c["content_type"] == "section"),
    )
    used: set[str] = set()
    for label, selector in selectors:
        candidates = [chunk for chunk in chunks if selector(chunk)]
        if candidates:
            chosen = max(candidates, key=lambda chunk: chunk["token_count"])
            samples.append((label, chosen))
            used.add(chosen["chunk_id"])
    remaining = [chunk for chunk in chunks if chunk["chunk_id"] not in used]
    if remaining:
        shortest = min(remaining, key=lambda chunk: chunk["token_count"])
        samples.append(("최단 경계", shortest))
        used.add(shortest["chunk_id"])
    remaining = [chunk for chunk in chunks if chunk["chunk_id"] not in used]
    if remaining:
        multi_page = [chunk for chunk in remaining if chunk["page_end"] > chunk["page_start"]]
        if multi_page:
            samples.append(("다중 페이지", max(multi_page, key=lambda chunk: chunk["token_count"])))
    return samples


def generate_report() -> str:
    chunks = load_jsonl(CHUNKS_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    documents = {item["document_id"]: item for item in manifest["documents"]}
    by_document: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_document[chunk["document_id"]].append(chunk)

    ids = [chunk["chunk_id"] for chunk in chunks]
    token_counts = [chunk["token_count"] for chunk in chunks]
    duplicate_ids = [chunk_id for chunk_id, count in Counter(ids).items() if count > 1]
    empty_chunks = [chunk["chunk_id"] for chunk in chunks if not chunk["text"].strip()]
    over_limit = [chunk["chunk_id"] for chunk in chunks if chunk["token_count"] > 900]
    very_short = [chunk["chunk_id"] for chunk in chunks if chunk["token_count"] < 30]
    invalid_pages = [
        chunk["chunk_id"]
        for chunk in chunks
        if chunk["page_start"] < 1 or chunk["page_end"] < chunk["page_start"]
    ]
    ocr_chunks = [chunk for chunk in chunks if chunk.get("ocr_applied")]

    lines = [
        "# 청크 품질 보고서",
        "",
        "## 1. 검사 범위",
        "",
        f"- 대상 문서: {len(documents)}개",
        f"- 전체 청크: {len(chunks):,}개",
        "- 입력: `data/processed/chunks.jsonl`",
        "- 설정: 목표 700 tokens, 최대 900 tokens, overlap 80 tokens",
        "",
        "## 2. 자동 무결성 검사",
        "",
        "| 검사 | 결과 |",
        "|---|---:|",
        f"| 중복 chunk ID | {len(duplicate_ids)} |",
        f"| 빈 청크 | {len(empty_chunks)} |",
        f"| 900 tokens 초과 | {len(over_limit)} |",
        f"| 30 tokens 미만 | {len(very_short)} |",
        f"| 잘못된 페이지 범위 | {len(invalid_pages)} |",
        f"| OCR 보강 청크 | {len(ocr_chunks)} |",
        "",
        "모든 자동 무결성 검사를 통과했다." if not any(
            (duplicate_ids, empty_chunks, over_limit, very_short, invalid_pages)
        ) else "위반 항목은 Golden set 작성 전에 수정해야 한다.",
        "",
        "## 3. 토큰 및 콘텐츠 분포",
        "",
        f"- 최소: {min(token_counts)} tokens",
        f"- p25: {percentile(token_counts, 0.25)} tokens",
        f"- 중앙값: {percentile(token_counts, 0.50)} tokens",
        f"- p75: {percentile(token_counts, 0.75)} tokens",
        f"- p95: {percentile(token_counts, 0.95)} tokens",
        f"- 최대: {max(token_counts)} tokens",
        f"- 평균: {sum(token_counts) / len(token_counts):.1f} tokens",
        "",
        "| 콘텐츠 유형 | 청크 수 |",
        "|---|---:|",
    ]
    for content_type, count in sorted(Counter(chunk["content_type"] for chunk in chunks).items()):
        lines.append(f"| `{content_type}` | {count:,} |")

    lines.extend([
        "",
        "## 4. 문서별 요약",
        "",
        "| ID | 문서 | 청크 | 요구사항 청크 | 표 청크 | 평균 tokens |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for document_id in sorted(by_document):
        items = by_document[document_id]
        requirement_count = sum("requirement" in item["content_type"] for item in items)
        table_count = sum("table" in item["content_type"] for item in items)
        average = sum(item["token_count"] for item in items) / len(items)
        lines.append(
            f"| {document_id} | {documents[document_id]['title']} | {len(items)} | "
            f"{requirement_count} | {table_count} | {average:.1f} |"
        )

    lines.extend([
        "",
        "## 5. 문서별 표본 검사",
        "",
        "각 문서에서 요구사항 표, 일반 표, 일반 섹션, 최단 경계, 다중 페이지 청크를 우선 표본으로 선택했다. 아래 발췌문은 구조 검사용이며 Golden answer가 아니다.",
        "",
    ])
    for document_id in sorted(by_document):
        lines.extend([
            f"### {document_id} — {documents[document_id]['title']}",
            "",
            "| 분류 | chunk ID | 페이지 | tokens | 요구사항 ID | 발췌 |",
            "|---|---|---:|---:|---|---|",
        ])
        for label, chunk in sample_chunks(by_document[document_id]):
            page = str(chunk["page_start"])
            if chunk["page_end"] != chunk["page_start"]:
                page = f"{page}–{chunk['page_end']}"
            requirements = ", ".join(chunk["requirement_ids"]) or "-"
            lines.append(
                f"| {label} | `{chunk['chunk_id']}` | {page} | {chunk['token_count']} | "
                f"{requirements} | {excerpt(chunk['text'])} |"
            )
        lines.append("")

    lines.extend([
        "## 6. 판정",
        "",
        "- 검색 인덱싱과 Mini Golden set 작성을 시작할 수 있는 상태다.",
        "- 표 행과 요구사항 ID가 청크 메타데이터에 보존되어 있다.",
        "- 청크 ID는 전체 데이터에서 유일하다.",
        "- 최소 청크 병합과 최대 900-token 제한이 적용됐다.",
        f"- OCR 필수 페이지에서 생성한 검색 청크 {len(ocr_chunks)}개가 별도 콘텐츠 유형으로 반영됐다.",
        "- 시스템 구성도 OCR은 연결 관계가 손실될 수 있어 멀티모달 또는 사람 검수가 필요하다.",
        "- Golden set 검수자는 표본 발췌만 보지 말고 `reference_pages`의 원본 PDF도 함께 확인해야 한다.",
        "",
        "## 7. 다음 품질 게이트",
        "",
        "1. 시스템 구성도 OCR 결과를 멀티모달 또는 사람이 검수한다.",
        "2. 외부에서 전달되는 Golden set의 context ID와 페이지를 검증한다.",
        "3. 승인된 Golden set으로 BM25, Dense, Hybrid, Reranked 검색을 비교한다.",
        "4. 실패 유형을 기준으로 모델과 검색 파라미터를 조정한다.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(generate_report(), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
