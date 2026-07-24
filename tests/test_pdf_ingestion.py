from pathlib import Path

import pymupdf

from src.chunking.structured_chunker import chunk_pages
from src.ingestion.models import DocumentManifest
from src.ingestion.models import PageRecord, TextBlock
from src.ingestion.pdf_parser import parse_document
from src.ingestion.pdf_parser import detect_requirement_ids


def make_manifest(path: Path) -> DocumentManifest:
    return DocumentManifest(
        document_id="상_1",
        difficulty="high",
        organization="테스트 기관",
        title="테스트 사업",
        csv_filename=path.name,
        pdf_path=str(path),
        filename=path.name,
        file_size=path.stat().st_size,
        sha256="test",
        match_score=1.0,
        target_accuracy="80%+",
        complexity="test",
    )


def test_parse_and_chunk_preserves_page_and_requirement(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "1. Security Requirements")
    page.insert_text((72, 110), "SER-004 Access control must be recorded and audited.")
    document.save(pdf_path)
    document.close()

    pages, diagnostics = parse_document(make_manifest(pdf_path))
    chunks = chunk_pages(pages, target_tokens=30, max_tokens=100, overlap_tokens=5)

    assert diagnostics["page_count"] == 1
    assert pages[0].page == 1
    assert "SER-004" in pages[0].requirement_ids
    assert chunks
    assert chunks[0].document_id == "상_1"
    assert chunks[0].page_start == 1
    assert "SER-004" in chunks[0].requirement_ids


def test_requirement_id_does_not_match_email_username() -> None:
    assert detect_requirement_ids("slhan23@example.com") == []
    assert detect_requirement_ids("표준 KICS.KO-10.0307을 적용한다") == []
    assert detect_requirement_ids("요구사항 ID SER-004") == ["SER-004"]


def test_chunk_ids_are_unique_and_overlap_does_not_exceed_maximum() -> None:
    text = "SER-001 " + "보안 요구사항을 준수해야 한다. " * 80
    pages = [
        PageRecord("상_1", "테스트", page, page - 1, text, len(text), 0, 0, False)
        for page in range(1, 4)
    ]
    chunks = chunk_pages(pages, target_tokens=100, max_tokens=140, overlap_tokens=20)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.token_count <= 140 for chunk in chunks)


def test_tiny_trailing_unit_is_merged() -> None:
    pages = [
        PageRecord("하_1", "테스트", 1, 0, "본문 " * 100, 300, 0, 0, False),
        PageRecord("하_1", "테스트", 2, 1, "짧음", 2, 0, 0, False),
    ]
    chunks = chunk_pages(pages, target_tokens=80, max_tokens=200, overlap_tokens=10)
    assert all(chunk.token_count >= 30 for chunk in chunks)


def test_single_line_heading_with_fact_is_preserved_in_chunk_text() -> None:
    page = PageRecord(
        "하_2", "테스트", 4, 3,
        "다. 무상유지보수기간 : 사업종료일로부터 12개월\n라. 사업예산 : 11,270,000,000원",
        70, 0, 0, False,
        blocks=[
            TextBlock("다. 무상유지보수기간 : 사업종료일로부터 12개월", (0, 0, 1, 1), 1),
            TextBlock("라. 사업예산 : 11,270,000,000원", (0, 1, 1, 2), 2),
        ],
    )

    chunks = chunk_pages([page], target_tokens=30, max_tokens=100, overlap_tokens=5)

    combined = "\n".join(chunk.text for chunk in chunks)
    assert "사업종료일로부터 12개월" in combined
    assert "11,270,000,000원" in combined
