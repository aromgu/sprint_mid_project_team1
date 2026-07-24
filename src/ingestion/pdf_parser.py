from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pymupdf

from src.ingestion.models import DocumentManifest, PageRecord, TextBlock


HEADING_PATTERNS = (
    re.compile(r"^제?\s*\d+\s*[장절편]\b.*"),
    re.compile(r"^\d+(?:\.\d+){0,3}[.)]?\s+\S+.*"),
    re.compile(r"^[가-힣]\s*[.)]\s+\S+.*"),
    re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX]+[.)]?\s+\S+.*", re.IGNORECASE),
    re.compile(r"^\[.+\]$"),
)
REQUIREMENT_PATTERN = re.compile(
    r"(?<![A-Z0-9.])(?:[A-Z]{2,8})[-_](?:\d{2,4})(?:[-_]\d{1,3})?\b",
    re.IGNORECASE,
)
WHITESPACE_PATTERN = re.compile(r"[ \t\u00a0]+")
PAGE_NUMBER_PATTERN = re.compile(r"^-?\s*\d{1,4}\s*-?$")


def clean_line(line: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", line).strip()


def detect_headings(text: str) -> list[str]:
    headings: list[str] = []
    for raw_line in text.splitlines():
        line = clean_line(raw_line)
        if not line or len(line) > 120:
            continue
        if any(pattern.match(line) for pattern in HEADING_PATTERNS):
            headings.append(line)
    return list(dict.fromkeys(headings))


def detect_requirement_ids(text: str) -> list[str]:
    return list(dict.fromkeys(match.upper().replace(" ", "-") for match in REQUIREMENT_PATTERN.findall(text)))


def _extract_blocks(page: pymupdf.Page) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for item in page.get_text("blocks", sort=True):
        x0, y0, x1, y1, text, block_no, block_type = item[:7]
        if block_type != 0:
            continue
        cleaned_lines = [clean_line(line) for line in text.splitlines()]
        cleaned = "\n".join(line for line in cleaned_lines if line)
        if cleaned:
            blocks.append(
                TextBlock(
                    text=cleaned,
                    bbox=(round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)),
                    block_no=int(block_no),
                    content_type="paragraph",
                )
            )
    return blocks


def _overlaps_table(block: TextBlock, table_bbox: tuple[float, float, float, float]) -> bool:
    x0 = max(block.bbox[0], table_bbox[0])
    y0 = max(block.bbox[1], table_bbox[1])
    x1 = min(block.bbox[2], table_bbox[2])
    y1 = min(block.bbox[3], table_bbox[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1.0, (block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1]))
    return intersection / area >= 0.5


def _extract_tables(page: pymupdf.Page) -> tuple[list[TextBlock], list[tuple[float, float, float, float]]]:
    table_blocks: list[TextBlock] = []
    table_bboxes: list[tuple[float, float, float, float]] = []
    try:
        tables = page.find_tables().tables
    except Exception:
        return table_blocks, table_bboxes

    block_no = 100_000
    for table_index, table in enumerate(tables):
        bbox = tuple(round(float(value), 2) for value in table.bbox)
        table_bboxes.append(bbox)
        rows = table.extract()
        if not rows:
            continue
        detected_header = getattr(getattr(table, "header", None), "names", None) or []
        header = [clean_line(str(cell or "")) for cell in detected_header]
        for row_index, row in enumerate(rows):
            values = [clean_line(str(cell or "")) for cell in row]
            if not any(values):
                continue
            if len(rows) > 1 and row_index == 0 and header and values == header:
                continue
            if header and any(header):
                cells = [
                    f"{(header[index] if index < len(header) else '') or f'열{index + 1}'}: {value}"
                    for index, value in enumerate(values)
                    if value
                ]
            else:
                cells = [value for value in values if value]
            text = "\n".join(cells)
            table_blocks.append(
                TextBlock(
                    text=text,
                    bbox=bbox,
                    block_no=block_no + table_index * 1000 + row_index,
                    content_type="table_row",
                )
            )
    return table_blocks, table_bboxes


def _margin_lines(page: pymupdf.Page, ratio: float) -> tuple[list[str], list[str]]:
    top: list[str] = []
    bottom: list[str] = []
    height = page.rect.height
    for block in _extract_blocks(page):
        lines = [clean_line(line) for line in block.text.splitlines() if clean_line(line)]
        if block.bbox[1] <= height * ratio:
            top.extend(lines)
        if block.bbox[3] >= height * (1 - ratio):
            bottom.extend(lines)
    return top, bottom


def _repeated_margin_lines(
    document: pymupdf.Document,
    ratio: float,
    minimum_pages: int,
) -> set[str]:
    counts: Counter[str] = Counter()
    for page in document:
        top, bottom = _margin_lines(page, ratio)
        counts.update(set(top + bottom))
    threshold = max(minimum_pages, int(len(document) * 0.4))
    return {line for line, count in counts.items() if count >= threshold and len(line) <= 100}


def parse_document(
    manifest: DocumentManifest,
    min_text_chars: int = 30,
    repeated_margin_ratio: float = 0.15,
    repeated_line_min_pages: int = 3,
) -> tuple[list[PageRecord], dict]:
    path = manifest.resolved_pdf_path()
    document = pymupdf.open(path)
    if document.needs_pass:
        document.close()
        raise ValueError(f"Encrypted PDF requires a password: {path}")

    repeated_lines = _repeated_margin_lines(
        document,
        ratio=repeated_margin_ratio,
        minimum_pages=repeated_line_min_pages,
    )
    records: list[PageRecord] = []
    total_images = 0
    total_tables = 0
    ocr_pages: list[int] = []
    empty_pages: list[int] = []

    for index, page in enumerate(document):
        blocks = _extract_blocks(page)
        table_blocks, table_bboxes = _extract_tables(page)
        blocks = [
            block for block in blocks
            if not any(_overlaps_table(block, table_bbox) for table_bbox in table_bboxes)
        ]
        blocks.extend(table_blocks)
        blocks.sort(key=lambda block: (block.bbox[1], block.bbox[0], block.block_no))
        filtered_blocks: list[TextBlock] = []
        for block in blocks:
            in_margin = (
                block.bbox[1] <= page.rect.height * repeated_margin_ratio
                or block.bbox[3] >= page.rect.height * (1 - repeated_margin_ratio)
            )
            lines = [
                line for line in block.text.splitlines()
                if line not in repeated_lines
                and not (in_margin and PAGE_NUMBER_PATTERN.match(line))
            ]
            text = "\n".join(lines).strip()
            if text:
                filtered_blocks.append(
                    TextBlock(
                        text=text,
                        bbox=block.bbox,
                        block_no=block.block_no,
                        content_type=block.content_type,
                    )
                )

        text = "\n\n".join(block.text for block in filtered_blocks)
        image_count = len(page.get_images(full=True))
        table_count = len(table_bboxes)
        total_images += image_count
        total_tables += table_count
        ocr_required = len(text.strip()) < min_text_chars and image_count > 0
        if ocr_required:
            ocr_pages.append(index + 1)
        if not text.strip():
            empty_pages.append(index + 1)

        records.append(
            PageRecord(
                document_id=manifest.document_id,
                document_title=manifest.title,
                page=index + 1,
                pdf_page_index=index,
                text=text,
                raw_text_chars=len(text),
                image_count=image_count,
                table_count=table_count,
                ocr_required=ocr_required,
                headings=detect_headings(text),
                requirement_ids=detect_requirement_ids(text),
                blocks=filtered_blocks,
            )
        )

    diagnostics = {
        "document_id": manifest.document_id,
        "filename": manifest.filename,
        "page_count": len(records),
        "text_pages": sum(bool(record.text.strip()) for record in records),
        "empty_pages": empty_pages,
        "ocr_required_pages": ocr_pages,
        "total_text_chars": sum(record.raw_text_chars for record in records),
        "total_images": total_images,
        "detected_tables": total_tables,
        "detected_requirement_ids": sum(len(record.requirement_ids) for record in records),
        "repeated_margin_lines_removed": sorted(repeated_lines),
        "status": "needs_ocr" if ocr_pages else "text_extractable",
    }
    document.close()
    return records, diagnostics
