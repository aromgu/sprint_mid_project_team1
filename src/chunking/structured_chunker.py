from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

from src.ingestion.models import ChunkRecord, PageRecord
from src.ingestion.pdf_parser import HEADING_PATTERNS, clean_line, detect_requirement_ids


@dataclass(slots=True)
class Unit:
    document_id: str
    document_title: str
    page: int
    section: str
    text: str
    requirement_ids: list[str] = field(default_factory=list)
    content_type: str = "paragraph"
    ocr_required: bool = False


class TokenCounter:
    def __init__(self) -> None:
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def tail(self, text: str, token_count: int) -> str:
        tokens = self.encoding.encode(text)
        return self.encoding.decode(tokens[-token_count:]) if token_count > 0 else ""

    def split(self, text: str, max_tokens: int) -> list[str]:
        tokens = self.encoding.encode(text)
        return [self.encoding.decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def _is_heading(line: str) -> bool:
    return len(line) <= 120 and any(pattern.match(line) for pattern in HEADING_PATTERNS)


def pages_to_units(pages: list[PageRecord]) -> list[Unit]:
    units: list[Unit] = []
    current_section = "본문"
    for page in pages:
        source_blocks = page.blocks or []
        paragraphs = [(block.text, block.content_type) for block in source_blocks]
        if not paragraphs:
            paragraphs = [(text, "paragraph") for text in re.split(r"\n\s*\n+", page.text)]
        for paragraph, source_type in paragraphs:
            lines = [clean_line(line) for line in paragraph.splitlines() if clean_line(line)]
            if not lines:
                continue

            if _is_heading(lines[0]):
                current_section = lines[0]
                # A one-line heading can also contain the complete fact, e.g.
                # "라. 사업예산 : 11,270,000,000원". Keep it in searchable
                # chunk text instead of using it only as section metadata.

            text = "\n".join(lines)
            requirement_ids = detect_requirement_ids(text)
            content_type = (
                "requirement_table_row"
                if source_type == "table_row" and requirement_ids
                else source_type
            )
            units.append(
                Unit(
                    document_id=page.document_id,
                    document_title=page.document_title,
                    page=page.page,
                    section=current_section,
                    text=text,
                    requirement_ids=requirement_ids,
                    content_type=content_type,
                    ocr_required=page.ocr_required,
                )
            )
    return units


def _make_chunk(
    sequence: int,
    units: list[Unit],
    text: str,
    token_count: int,
) -> ChunkRecord:
    document_id = units[0].document_id
    requirement_ids = list(dict.fromkeys(req for unit in units for req in unit.requirement_ids))
    unit_types = {unit.content_type for unit in units}
    if "requirement_table_row" in unit_types:
        content_type = "requirement_table_row"
    elif requirement_ids:
        content_type = "requirement"
    elif "table_row" in unit_types:
        content_type = "table_row"
    else:
        content_type = "section"
    requirement_part = requirement_ids[0] if len(requirement_ids) == 1 else ""
    page_part = f"p{min(unit.page for unit in units):03d}"
    suffix = requirement_part or f"c{sequence:04d}"
    return ChunkRecord(
        chunk_id=f"{document_id}_{page_part}_{suffix}",
        document_id=document_id,
        document_title=units[0].document_title,
        page_start=min(unit.page for unit in units),
        page_end=max(unit.page for unit in units),
        section_path=list(dict.fromkeys(unit.section for unit in units)),
        requirement_ids=requirement_ids,
        content_type=content_type,
        text=text.strip(),
        token_count=token_count,
        ocr_applied=False,
    )


def chunk_pages(
    pages: list[PageRecord],
    target_tokens: int = 700,
    max_tokens: int = 900,
    overlap_tokens: int = 80,
) -> list[ChunkRecord]:
    if not 0 <= overlap_tokens < target_tokens <= max_tokens:
        raise ValueError("Expected 0 <= overlap_tokens < target_tokens <= max_tokens")
    counter = TokenCounter()
    units = pages_to_units(pages)
    chunks: list[ChunkRecord] = []
    buffer: list[Unit] = []
    buffer_text = ""

    def flush() -> None:
        nonlocal buffer, buffer_text
        if not buffer or not buffer_text.strip():
            return
        chunks.append(_make_chunk(len(chunks) + 1, buffer, buffer_text, counter.count(buffer_text)))
        overlap = counter.tail(buffer_text, overlap_tokens)
        last = buffer[-1]
        buffer = [last] if overlap else []
        buffer_text = overlap

    for unit in units:
        unit_tokens = counter.count(unit.text)
        if unit_tokens > max_tokens:
            flush()
            pieces = counter.split(unit.text, max_tokens)
            for piece in pieces:
                chunks.append(_make_chunk(len(chunks) + 1, [unit], piece, counter.count(piece)))
            buffer = []
            buffer_text = ""
            continue

        candidate = f"{buffer_text}\n\n{unit.text}".strip()
        section_changed = bool(buffer and buffer[-1].section != unit.section)
        requirement_boundary = bool(buffer and (buffer[-1].requirement_ids or unit.requirement_ids))
        if buffer and (
            counter.count(candidate) > max_tokens
            or (counter.count(buffer_text) >= target_tokens and (section_changed or requirement_boundary))
        ):
            flush()
            candidate = f"{buffer_text}\n\n{unit.text}".strip()
            if counter.count(candidate) > max_tokens:
                buffer = []
                buffer_text = ""
                candidate = unit.text

        buffer.append(unit)
        buffer_text = candidate
        if counter.count(buffer_text) >= target_tokens and unit.requirement_ids:
            flush()

    flush()

    merged: list[ChunkRecord] = []
    for chunk in chunks:
        if chunk.token_count < 30 and merged:
            previous = merged[-1]
            combined_text = f"{previous.text}\n\n{chunk.text}".strip()
            combined_tokens = counter.count(combined_text)
            if combined_tokens <= max_tokens:
                previous.text = combined_text
                previous.token_count = combined_tokens
                previous.page_start = min(previous.page_start, chunk.page_start)
                previous.page_end = max(previous.page_end, chunk.page_end)
                previous.section_path = list(dict.fromkeys(previous.section_path + chunk.section_path))
                previous.requirement_ids = list(
                    dict.fromkeys(previous.requirement_ids + chunk.requirement_ids)
                )
                if chunk.content_type == "requirement_table_row":
                    previous.content_type = "requirement_table_row"
                elif chunk.content_type == "table_row" and previous.content_type == "section":
                    previous.content_type = "table_row"
                continue
        merged.append(chunk)
    chunks = merged

    merged = []
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        if chunk.token_count < 30 and index + 1 < len(chunks):
            following = chunks[index + 1]
            combined_text = f"{chunk.text}\n\n{following.text}".strip()
            combined_tokens = counter.count(combined_text)
            if combined_tokens <= max_tokens:
                following.text = combined_text
                following.token_count = combined_tokens
                following.page_start = min(chunk.page_start, following.page_start)
                following.page_end = max(chunk.page_end, following.page_end)
                following.section_path = list(
                    dict.fromkeys(chunk.section_path + following.section_path)
                )
                following.requirement_ids = list(
                    dict.fromkeys(chunk.requirement_ids + following.requirement_ids)
                )
                if chunk.content_type == "requirement_table_row":
                    following.content_type = "requirement_table_row"
                elif chunk.content_type == "table_row" and following.content_type == "section":
                    following.content_type = "table_row"
                index += 1
                continue
        merged.append(chunk)
        index += 1
    chunks = merged

    seen: dict[str, int] = {}
    for chunk in chunks:
        base_id = chunk.chunk_id
        count = seen.get(base_id, 0) + 1
        seen[base_id] = count
        if count > 1:
            chunk.chunk_id = f"{base_id}_{count}"
    return chunks
