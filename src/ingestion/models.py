from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class DocumentManifest:
    document_id: str
    difficulty: str
    organization: str
    title: str
    csv_filename: str
    pdf_path: str
    filename: str
    file_size: int
    sha256: str
    match_score: float
    target_accuracy: str
    complexity: str

    def resolved_pdf_path(self) -> Path:
        """Resolve portable paths and manifests created in another checkout."""
        configured = Path(self.pdf_path)
        if not configured.is_absolute():
            return PROJECT_ROOT / configured
        if configured.exists():
            return configured
        candidates = list((PROJECT_ROOT / "data" / "raw").glob(f"**/{self.filename}"))
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(f"Could not resolve PDF for {self.document_id}: {self.filename}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TextBlock:
    text: str
    bbox: tuple[float, float, float, float]
    block_no: int
    content_type: str = "paragraph"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PageRecord:
    document_id: str
    document_title: str
    page: int
    pdf_page_index: int
    text: str
    raw_text_chars: int
    image_count: int
    table_count: int
    ocr_required: bool
    headings: list[str] = field(default_factory=list)
    requirement_ids: list[str] = field(default_factory=list)
    blocks: list[TextBlock] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["blocks"] = [block.to_dict() for block in self.blocks]
        return result


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    section_path: list[str]
    requirement_ids: list[str]
    content_type: str
    text: str
    token_count: int
    ocr_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
