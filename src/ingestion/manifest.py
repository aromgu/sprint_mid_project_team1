from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from src.ingestion.models import DocumentManifest


DIFFICULTY_MAP = {"상": "high", "중": "medium", "하": "low"}


def normalize_filename(value: str) -> str:
    """Normalize punctuation and spacing while retaining Korean and ASCII text."""
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^0-9a-z가-힣]+", "", value)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _difficulty(value: str) -> str:
    for label, normalized in DIFFICULTY_MAP.items():
        if label in value:
            return normalized
    raise ValueError(f"Unknown difficulty label: {value!r}")


def _similarity(csv_filename: str, actual_filename: str) -> float:
    expected = normalize_filename(Path(csv_filename).stem)
    actual = normalize_filename(Path(actual_filename).stem)
    if expected == actual:
        return 1.0
    return SequenceMatcher(None, expected, actual).ratio()


def build_manifest(
    csv_path: Path,
    pdf_dir: Path,
    minimum_match_score: float = 0.72,
) -> list[DocumentManifest]:
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    manifests: list[DocumentManifest] = []
    used_paths: set[Path] = set()
    for row in rows:
        expected = row["파일명"].strip()
        ranked = sorted(
            ((_similarity(expected, path.name), path) for path in pdf_files if path not in used_paths),
            reverse=True,
            key=lambda item: item[0],
        )
        if not ranked or ranked[0][0] < minimum_match_score:
            best = ranked[0] if ranked else (0.0, None)
            raise FileNotFoundError(
                f"Could not match {expected!r}; best={best[1]}, score={best[0]:.3f}"
            )
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.03:
            raise ValueError(
                f"Ambiguous PDF match for {expected!r}: "
                f"{ranked[0][1].name!r} ({ranked[0][0]:.3f}) and "
                f"{ranked[1][1].name!r} ({ranked[1][0]:.3f})"
            )

        score, path = ranked[0]
        used_paths.add(path)
        manifests.append(
            DocumentManifest(
                document_id=row["순번"].strip(),
                difficulty=_difficulty(row["난이도"]),
                organization=row["발주기관"].strip(),
                title=row["사업명"].strip(),
                csv_filename=expected,
                pdf_path=str(path.resolve()),
                filename=path.name,
                file_size=path.stat().st_size,
                sha256=sha256_file(path),
                match_score=round(score, 6),
                target_accuracy=row["목표정확도"].strip(),
                complexity=row["복잡도"].strip(),
            )
        )

    ids = [item.document_id for item in manifests]
    if len(ids) != 9 or len(set(ids)) != 9:
        raise ValueError(f"Expected exactly 9 unique documents, got {ids}")
    return manifests


def write_manifest(manifests: list[DocumentManifest], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "document_count": len(manifests),
        "documents": [manifest.to_dict() for manifest in manifests],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

