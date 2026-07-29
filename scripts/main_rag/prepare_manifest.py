"""Convert the current JSON document manifest to the Advanced JSONL contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.main_rag.loader.load_documents import sha256_file
from src.main_rag.settings import DEFAULT_CONFIG_PATH, load_settings


def build_manifest(source_manifest: Path, source_dir: Path) -> list[dict[str, object]]:
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    rows = payload.get("documents", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("현재 문서 manifest의 documents가 비어 있습니다")

    advanced: list[dict[str, object]] = []
    for row in rows:
        filename = str(row.get("filename") or row.get("csv_filename") or "")
        source_path = source_dir / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"manifest 원본 PDF가 없습니다: {source_path}")
        digest = sha256_file(source_path)
        expected = str(row.get("sha256") or "")
        if expected and digest != expected:
            raise ValueError(f"원본 SHA-256이 manifest와 다릅니다: {filename}")
        advanced.append(
            {
                "schema_version": "main_advanced_manifest_v1",
                "source_id": digest[:16],
                "document_id": str(row.get("document_id") or digest[:16]),
                "source_sha256": digest,
                "source_relative_path": filename,
                "source_filename": filename,
                "source_file_size_bytes": source_path.stat().st_size,
                "file_type": source_path.suffix.casefold().lstrip("."),
                "duplicate_alias_count": 0,
                "filename_aliases": [],
                "all_source_filenames": [filename],
                "canonical_selection_reason": "current_golden_manifest",
                "metadata_validation_status": "current_manifest",
                "business_metadata": {
                    "project_name": row.get("title"),
                    "issuer": row.get("organization"),
                },
            }
        )
    return advanced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.config)
    output = settings.path("advanced_manifest")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"기존 manifest를 보호합니다: {output} (--overwrite 필요)")
    rows = build_manifest(settings.path("source_manifest"), settings.path("source_dir"))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), "document_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
