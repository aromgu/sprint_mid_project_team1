from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.chunking.structured_chunker import chunk_pages
from src.ingestion.manifest import build_manifest, write_manifest
from src.ingestion.models import PageRecord
from src.ingestion.ocr import load_ocr_records
from src.ingestion.pdf_parser import parse_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]

    manifests = build_manifest(
        resolve_path(paths["csv"]),
        resolve_path(paths["pdf_dir"]),
        minimum_match_score=config["manifest"]["minimum_match_score"],
    )
    write_manifest(manifests, resolve_path(paths["manifest"]))

    ocr_records = load_ocr_records(resolve_path(paths["ocr_results"]))
    ocr_by_page = {
        (record["document_id"], int(record["page"])): record for record in ocr_records
    }

    all_pages = []
    all_chunks = []
    diagnostics = []
    for manifest in manifests:
        pages, diagnostic = parse_document(manifest, **config["parser"])
        chunks = chunk_pages(pages, **config["chunking"])
        document_ocr = [
            record for record in ocr_records if record["document_id"] == manifest.document_id
        ]
        for record in document_ocr:
            ocr_page = PageRecord(
                document_id=manifest.document_id,
                document_title=manifest.title,
                page=int(record["page"]),
                pdf_page_index=int(record["page"]) - 1,
                text=record["text"],
                raw_text_chars=len(record["text"]),
                image_count=1,
                table_count=0,
                ocr_required=False,
            )
            ocr_chunks = chunk_pages([ocr_page], **config["chunking"])
            for sequence, chunk in enumerate(ocr_chunks, 1):
                chunk.chunk_id = (
                    f"{manifest.document_id}_p{int(record['page']):03d}_ocr_{sequence:02d}"
                )
                chunk.content_type = f"ocr_{record['content_kind']}"
                chunk.section_path = ["OCR", record["content_kind"]]
                chunk.ocr_applied = True
            chunks.extend(ocr_chunks)

        for page in pages:
            page_payload = page.to_dict()
            ocr_record = ocr_by_page.get((page.document_id, page.page))
            if ocr_record:
                page_payload.update(
                    {
                        "ocr_text": ocr_record["text"],
                        "ocr_applied": True,
                        "ocr_engine": ocr_record["ocr_engine"],
                        "multimodal_review_required": ocr_record[
                            "multimodal_review_required"
                        ],
                    }
                )
            else:
                page_payload["ocr_applied"] = False
            all_pages.append(page_payload)
        all_chunks.extend(chunk.to_dict() for chunk in chunks)
        diagnostic["chunk_count"] = len(chunks)
        diagnostic["average_chunk_tokens"] = round(
            sum(chunk.token_count for chunk in chunks) / len(chunks), 2
        ) if chunks else 0
        diagnostic["ocr_applied_pages"] = [int(record["page"]) for record in document_ocr]
        diagnostics.append(diagnostic)
        print(
            f"{manifest.document_id}: pages={len(pages)}, chunks={len(chunks)}, "
            f"status={diagnostic['status']}"
        )

    write_jsonl(all_pages, resolve_path(paths["pages"]))
    write_jsonl(all_chunks, resolve_path(paths["chunks"]))
    diagnostic_path = resolve_path(paths["diagnostics"])
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "document_count": len(manifests),
                "total_pages": len(all_pages),
                "total_chunks": len(all_chunks),
                "documents": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "documents": len(manifests),
        "pages": len(all_pages),
        "chunks": len(all_chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest, diagnose PDFs, parse pages, and create chunks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ingestion.yaml",
    )
    args = parser.parse_args()
    summary = run(args.config.resolve())
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
