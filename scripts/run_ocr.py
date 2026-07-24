from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.ingestion.ocr import TesseractRuntime, ocr_pdf_page
from src.ingestion.models import DocumentManifest
from src.search.service import PROJECT_ROOT, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR selected RFP PDF pages using user-space Tesseract.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "ocr.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(resolve_path(config["paths"]["manifest"]).read_text(encoding="utf-8"))
    pdf_paths = {
        item["document_id"]: DocumentManifest(**item).resolved_pdf_path()
        for item in manifest["documents"]
    }
    runtime_config = config["runtime"]
    runtime = TesseractRuntime(Path(runtime_config["root"]))
    records = []
    for item in config["pages"]:
        text = ocr_pdf_page(
            pdf_paths[item["document_id"]],
            int(item["page"]),
            runtime,
            languages=runtime_config["languages"],
            dpi=int(runtime_config["dpi"]),
            psm=int(runtime_config["psm"]),
        )
        record = {
            **item,
            "text": text,
            "text_chars": len(text),
            "ocr_applied": True,
            "ocr_engine": "tesseract-5.3.4",
            "ocr_languages": runtime_config["languages"],
            "ocr_dpi": int(runtime_config["dpi"]),
        }
        records.append(record)
        print(f"{item['document_id']} p.{item['page']}: {len(text)} chars")
    output = resolve_path(config["paths"]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} OCR pages to {output}")


if __name__ == "__main__":
    main()
