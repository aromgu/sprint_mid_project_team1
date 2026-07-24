from __future__ import annotations

import argparse
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from src.chunking.structured_chunker import chunk_pages
from src.ingestion.models import DocumentManifest
from src.ingestion.pdf_parser import parse_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = PROJECT_ROOT / "goldenset" / "golden_set_v3.jsonl"
DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "raw" / "unused_pdfs"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval_corpus_v3"


def normalized_filename(value: str) -> str:
    return "".join(re.findall(r"[0-9a-z가-힣]+", Path(value).stem.casefold()))


def load_golden_documents(path: Path) -> list[str]:
    documents = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name = json.loads(line)["source_document"]
        if name not in documents:
            documents.append(name)
    return documents


def match_documents(golden_names: list[str], pdf_paths: list[Path]) -> list[tuple[str, Path, float]]:
    available = set(pdf_paths)
    matches = []
    for golden_name in golden_names:
        target = normalized_filename(golden_name)
        ranked = sorted(
            ((SequenceMatcher(None, target, normalized_filename(path.name)).ratio(), path) for path in available),
            reverse=True,
        )
        if not ranked or ranked[0][0] < 0.72:
            raise ValueError(f"No reliable PDF match for {golden_name!r}")
        score, path = ranked[0]
        available.remove(path)
        matches.append((golden_name, path, score))
    return matches


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an isolated corpus for Golden Set v3.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    golden_names = load_golden_documents(args.golden)
    matches = match_documents(golden_names, sorted(args.pdf_dir.glob("*.pdf")))
    all_pages: list[dict] = []
    all_chunks: list[dict] = []
    manifest_rows = []
    diagnostics = []
    source_map = {}

    for index, (golden_name, pdf_path, match_score) in enumerate(matches, 1):
        content = pdf_path.read_bytes()
        document_id = f"eval_{index:02d}"
        title = pdf_path.stem
        manifest = DocumentManifest(
            document_id=document_id,
            difficulty="evaluation",
            organization=title.split("_", 1)[0],
            title=title,
            csv_filename=pdf_path.name,
            pdf_path=str(pdf_path.resolve()),
            filename=pdf_path.name,
            file_size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            match_score=match_score,
            target_accuracy="golden-v3",
            complexity="evaluation-corpus",
        )
        pages, diagnostic = parse_document(manifest)
        chunks = chunk_pages(pages)
        all_pages.extend(page.to_dict() for page in pages)
        all_chunks.extend(chunk.to_dict() for chunk in chunks)
        manifest_rows.append(manifest.to_dict())
        diagnostic["chunk_count"] = len(chunks)
        diagnostics.append(diagnostic)
        source_map[golden_name] = {
            "document_id": document_id,
            "pdf_filename": pdf_path.name,
            "match_score": round(match_score, 4),
        }
        print(f"{document_id}: {pdf_path.name} pages={len(pages)} chunks={len(chunks)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "processed" / "pages.jsonl", all_pages)
    write_jsonl(args.output_dir / "processed" / "chunks.jsonl", all_chunks)
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"documents": manifest_rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "source_map.json").write_text(
        json.dumps(source_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps({"documents": diagnostics}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"documents": len(matches), "pages": len(all_pages), "chunks": len(all_chunks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
