import json
from pathlib import Path

from src.ingestion.manifest import build_manifest, normalize_filename
from src.ingestion.models import DocumentManifest, PROJECT_ROOT


def test_normalize_filename_handles_rfp_punctuation() -> None:
    left = "고려대학교_차세대_포털_학사_정보시스템_구축사업.pdf"
    right = "고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf"
    assert normalize_filename(left) == normalize_filename(right)


def test_build_manifest_matches_normalized_names(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdf"
    pdf_dir.mkdir()
    csv_path = tmp_path / "documents.csv"
    rows = [
        "난이도,순번,발주기관,사업명,파일명,복잡도,목표정확도",
        "상,상_1,기관,테스트 사업,기관_테스트_사업.pdf,복잡,80%+",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8-sig")
    (pdf_dir / "기관_테스트 사업.pdf").write_bytes(b"pdf-placeholder")

    try:
        build_manifest(csv_path, pdf_dir)
    except ValueError as exc:
        assert "Expected exactly 9" in str(exc)
    else:
        raise AssertionError("The nine-document invariant must be enforced")


def test_manifest_resolves_pdf_after_checkout_path_changes() -> None:
    payload = json.loads((PROJECT_ROOT / "data/manifests/documents.json").read_text(encoding="utf-8"))
    row = {**payload["documents"][0], "pdf_path": "/missing/checkout/data/raw/document.pdf"}

    resolved = DocumentManifest(**row).resolved_pdf_path()

    assert resolved.exists()
    assert resolved.name == row["filename"]
