from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(slots=True)
class TesseractRuntime:
    root: Path

    @property
    def executable(self) -> Path:
        return self.root / "usr" / "bin" / "tesseract"

    @property
    def library_dir(self) -> Path:
        return self.root / "usr" / "lib" / "x86_64-linux-gnu"

    @property
    def tessdata_dir(self) -> Path:
        return self.root / "usr" / "share" / "tesseract-ocr" / "5" / "tessdata"

    def validate(self, languages: str) -> None:
        if not self.executable.exists():
            raise FileNotFoundError(
                f"Tesseract not found at {self.executable}. "
                "Run the user-space Tesseract bootstrap first."
            )
        for language in languages.split("+"):
            if not (self.tessdata_dir / f"{language}.traineddata").exists():
                raise FileNotFoundError(f"Missing Tesseract language data: {language}")

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{self.library_dir}:{existing}" if existing else str(self.library_dir)
        env["TESSDATA_PREFIX"] = str(self.tessdata_dir)
        return env


def ocr_pdf_page(
    pdf_path: Path,
    page_number: int,
    runtime: TesseractRuntime,
    languages: str = "kor+eng",
    dpi: int = 300,
    psm: int = 6,
) -> str:
    runtime.validate(languages)
    document = pymupdf.open(pdf_path)
    if not 1 <= page_number <= len(document):
        document.close()
        raise ValueError(f"Page {page_number} is outside 1..{len(document)} for {pdf_path}")
    page = document[page_number - 1]
    scale = dpi / 72
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    with tempfile.TemporaryDirectory(prefix="rag_ocr_") as temp_dir:
        image_path = Path(temp_dir) / "page.png"
        pixmap.save(image_path)
        command = [
            str(runtime.executable), str(image_path), "stdout", "-l", languages,
            "--psm", str(psm), "--dpi", str(dpi),
        ]
        completed = subprocess.run(
            command,
            env=runtime.environment(),
            capture_output=True,
            text=True,
            check=True,
        )
    document.close()
    return completed.stdout.strip()


def load_ocr_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

