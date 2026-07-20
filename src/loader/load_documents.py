"""원본 RFP 파일을 찾아 전처리 단계에 넘길 목록을 만든다.

이 단계에서는 HWP/PDF 원문을 변환하거나 정제하지 않는다. 파일 위치, 형식,
크기, SHA-256처럼 원본에서 바로 확인할 수 있는 정보만 읽는다. 실제 본문과
표·이미지 구조를 꺼내는 작업은 ``preprocessing`` 모듈에서 담당한다.

원본 파일은 비공개 데이터이므로 이 모듈은 파일을 Git 저장소로 복사하거나
새 산출물로 저장하지 않는다.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset({".hwp", ".hwpx", ".pdf"})
HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """전처리 전 원본 파일 한 개의 출처 정보다.

    내용이 완전히 같은 파일은 ``source_id``를 공유한다. 대표 파일 선택은
    팀 검토 정책이 필요한 작업이므로, loader에서는 경로 순서로 고른 기본
    대표만 표시하고 최종 판단은 preprocessing 단계에 맡긴다.
    """

    source_id: str
    document_id: str
    source_path: Path
    source_relative_path: str
    source_filename: str
    source_sha256: str
    file_type: str
    source_file_size_bytes: int
    duplicate_group_size: int
    is_default_canonical: bool
    default_canonical_filename: str

    def as_metadata(self) -> dict[str, str | int | bool]:
        """Path 객체를 제외하고 JSON으로 저장 가능한 출처 정보만 반환한다."""
        return {
            "source_id": self.source_id,
            "document_id": self.document_id,
            "source_relative_path": self.source_relative_path,
            "source_filename": self.source_filename,
            "source_sha256": self.source_sha256,
            "file_type": self.file_type,
            "source_file_size_bytes": self.source_file_size_bytes,
            "duplicate_group_size": self.duplicate_group_size,
            "is_default_canonical": self.is_default_canonical,
            "default_canonical_filename": self.default_canonical_filename,
        }


def _normalize_extensions(extensions: Iterable[str]) -> frozenset[str]:
    """``hwp``와 ``.HWP`` 같은 입력을 모두 ``.hwp`` 형식으로 통일한다."""
    normalized = {
        extension.casefold()
        if extension.startswith(".")
        else f".{extension.casefold()}"
        for extension in extensions
        if extension
    }
    if not normalized:
        raise ValueError("지원할 파일 확장자가 하나 이상 필요합니다")
    return frozenset(normalized)


def _source_root(source_dir: str | Path) -> Path:
    """원본 폴더가 실제로 존재하는지 확인하고 절대경로로 통일한다."""
    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"원본 문서 폴더를 찾을 수 없습니다: {root}")
    return root


def discover_source_files(
    source_dir: str | Path,
    *,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
) -> list[Path]:
    """원본 폴더 아래에서 지원하는 문서만 찾아 항상 같은 순서로 반환한다."""
    root = _source_root(source_dir)
    supported = _normalize_extensions(extensions)
    source_files: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative_path = path.relative_to(root)
        # macOS 임시 파일이나 숨김 폴더 안의 파일은 원본 문서로 세지 않는다.
        if any(part.startswith(".") for part in relative_path.parts):
            continue
        if path.suffix.casefold() in supported:
            source_files.append(path)

    return sorted(
        source_files,
        key=lambda path: unicodedata.normalize(
            "NFC", str(path.relative_to(root))
        ).casefold(),
    )


def sha256_file(path: str | Path) -> str:
    """큰 파일도 메모리에 전부 올리지 않고 1MB씩 읽어 SHA-256을 계산한다."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"원본 파일을 찾을 수 없습니다: {source}")

    digest = hashlib.sha256()
    with source.open("rb") as file:
        for chunk in iter(lambda: file.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_documents(
    source_dir: str | Path,
    *,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
) -> list[SourceDocument]:
    """원본 문서 목록을 만들고 SHA-256 기준 중복 관계를 표시한다.

    반환 목록에는 중복 파일도 모두 남긴다. 원본 100개가 실제로 들어왔는지
    확인할 수 있어야 하기 때문이다. 전처리에서 기본 대표만 사용하려면
    :func:`select_default_canonical_documents`를 호출한다.
    """
    root = _source_root(source_dir)
    source_files = discover_source_files(root, extensions=extensions)
    files_by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in source_files:
        files_by_hash[sha256_file(path)].append(path)

    # source_id는 기존 전처리 결과와 호환되도록 SHA-256 앞 16자를 사용한다.
    # 서로 다른 전체 해시가 같은 짧은 ID를 만들면 조용히 진행하지 않는다.
    full_hash_by_source_id: dict[str, str] = {}
    for digest in files_by_hash:
        source_id = digest[:16]
        previous = full_hash_by_source_id.setdefault(source_id, digest)
        if previous != digest:
            raise ValueError(f"SHA-256 축약 ID가 충돌했습니다: {source_id}")

    documents: list[SourceDocument] = []
    for digest, paths in files_by_hash.items():
        ordered_paths = sorted(
            paths,
            key=lambda path: unicodedata.normalize(
                "NFC", str(path.relative_to(root))
            ).casefold(),
        )
        canonical_path = ordered_paths[0]
        canonical_filename = unicodedata.normalize("NFC", canonical_path.name)
        source_id = digest[:16]

        for path in ordered_paths:
            documents.append(
                SourceDocument(
                    source_id=source_id,
                    document_id=source_id,
                    source_path=path,
                    source_relative_path=str(path.relative_to(root)),
                    source_filename=unicodedata.normalize("NFC", path.name),
                    source_sha256=digest,
                    file_type=path.suffix.casefold().lstrip("."),
                    source_file_size_bytes=path.stat().st_size,
                    duplicate_group_size=len(ordered_paths),
                    is_default_canonical=path == canonical_path,
                    default_canonical_filename=canonical_filename,
                )
            )

    documents.sort(
        key=lambda document: unicodedata.normalize(
            "NFC", document.source_relative_path
        ).casefold()
    )
    return documents


def select_default_canonical_documents(
    documents: Iterable[SourceDocument],
) -> list[SourceDocument]:
    """각 SHA-256 그룹에서 loader가 표시한 기본 대표 파일만 반환한다.

    이 함수의 선택은 자동 기본값이다. 팀에서 정한 대표 파일 정책이 있다면
    preprocessing에서 그 정책을 우선 적용해야 한다.
    """
    canonical = [document for document in documents if document.is_default_canonical]
    canonical.sort(key=lambda document: document.source_filename.casefold())
    return canonical
