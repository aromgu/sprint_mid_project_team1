"""외부 임베딩 API에 보낼 텍스트에서 개인정보·비밀값을 마스킹한다."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, TextIO

import tiktoken


PRIVACY_SCHEMA_VERSION = "embedding_text_redaction_v1"
TOKENIZER_ENCODING = "cl100k_base"
MAX_EMBEDDING_TOKENS = 512
Replacement = str | Callable[[re.Match[str]], str]


@dataclass(frozen=True)
class RedactionRule:
    """한 종류의 민감정보를 찾고 대체하는 결정적 규칙."""

    name: str
    pattern: re.Pattern[str]
    replacement: Replacement


@dataclass(frozen=True)
class RedactionReport:
    """마스킹 입력 계보와 검출·토큰 품질 결과."""

    input_path: str
    input_sha256: str
    output_path: str
    output_sha256: str
    privacy_schema_version: str
    tokenizer_encoding: str
    chunk_count: int
    document_count: int
    changed_chunk_count: int
    changed_document_count: int
    project_summary_changed_chunk_count: int
    redaction_occurrence_count: int
    redaction_type_counts: dict[str, int]
    embedding_total_tokens: int
    embedding_max_tokens: int
    empty_embedding_text_count: int
    over_limit_embedding_text_count: int
    source_text_field: str
    external_api_text_field: str


def preserve_prefix(marker: str) -> Callable[[re.Match[str]], str]:
    """라벨은 남기고 개인정보 값만 marker로 바꾸는 함수를 만든다."""

    def replace(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{marker}"

    return replace


REDACTION_RULES = (
    RedactionRule(
        "private_key",
        re.compile(
            r"-----BEGIN (?:(?:RSA|EC|OPENSSH) )?PRIVATE KEY-----.*?"
            r"-----END (?:(?:RSA|EC|OPENSSH) )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    RedactionRule(
        "bearer_token",
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{16,}={0,2}"),
        "Bearer [REDACTED_TOKEN]",
    ),
    RedactionRule(
        "aws_access_key",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
        "[REDACTED_ACCESS_KEY]",
    ),
    RedactionRule(
        "password",
        re.compile(
            r"(?P<prefix>(?:password|passwd|비밀번호|접속암호)\s*[:=：]\s*)"
            r"(?P<value>[^\s|,;]{4,})",
            re.IGNORECASE,
        ),
        preserve_prefix("[REDACTED_PASSWORD]"),
    ),
    RedactionRule(
        "resident_or_foreign_id",
        re.compile(
            r"(?P<prefix>(?:주민(?:등록)?번호|외국인등록번호)\s*[:：]?\s*)"
            r"(?P<value>\d{6}[-\s]?[1-8]\d{6})"
        ),
        preserve_prefix("[REDACTED_ID]"),
    ),
    RedactionRule(
        "birthdate",
        re.compile(
            r"(?P<prefix>(?:생년월일|출생일)\s*[:：]?\s*)"
            r"(?P<value>(?:19|20)?\d{2}[-./년\s]\d{1,2}[-./월\s]\d{1,2}일?)"
        ),
        preserve_prefix("[REDACTED_BIRTHDATE]"),
    ),
    RedactionRule(
        "account_number",
        re.compile(
            r"(?P<prefix>(?:계좌번호|입금계좌)\s*[:：]?\s*)"
            r"(?P<value>\d[\d-]{7,29})"
        ),
        preserve_prefix("[REDACTED_ACCOUNT]"),
    ),
    RedactionRule(
        "card_number",
        re.compile(r"(?P<prefix>카드번호\s*[:：]?\s*)(?P<value>\d[\d-]{12,24})"),
        preserve_prefix("[REDACTED_CARD]"),
    ),
    RedactionRule(
        "email",
        re.compile(
            r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"
            r"(?![\w.-])",
            re.IGNORECASE,
        ),
        "[REDACTED_EMAIL]",
    ),
    RedactionRule(
        "phone",
        re.compile(r"(?<!\d)(?:\+?82[-.\s]?)?0\d{1,2}[-.\s]\d{3,4}[-.\s]\d{4}(?!\d)"),
        "[REDACTED_PHONE]",
    ),
    RedactionRule(
        "mobile_compact",
        re.compile(r"(?<!\d)01[016789]\d{7,8}(?!\d)"),
        "[REDACTED_PHONE]",
    ),
    RedactionRule(
        "person_name_labeled",
        re.compile(
            r"(?P<prefix>(?:담당자|성명|책임자)\s*[:：|]\s*)"
            r"(?P<value>"
            r"(?!(?:연락처|전화번호|이메일|부서명|소속|직위|직책|기관명|회사명|제안사)"
            r"(?![가-힣]))"
            r"[가-힣]{2,4})(?![가-힣])"
        ),
        preserve_prefix("[이름]"),
    ),
    RedactionRule(
        "ipv4",
        re.compile(
            r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)"
            r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)"
        ),
        "[REDACTED_IP]",
    ),
    RedactionRule(
        "mac_address",
        re.compile(
            r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}[:-]){5}"
            r"[0-9A-Fa-f]{2}(?![0-9A-Fa-f])"
        ),
        "[REDACTED_MAC]",
    ),
)


def sha256_file(path: Path) -> str:
    """파일 전체를 메모리에 올리지 않고 SHA-256을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_jsonl(path: Path) -> TextIO:
    """일반 JSONL과 gzip JSONL을 같은 텍스트 스트림으로 연다."""

    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


@contextmanager
def deterministic_gzip_writer(path: Path) -> Iterator[TextIO]:
    """같은 입력과 규칙이면 같은 gzip SHA가 나오도록 mtime을 고정한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_file:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_file, mtime=0
        ) as gzip_file:
            with io.TextIOWrapper(
                gzip_file, encoding="utf-8", newline="\n"
            ) as text_file:
                yield text_file


def redact_text(value: str) -> tuple[str, Counter[str]]:
    """정해진 순서로 민감정보를 마스킹하고 유형별 횟수를 반환한다."""

    redacted = value
    counts: Counter[str] = Counter()
    for rule in REDACTION_RULES:
        redacted, count = rule.pattern.subn(rule.replacement, redacted)
        if count:
            counts[rule.name] += count
    return redacted, counts


def redact_embedding_corpus(
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> RedactionReport:
    """원문을 유지하면서 외부 전송 전용 embedding_text를 추가한다."""

    codec = tiktoken.get_encoding(TOKENIZER_ENCODING)
    chunk_count = 0
    documents: set[str] = set()
    changed_chunks: set[str] = set()
    changed_documents: set[str] = set()
    summary_changed_chunks: set[str] = set()
    total_counts: Counter[str] = Counter()
    embedding_total_tokens = 0
    embedding_max_tokens = 0
    empty_count = 0
    over_limit_count = 0

    with open_jsonl(input_path) as input_file:
        with deterministic_gzip_writer(output_path) as output_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                chunk_id = str(row.get("chunk_id") or "")
                document_id = str(row.get("document_id") or "")
                retrieval_text = row.get("retrieval_text")
                if not chunk_id or not document_id:
                    raise ValueError(f"청크 식별자가 비어 있습니다: line={line_number}")
                if not isinstance(retrieval_text, str) or not retrieval_text.strip():
                    raise ValueError(f"retrieval_text가 비어 있습니다: {chunk_id}")

                embedding_text, text_counts = redact_text(retrieval_text)
                project_summary = row.get("project_summary")
                summary_counts: Counter[str] = Counter()
                if isinstance(project_summary, str) and project_summary:
                    redacted_summary, summary_counts = redact_text(project_summary)
                    row["project_summary"] = redacted_summary
                    if redacted_summary != project_summary:
                        summary_changed_chunks.add(chunk_id)

                row_counts = text_counts + summary_counts
                total_counts.update(row_counts)
                if embedding_text != retrieval_text:
                    changed_chunks.add(chunk_id)
                    changed_documents.add(document_id)

                embedding_token_count = len(codec.encode(embedding_text))
                if not embedding_text.strip():
                    empty_count += 1
                if embedding_token_count > MAX_EMBEDDING_TOKENS:
                    over_limit_count += 1

                row["embedding_text"] = embedding_text
                row["embedding_token_count"] = embedding_token_count
                row["privacy_schema_version"] = PRIVACY_SCHEMA_VERSION
                row["sensitive_text_redaction_count"] = sum(row_counts.values())
                row["sensitive_text_redaction_types"] = sorted(row_counts)

                embedding_total_tokens += embedding_token_count
                embedding_max_tokens = max(
                    embedding_max_tokens,
                    embedding_token_count,
                )
                documents.add(document_id)
                chunk_count += 1
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    if empty_count or over_limit_count:
        raise ValueError(
            f"마스킹 결과 품질 오류: empty={empty_count}, over_limit={over_limit_count}"
        )

    report = RedactionReport(
        input_path=str(input_path),
        input_sha256=sha256_file(input_path),
        output_path=str(output_path),
        output_sha256=sha256_file(output_path),
        privacy_schema_version=PRIVACY_SCHEMA_VERSION,
        tokenizer_encoding=TOKENIZER_ENCODING,
        chunk_count=chunk_count,
        document_count=len(documents),
        changed_chunk_count=len(changed_chunks),
        changed_document_count=len(changed_documents),
        project_summary_changed_chunk_count=len(summary_changed_chunks),
        redaction_occurrence_count=sum(total_counts.values()),
        redaction_type_counts=dict(sorted(total_counts.items())),
        embedding_total_tokens=embedding_total_tokens,
        embedding_max_tokens=embedding_max_tokens,
        empty_embedding_text_count=empty_count,
        over_limit_embedding_text_count=over_limit_count,
        source_text_field="retrieval_text",
        external_api_text_field="embedding_text",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
