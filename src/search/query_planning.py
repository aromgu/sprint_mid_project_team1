from __future__ import annotations

import re
from dataclasses import dataclass

from src.search.models import SearchResult


TERM_GROUPS: dict[str, tuple[str, ...]] = {
    "예산": ("예산", "총예산", "사업비", "소요예산", "추정가격", "계약금액"),
    "유지보수": ("유지보수", "무상유지보수", "하자보수", "유지관리", "안정화 지원"),
}


@dataclass(frozen=True, slots=True)
class SearchQuestion:
    label: str
    question: str
    expanded_query: str


def plan_search_questions(question: str) -> list[SearchQuestion]:
    """Split a compound question by requested field and add RFP terminology."""
    normalized = " ".join(question.split())
    # A bare `과/와 + space` is not sufficient: words such as `부과 대상` would
    # be damaged. Mark conjunctions only after nouns commonly used as requested
    # fields, while commas and explicit conjunctions remain unconditional.
    field_nouns = (
        "무상유지보수기간|유지보수기간|총예산|소요예산|사업비|예산|항목|기간|"
        "방식|금액|구성|기한|조건|기준|방법|범위|책임|목적|점수|"
        "서류|서식|코드|제한|목표값|처리|절차|내용|대상|비율|지분율|요건|자격"
    )
    marked = re.sub(
        rf"({field_nouns})(?:과|와)\s+",
        r"\1|||",
        normalized,
    )
    parts = [
        part.strip(" ,?？")
        for part in re.split(r"\s*(?:\|\|\||,|및|그리고)\s*", marked)
        if part.strip(" ,?？")
    ]
    if not parts:
        return []

    tail = parts[-1]
    predicate_match = re.search(r"(은|는|이|가|을|를).+$", tail)
    predicate = predicate_match.group(0) if predicate_match else ""
    plans: list[SearchQuestion] = []
    for part in parts:
        label = part
        expanded_terms = [part]
        for group_label, terms in TERM_GROUPS.items():
            if any(term in part for term in terms):
                label = group_label
                expanded_terms.extend(terms)
        subquestion = part + (predicate if len(parts) > 1 and predicate and predicate not in part else "")
        plans.append(SearchQuestion(
            label=label,
            question=subquestion,
            expanded_query=" ".join(dict.fromkeys(expanded_terms)),
        ))
    return plans


_AMOUNT_RE = re.compile(r"(?:\d[\d,]*(?:\.\d+)?\s*(?:원|만원|억원)|금\s*\d[\d,]*)")
_PERIOD_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:일|개월|년|주|시간)")
_REQUIREMENT_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z]+)?-?\d{2,}\b", re.IGNORECASE)


def merge_and_boost_results(
    result_groups: list[list[SearchResult]],
    *,
    top_k: int,
    requirement_boost: float = 0.15,
    amount_boost: float = 0.12,
    period_boost: float = 0.12,
) -> list[SearchResult]:
    """Deduplicate subquery results, reward repeated hits and evidence-rich chunks."""
    merged: dict[str, SearchResult] = {}
    hit_counts: dict[str, int] = {}
    for group in result_groups:
        for result in group:
            chunk_id = result.chunk.chunk_id
            hit_counts[chunk_id] = hit_counts.get(chunk_id, 0) + 1
            if chunk_id not in merged or result.score > merged[chunk_id].score:
                merged[chunk_id] = result

    for chunk_id, result in merged.items():
        text = result.chunk.text
        multiplier = 1.0
        if result.chunk.requirement_ids or _REQUIREMENT_RE.search(text):
            multiplier += requirement_boost
        if _AMOUNT_RE.search(text):
            multiplier += amount_boost
        if _PERIOD_RE.search(text):
            multiplier += period_boost
        # A chunk found by several independent field searches is stronger evidence.
        multiplier += 0.05 * (hit_counts[chunk_id] - 1)
        result.score *= multiplier
        result.component_scores["evidence_boost"] = multiplier

    ranked = sorted(merged.values(), key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k]
    for rank, result in enumerate(ranked, 1):
        result.rank = rank
    return ranked
