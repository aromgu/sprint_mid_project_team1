from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from statistics import fmean
from typing import Any


GENERIC_SECTION_PARTS = {
    "사업", "과업", "개요", "내용", "상세", "요구", "요구사항", "제안", "제안요청내용",
    "상세요구사항", "사업개요", "과업의개요", "일반", "사항", "붙임", "별표",
}


def normalize(value: str) -> str:
    return "".join(re.findall(r"[0-9a-z가-힣]+", value.casefold()))


def requirement_ids(value: str) -> list[str]:
    return [
        match.upper().replace("_", "-")
        for match in re.findall(r"(?<![A-Za-z0-9])[A-Za-z]{2,5}[-_]\d{1,4}(?![A-Za-z0-9])", value)
    ]


def section_parts(value: str) -> list[str]:
    parts = [normalize(part) for part in value.split("_")]
    return [part for part in parts if part and not part.isdigit() and part not in GENERIC_SECTION_PARTS]


def section_match(gold_section: str, chunk) -> tuple[bool, float]:
    gold_requirements = requirement_ids(gold_section)
    chunk_requirements = {item.upper() for item in chunk.requirement_ids}
    chunk_text = normalize(chunk.text)
    if gold_requirements:
        matched = all(req in chunk_requirements or normalize(req) in chunk_text for req in gold_requirements)
        return matched, 1.0 if matched else 0.0

    candidates = [normalize(item) for item in chunk.section_path if item]
    parts = section_parts(gold_section)
    if not parts:
        return False, 0.0
    keys = [parts[-1]]
    if len(parts) > 1:
        keys.append("".join(parts[-2:]))
    haystacks = candidates + [chunk_text]
    best = 0.0
    for key in keys:
        for candidate in haystacks:
            if not key or not candidate:
                continue
            if key in candidate:
                return True, 1.0
            if candidate in key and len(candidate) >= 4:
                best = max(best, len(candidate) / len(key))
            elif candidate in candidates:
                best = max(best, SequenceMatcher(None, key, candidate).ratio())
    return best >= 0.72, best


def fact_match_score(fact: str, context: str) -> float:
    fact_tokens = list(dict.fromkeys(re.findall(r"[0-9]+(?:[.,][0-9]+)*%?|[a-zA-Z]{2,}|[가-힣]{2,}", fact.casefold())))
    if not fact_tokens:
        return 0.0
    normalized_context = normalize(context)
    matched = sum(normalize(token) in normalized_context for token in fact_tokens)
    numeric_tokens = [token for token in fact_tokens if any(char.isdigit() for char in token)]
    if numeric_tokens and not all(normalize(token) in normalized_context for token in numeric_tokens):
        return 0.0
    return matched / len(fact_tokens)


@dataclass(slots=True)
class GoldenV3Item:
    question_id: str
    question: str
    source_document: str
    gold_sections: list[str]
    gold_sections_optional: list[str]
    ground_truth: str
    required_facts: list[str]
    answerable: bool
    query_type: str
    difficulty: str
    hop_count: int
    review_status: str
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "GoldenV3Item":
        known = {field.name for field in cls.__dataclass_fields__.values()} - {"extra"}
        return cls(**{key: row[key] for key in known}, extra={key: value for key, value in row.items() if key not in known})


def evaluate_result_set(item: GoldenV3Item, results: list, ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    section_hits: dict[str, list[int]] = defaultdict(list)
    ranked_relevant = []
    result_rows = []
    accumulated_context = ""
    contexts = []
    for rank, result in enumerate(results, 1):
        matched_sections = []
        section_scores = {}
        for section in item.gold_sections:
            matched, score = section_match(section, result.chunk)
            section_scores[section] = round(score, 4)
            if matched:
                section_hits[section].append(rank)
                matched_sections.append(section)
        accumulated_context = f"{accumulated_context}\n{result.context_text or result.chunk.text}"
        contexts.append(result.context_text or result.chunk.text)
        current_fact_scores = {fact: fact_match_score(fact, accumulated_context) for fact in item.required_facts}
        relevant = bool(matched_sections) or any(score >= 0.7 for score in current_fact_scores.values())
        if relevant:
            ranked_relevant.append(rank)
        result_rows.append({
            "rank": rank,
            "chunk_id": result.chunk.chunk_id,
            "page_start": result.chunk.page_start,
            "page_end": result.chunk.page_end,
            "matched_sections": matched_sections,
            "section_scores": section_scores,
        })

    metrics = {}
    fact_scores_by_k = {}
    for k in ks:
        required_count = len(item.gold_sections)
        hit_count = sum(any(rank <= k for rank in section_hits.get(section, [])) for section in item.gold_sections)
        context_at_k = "\n".join(contexts[:k])
        fact_scores = {fact: fact_match_score(fact, context_at_k) for fact in item.required_facts}
        fact_scores_by_k[str(k)] = {key: round(value, 4) for key, value in fact_scores.items()}
        matched_facts = sum(score >= 0.7 for score in fact_scores.values())
        metrics[f"section_recall@{k}"] = hit_count / required_count if required_count else None
        metrics[f"full_section_hit@{k}"] = float(hit_count == required_count) if required_count else None
        metrics[f"fact_coverage@{k}"] = matched_facts / len(fact_scores) if fact_scores else None
    metrics["mrr@10"] = 1 / ranked_relevant[0] if ranked_relevant and ranked_relevant[0] <= 10 else 0.0
    return {
        "question_id": item.question_id,
        "question": item.question,
        "source_document": item.source_document,
        "query_type": item.query_type,
        "difficulty": item.difficulty,
        "hop_count": item.hop_count,
        "gold_sections": item.gold_sections,
        "section_hits": dict(section_hits),
        "fact_scores_by_k": fact_scores_by_k,
        "metrics": metrics,
        "results": result_rows,
    }


def summarize(rows: list[dict], retriever: str, latency_values: list[float]) -> dict:
    metric_names = sorted({name for row in rows for name, value in row["metrics"].items() if value is not None})
    summary = {
        "retriever": retriever,
        "scope": "document",
        "query_count": len(rows),
        **{name: fmean(row["metrics"][name] for row in rows if row["metrics"].get(name) is not None) for name in metric_names},
        "latency_ms": fmean(latency_values) if latency_values else 0.0,
    }
    breakdown = {}
    for dimension in ("difficulty", "query_type"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[dimension]].append(row)
        breakdown[dimension] = {
            key: {
                "query_count": len(group),
                **{
                    name: fmean(row["metrics"][name] for row in group if row["metrics"].get(name) is not None)
                    for name in metric_names
                    if any(row["metrics"].get(name) is not None for row in group)
                },
            }
            for key, group in groups.items()
        }
    summary["breakdown"] = breakdown
    return summary
