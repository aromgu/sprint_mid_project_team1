from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from src.evaluation.golden_v3 import GoldenV3Item, fact_match_score, normalize
from src.search.service import PROJECT_ROOT


def numbers(value: str) -> set[str]:
    return {token.replace(",", "") for token in re.findall(r"\d[\d,]*(?:\.\d+)?%?", value)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score generated answers against content-based Golden Set v3.")
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "goldenset" / "golden_set_v3.jsonl")
    parser.add_argument("--responses", type=Path, default=PROJECT_ROOT / "reports" / "evaluation_v3" / "answers.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "evaluation_v3" / "answer_summary.json")
    args = parser.parse_args()

    golden = {
        item.question_id: item
        for item in (GoldenV3Item.from_dict(json.loads(line)) for line in args.golden.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    responses = [json.loads(line) for line in args.responses.read_text(encoding="utf-8").splitlines() if line.strip()]
    details = []
    for response in responses:
        item = golden[response["question_id"]]
        answer = response.get("answer", "")
        fact_scores = {fact: fact_match_score(fact, answer) for fact in item.required_facts}
        fact_coverage = sum(score >= 0.7 for score in fact_scores.values()) / len(fact_scores) if fact_scores else None
        expected_numbers = numbers(item.ground_truth)
        answer_numbers = numbers(answer)
        numeric_recall = len(expected_numbers & answer_numbers) / len(expected_numbers) if expected_numbers else None
        numeric_precision = len(expected_numbers & answer_numbers) / len(answer_numbers) if answer_numbers else (1.0 if not expected_numbers else 0.0)
        details.append({
            "question_id": item.question_id,
            "difficulty": item.difficulty,
            "query_type": item.query_type,
            "expected_answerable": item.answerable,
            "predicted_answerable": bool(response.get("is_answerable")),
            "answerability_correct": bool(response.get("is_answerable")) == item.answerable,
            "required_fact_coverage": fact_coverage,
            "numeric_recall": numeric_recall,
            "numeric_precision": numeric_precision,
            "ground_truth_token_coverage": fact_match_score(item.ground_truth, answer),
            "fact_scores": {key: round(value, 4) for key, value in fact_scores.items()},
            "expected_numbers": sorted(expected_numbers),
            "answer_numbers": sorted(answer_numbers),
        })

    metric_names = ["required_fact_coverage", "numeric_recall", "numeric_precision", "ground_truth_token_coverage"]
    summary = {
        "evaluated_questions": len(details),
        "answerability_accuracy": fmean(row["answerability_correct"] for row in details) if details else 0.0,
        **{
            name: fmean(row[name] for row in details if row[name] is not None)
            for name in metric_names
            if any(row[name] is not None for row in details)
        },
        "estimated_cost_usd": sum(float(row.get("estimated_cost_usd") or 0) for row in responses),
    }
    for dimension in ("difficulty", "query_type"):
        groups = defaultdict(list)
        for row in details:
            groups[row[dimension]].append(row)
        summary[f"by_{dimension}"] = {
            key: {
                "question_count": len(group),
                "answerability_accuracy": fmean(row["answerability_correct"] for row in group),
                **{
                    name: fmean(row[name] for row in group if row[name] is not None)
                    for name in metric_names if any(row[name] is not None for row in group)
                },
            }
            for key, group in groups.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
