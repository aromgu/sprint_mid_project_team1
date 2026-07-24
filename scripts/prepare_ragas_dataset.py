"""Validate/normalize the Golden set fields needed by RAGAS later."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {"question_id", "question", "reference_context_ids"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation/ragas_dataset.jsonl"))
    args = parser.parse_args()
    rows = []
    for line_no, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"line {line_no}: missing {sorted(missing)}")
        rows.append({
            "user_input": row["question"],
            "reference": row.get("reference_answer"),
            "reference_contexts": row.get("reference_contexts", row["reference_context_ids"]),
            "question_id": row["question_id"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"prepared {len(rows)} rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
