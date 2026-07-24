from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.golden_v3 import GoldenV3Item, evaluate_result_set, summarize
from src.search.service import PROJECT_ROOT, SearchService


def load_items(path: Path) -> list[GoldenV3Item]:
    return [GoldenV3Item.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval natively against content/section Golden Set v3.")
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "goldenset" / "golden_set_v3.jsonl")
    parser.add_argument("--source-map", type=Path, default=PROJECT_ROOT / "data" / "eval_corpus_v3" / "source_map.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "search_eval_v3.yaml")
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid", "all"), default="all")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "evaluation_v3")
    args = parser.parse_args()

    items = load_items(args.golden)
    source_map = json.loads(args.source_map.read_text(encoding="utf-8"))
    service = SearchService(args.config)
    retrievers = ("bm25", "dense", "hybrid") if args.retriever == "all" else (args.retriever,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    section_audit: dict[str, dict] = {}

    answerable_items = [item for item in items if item.answerable]
    for retriever in retrievers:
        details = []
        latencies = []
        for item in answerable_items:
            document = source_map.get(item.source_document)
            if not document:
                raise KeyError(f"No source mapping for {item.source_document}")
            results = service.search(
                item.question, retriever=retriever, top_k=args.top_k,
                document_ids={document["document_id"]},
            )
            if results and results[0].latency_ms is not None:
                latencies.append(results[0].latency_ms)
            row = evaluate_result_set(item, results)
            row["document_id"] = document["document_id"]
            details.append(row)
            for section in item.gold_sections:
                audit = section_audit.setdefault(section, {"question_count": 0, "matched_questions": 0, "example_chunks": []})
                audit["question_count"] += 1
                hits = row["section_hits"].get(section, [])
                if hits:
                    audit["matched_questions"] += 1
                    for result in row["results"]:
                        if section in result["matched_sections"] and result["chunk_id"] not in audit["example_chunks"]:
                            audit["example_chunks"].append(result["chunk_id"])
                            if len(audit["example_chunks"]) >= 3:
                                break
        summary = summarize(details, retriever, latencies)
        summaries.append(summary)
        with (args.output_dir / f"{retriever}_details.jsonl").open("w", encoding="utf-8") as handle:
            for row in details:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    (args.output_dir / "retrieval_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "section_audit.json").write_text(json.dumps(section_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    unmapped = [section for section, value in section_audit.items() if value["matched_questions"] == 0]
    print(f"section audit: {len(section_audit) - len(unmapped)}/{len(section_audit)} mapped; unmapped={len(unmapped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
