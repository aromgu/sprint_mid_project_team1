"""Evaluate Main Advanced dense retrieval against Golden Set v3."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean

from src.evaluation.golden_v3 import GoldenV3Item, evaluate_result_set, summarize
from src.main_rag.evaluation import adapt_retrieval_results
from src.main_rag.retrieval.advanced_retriever import AdvancedRetriever
from src.main_rag.settings import load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "goldenset/golden_set_v3.jsonl")
    parser.add_argument("--source-map", type=Path, default=PROJECT_ROOT / "data/eval_corpus_v3/source_map.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/main_advanced")
    args = parser.parse_args()
    items = [
        GoldenV3Item.from_dict(json.loads(line))
        for line in args.golden.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_map = json.loads(args.source_map.read_text(encoding="utf-8"))
    retriever = AdvancedRetriever(load_settings())
    details = []
    latencies = []
    for index, item in enumerate((item for item in items if item.answerable), 1):
        document = source_map[item.source_document]
        started = time.perf_counter()
        raw = retriever.search_documents(
            item.question, top_k=args.top_k, document_id=document["document_id"]
        )
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        row = evaluate_result_set(
            item, adapt_retrieval_results(raw, latency_ms=latency_ms), ks=(1, 3, 5, 10)
        )
        row["document_id"] = document["document_id"]
        row["latency_ms"] = round(latency_ms, 4)
        row["wrong_document"] = any(
            str((value.get("metadata") or {}).get("document_id")) != document["document_id"]
            for value in raw
        )
        # Hit means at least one gold section or required fact is represented.
        for k in (1, 3, 5):
            section_hit = any(rank <= k for ranks in row["section_hits"].values() for rank in ranks)
            fact_hit = any(score >= 0.7 for score in row["fact_scores_by_k"][str(k)].values())
            row["metrics"][f"hit@{k}"] = float(section_hit or fact_hit)
        details.append(row)
        print(f"{index}/{sum(i.answerable for i in items)} {item.question_id}", flush=True)

    summary = summarize(details, "main_advanced_dense", latencies)
    summary["latency_ms_max"] = max(latencies, default=0.0)
    summary["empty_result_count"] = sum(not row["results"] for row in details)
    summary["wrong_document_count"] = sum(row["wrong_document"] for row in details)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "retrieval_details.jsonl").open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "retrieval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
