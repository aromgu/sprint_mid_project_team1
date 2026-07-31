"""Evaluate and visualize Main Advanced Dense/Hybrid retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

from src.evaluation.golden_v3 import GoldenV3Item, evaluate_result_set, summarize
from src.main_rag.evaluation import adapt_retrieval_results
from src.main_rag.retrieval.advanced_retriever import AdvancedRetriever
from src.main_rag.settings import load_settings
from src.observability.wandb import init_wandb_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def evaluate(mode: str, items: list[GoldenV3Item], source_map: dict, top_k: int) -> tuple[list[dict], dict]:
    retriever = AdvancedRetriever(load_settings(), mode=mode)
    details, latencies = [], []
    answerable = [item for item in items if item.answerable]
    for index, item in enumerate(answerable, 1):
        document = source_map[item.source_document]
        started = time.perf_counter()
        raw = retriever.search_documents(item.question, top_k=top_k, document_id=document["document_id"])
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        row = evaluate_result_set(item, adapt_retrieval_results(raw, latency_ms=latency_ms), ks=(1, 3, 5, 10))
        row.update({
            "document_id": document["document_id"], "latency_ms": round(latency_ms, 4),
            "wrong_document": any(str((value.get("metadata") or {}).get("document_id")) != document["document_id"] for value in raw),
            "diagnostics": [{key: value for key, value in value.items() if key in {"chunk_id", "dense_rank", "bm25_rank", "bm25_score", "rrf_score", "score_type"}} for value in raw],
        })
        for k in (1, 3, 5):
            section_hit = any(rank <= k for ranks in row["section_hits"].values() for rank in ranks)
            fact_hit = any(score >= 0.7 for score in row["fact_scores_by_k"][str(k)].values())
            row["metrics"][f"hit@{k}"] = float(section_hit or fact_hit)
        details.append(row)
        print(f"{mode} {index}/{len(answerable)} {item.question_id}", flush=True)
    summary = summarize(details, "hybrid_rrf" if mode == "hybrid_rrf" else "main_advanced_dense", latencies)
    summary.update({
        "latency_ms_p50": _percentile(latencies, .50), "latency_ms_p95": _percentile(latencies, .95),
        "latency_ms_max": max(latencies, default=0.0), "empty_result_count": sum(not row["results"] for row in details),
        "wrong_document_count": sum(row["wrong_document"] for row in details), "max_workers": 1,
    })
    return details, summary


def log_wandb(run, details: list[dict], summary: dict, output_dir: Path) -> None:
    import wandb
    metrics = {
        "retrieval/hit_at_1": summary.get("hit@1", 0), "retrieval/hit_at_3": summary.get("hit@3", 0),
        "retrieval/hit_at_5": summary.get("hit@5", 0), "retrieval/mrr_at_10": summary.get("mrr@10", 0),
        "retrieval/section_recall_at_5": summary.get("section_recall@5", 0),
        "retrieval/fact_coverage_at_5": summary.get("fact_coverage@5", 0),
        "retrieval/query_count": summary.get("query_count", 0),
        "retrieval/empty_result_count": summary.get("empty_result_count", 0),
        "retrieval/wrong_document_count": summary.get("wrong_document_count", 0),
        "latency/search_mean_ms": summary.get("latency_ms", 0),
        "latency/search_p50_ms": summary.get("latency_ms_p50", 0),
        "latency/search_p95_ms": summary.get("latency_ms_p95", 0),
        "latency/search_max_ms": summary.get("latency_ms_max", 0),
        "runtime/max_workers": summary["max_workers"],
    }
    run.log(metrics)
    columns = ["query_id", "document_id", "question", "difficulty", "query_type", "hit_at_1", "hit_at_3", "hit_at_5", "mrr_at_10", "latency_ms", "retrieved_chunks", "diagnostics", "wrong_document"]
    table = wandb.Table(columns=columns)
    for row in details:
        table.add_data(row["question_id"], row["document_id"], row["question"], row["difficulty"], row["query_type"], row["metrics"].get("hit@1"), row["metrics"].get("hit@3"), row["metrics"].get("hit@5"), row["metrics"].get("mrr@10"), row["latency_ms"], json.dumps([value["chunk_id"] for value in row["results"]], ensure_ascii=False), json.dumps(row["diagnostics"], ensure_ascii=False), row["wrong_document"])
    run.log({"retrieval/query_results": table})
    breakdown = wandb.Table(columns=["dimension", "value", "query_count", "hit_at_5", "mrr_at_10", "section_recall_at_5", "fact_coverage_at_5"])
    for dimension, groups in summary.get("breakdown", {}).items():
        for value, values in groups.items():
            breakdown.add_data(dimension, value, values.get("query_count"), values.get("hit@5"), values.get("mrr@10"), values.get("section_recall@5"), values.get("fact_coverage@5"))
    run.log({"retrieval/breakdown": breakdown})
    artifact = wandb.Artifact(f"main-advanced-9-{summary['retriever']}-evaluation", type="retrieval-evaluation", metadata={"retriever": summary["retriever"], "scope": "document", "document_count": 9})
    artifact.add_file(str(output_dir / "retrieval_summary.json"))
    artifact.add_file(str(output_dir / "retrieval_details.jsonl"))
    run.log_artifact(artifact)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "goldenset/golden_set_v3.jsonl")
    parser.add_argument("--source-map", type=Path, default=PROJECT_ROOT / "data/eval_corpus_v3/source_map.json")
    parser.add_argument("--retriever", choices=("dense", "hybrid_rrf", "all"), default="all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, help="대표 질문 smoke용 최대 answerable 질문 수")
    parser.add_argument("--max-workers", type=int, default=1, help="Latency 비교 재현성을 위해 1만 지원합니다.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/main_advanced/retrieval_comparison")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="main-advanced-9-hybrid-validation-v1")
    args = parser.parse_args()
    if args.max_workers != 1:
        parser.error("retrieval latency 비교는 --max-workers 1로 실행해야 합니다")
    items = [GoldenV3Item.from_dict(json.loads(line)) for line in args.golden.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit은 1 이상이어야 합니다")
        selected_ids = {item.question_id for item in [value for value in items if value.answerable][:args.limit]}
        items = [item for item in items if item.question_id in selected_ids]
    source_map = json.loads(args.source_map.read_text(encoding="utf-8"))
    modes = ("dense", "hybrid_rrf") if args.retriever == "all" else (args.retriever,)
    completed: dict[str, tuple[list[dict], dict]] = {}
    for mode in modes:
        output_dir = args.output_dir / mode
        summary_path, details_path = output_dir / "retrieval_summary.json", output_dir / "retrieval_details.jsonl"
        if args.resume and summary_path.is_file() and details_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            details = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            details, summary = evaluate(mode, items, source_map, args.top_k)
            output_dir.mkdir(parents=True, exist_ok=True)
            details_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details), encoding="utf-8")
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        settings = load_settings()
        config = {
            "phase": "retrieval_validation_v1", "retriever": mode, "scope": "document",
            "document_count": 9, "top_k": args.top_k, "limit": args.limit, "reranker_enabled": False,
            "golden_sha256": _sha256(args.golden), "source_map_sha256": _sha256(args.source_map),
            "collection_name": settings.get("index", "collection_name"),
            "dense_candidate_k": settings.get("retrieval", "dense_candidate_k"),
            "bm25_candidate_k": settings.get("retrieval", "bm25_candidate_k"),
            "dense_weight": settings.get("retrieval", "dense_weight"), "bm25_weight": settings.get("retrieval", "bm25_weight"),
        }
        context = init_wandb_run(job_type="retrieval-evaluation", name="dense-baseline" if mode == "dense" else "hybrid-rrf", group=args.wandb_group, tags=["main-advanced", "corpus-9", "document-filtered", mode, "no-reranker"], config=config, project_root=PROJECT_ROOT) if args.wandb else nullcontext(None)
        with context as run:
            if run is not None:
                log_wandb(run, details, summary, output_dir)
        completed[mode] = (details, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if set(completed) == {"dense", "hybrid_rrf"}:
        dense = {row["question_id"]: row for row in completed["dense"][0]}
        hybrid = {row["question_id"]: row for row in completed["hybrid_rrf"][0]}
        comparison = []
        for query_id in sorted(dense.keys() & hybrid.keys()):
            left, right = dense[query_id], hybrid[query_id]
            dense_hit, hybrid_hit = bool(left["metrics"].get("hit@5")), bool(right["metrics"].get("hit@5"))
            classification = "rescued" if not dense_hit and hybrid_hit else "regressed" if dense_hit and not hybrid_hit else "unchanged"
            comparison.append({"query_id": query_id, "document_id": left["document_id"], "question": left["question"], "dense_hit_at_5": dense_hit, "hybrid_hit_at_5": hybrid_hit, "dense_mrr": left["metrics"].get("mrr@10", 0), "hybrid_mrr": right["metrics"].get("mrr@10", 0), "mrr_delta": right["metrics"].get("mrr@10", 0) - left["metrics"].get("mrr@10", 0), "dense_latency_ms": left["latency_ms"], "hybrid_latency_ms": right["latency_ms"], "classification": classification})
        comparison_path = args.output_dir / "dense_vs_hybrid.jsonl"
        comparison_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in comparison), encoding="utf-8")
        if args.wandb:
            import wandb
            with init_wandb_run(job_type="retrieval-comparison", name="dense-vs-hybrid", group=args.wandb_group, tags=["main-advanced", "corpus-9", "comparison", "no-reranker"], config={"scope": "document", "document_count": 9, "top_k": args.top_k}, project_root=PROJECT_ROOT) as run:
                table = wandb.Table(columns=list(comparison[0]) if comparison else ["query_id"])
                for row in comparison:
                    table.add_data(*row.values())
                counts = {name: sum(row["classification"] == name for row in comparison) for name in ("rescued", "regressed", "unchanged")}
                run.log({"comparison/query_results": table, **{f"comparison/{key}_count": value for key, value in counts.items()}})
                artifact = wandb.Artifact("main-advanced-9-dense-vs-hybrid", type="retrieval-comparison")
                artifact.add_file(str(comparison_path)); run.log_artifact(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
