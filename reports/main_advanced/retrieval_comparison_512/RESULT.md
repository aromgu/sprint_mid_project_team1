# 9-document Dense vs Hybrid RRF validation (512/51 baseline)

## Scope

- Corpus: existing 9 Main Advanced documents
- Search scope: selected `document_id` only
- Golden v3 answerable queries: 95
- Final `top_k`: 5
- Dense: OpenAI embedding + Chroma MMR
- Hybrid: Dense MMR 0.3 + Kiwi BM25 0.7 + weighted RRF
- Qwen reranker: disabled
- Effective concurrency: 1 (latency comparability)

## Results

| Metric | Dense | Hybrid RRF | Delta |
|---|---:|---:|---:|
| Hit@1 | 0.6105 | 0.5684 | -0.0421 |
| Hit@3 | 0.7789 | 0.7684 | -0.0105 |
| Hit@5 | 0.8842 | 0.8105 | -0.0737 |
| MRR@10 | 0.7084 | 0.6679 | -0.0405 |
| Section recall@5 | 0.5070 | 0.5860 | +0.0790 |
| Fact coverage@5 | 0.6877 | 0.6035 | -0.0842 |
| Mean latency | 284.4 ms | 341.2 ms | +56.8 ms |
| p95 latency | 384.2 ms | 422.7 ms | +38.5 ms |

Question comparison:

- Rescued by Hybrid: 7
- Regressed under Hybrid: 14
- Unchanged: 74
- Wrong-document results: 0 for both retrievers

## Decision

Hybrid RRF does not pass the adoption gate on the current 512/51 index. Dense remains
the service default. Hybrid remains available as an evaluation mode.

The next experiment should tune Dense/BM25 weights and candidate counts, then repeat
the same evaluation on the planned 1024/102 index before considering Qwen reranking.

## W&B

- Project: `hyojin33-kim-rfp-ai/ux-rag`
- Group: `main-advanced-9-hybrid-validation-512-v1`
- Dense run: `pw7zti9j`
- Hybrid run: `es1o1gam`
- Comparison run: `rfm9rro3`

The runs include aggregate quality and latency metrics, per-query result tables,
difficulty/query-type breakdown tables, evaluation artifacts, and rescued/regressed
comparison rows.
