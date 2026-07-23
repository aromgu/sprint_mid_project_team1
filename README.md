# sprint_mid_project_team1
mid project with RAG

## Naive 데이터 파이프라인 현황

- [Naive 전처리·청킹·메타데이터 보강 수정 및 GCP 인계 정리](docs/naive_data_pipeline_handoff.md)
- 현재 완료 범위: 메타데이터 보강까지
- 임베딩·공용 Vector DB 생성: GCP 실행 대기


## Settings
작업할 환경에서 ```pip install uv```

해당 repo를 clone 한 후에, repo directory에서  ```uv sync``` 와 ```uvx prek install```  터미널에 실행.



## Project Structure

```text
rag-project/
├─ README.md
├─ .gitignore
├─ .env.example
├─ src/
│  ├─ loader/
│  │  └─ load_documents.py
│  ├─ preprocessing/
│  │  └─ clean_text.py
│  ├─ chunking/
│  │  └─ split_text.py
│  ├─ embeddings/
│  │  └─ build_embeddings.py
│  ├─ retrieval/
│  │  ├─ retriever.py
│  │  └─ reranker.py
│  ├─ generation/
│  │  └─ generate_answer.py
│  └─ evaluation/
│     └─ eval_rag.py
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ eval/
├─ scripts/
│  ├─ run_chunking.py
│  ├─ run_metadata_enrichment.py
│  ├─ run_indexing.py
│  ├─ run_rag.py
│  └─ run_eval.py
├─ notebooks/
├─ tests/
├─ docs/
└─ .github/
   ├─ ISSUE_TEMPLATE/
   └─ pull_request_template.md
```
