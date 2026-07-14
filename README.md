# sprint_mid_project_team1
mid project with RAG

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
