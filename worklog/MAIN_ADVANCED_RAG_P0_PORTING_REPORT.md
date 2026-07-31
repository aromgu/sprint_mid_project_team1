# Main Advanced RAG P0 포팅 결과

## 포팅 범위와 의존성

원본 `../sprint_mid_project_team1_main`에서 다음 실행 경계를
`src/main_rag/`로 격리했다.

```text
loader/load_documents.py
  -> preprocessing/{clean_text,table_formats,prepare_advanced}.py
  -> chunking/{split_text,advanced_chunking}.py
  -> embeddings/build_advanced_index.py
  -> retrieval/advanced_retriever.py
  -> generation/generate_answer.py
  -> service.py
```

내부 import는 모두 `src.main_rag.*`를 사용한다. Naive 모듈 import는 없다.
추가 의존성은 `pdfplumber`, `rhwp-python`, `kss`, `kiwipiepy`,
`langchain-text-splitters`이며 `pyproject.toml`과 `uv.lock`에 병합했다.

## 파일 흐름과 schema

```text
data/manifests/documents.json
  -> data/main_advanced/manifest/documents.jsonl
     (source_id=SHA-256 앞 16자, document_id=eval_XX)
  -> data/main_advanced/preprocessed/documents_advanced_v1.jsonl
     data/main_advanced/preprocessed/blocks_advanced_v1.jsonl
     data/main_advanced/preprocessed/tables_advanced_v1.jsonl
     data/main_advanced/preprocessed/images_advanced_v1.jsonl
     schema: rfp_advanced_preprocessing_v1
  -> data/main_advanced/chunks/chunks_advanced.jsonl.gz
     schema: rfp_advanced_chunk_v2
  -> data/main_advanced/chroma
     collection: ai11_policy_advanced_v2
```

모든 workspace 경로와 모델/collection/top-k 설정은
`configs/main_advanced_rag.yaml`에서 관리한다. 포팅 코드에는 `/home/data`
절대경로가 없다.

## 실행 결과

- 원본 PDF: 9개, SHA/manifest 일치
- 전처리: documents 9, blocks 9,346, tables 1,116, images 367
- 청킹: 2,850개 (text 1,130, table 1,720), 품질 gate 전체 통과
- 청크 SHA-256: `e16e323598dc1be5e33ffd7818239786034d26d9a4a7b46590f8612add3a571b`
- Dense: 2,850 입력 = 2,850 Chroma records
- embedding: `text-embedding-3-small`, 1,536 dimensions
- 전체 테스트: 46 passed

## 실행 명령

```bash
uv run python -m scripts.main_rag.prepare_manifest --overwrite
uv run python -m scripts.main_rag.run_advanced_preprocessing --overwrite
uv run python -m scripts.main_rag.run_advanced_chunking --overwrite
uv run python -m scripts.main_rag.run_advanced_indexing
uv run python scripts/run_main_advanced_rag.py \
  --document-id eval_01 \
  --question "사업 수행 기간은?"
```

마지막 명령은 `answer`, `evidence`, `page`, `chunk_id`, `confidence`,
`latency`를 반환한다.
