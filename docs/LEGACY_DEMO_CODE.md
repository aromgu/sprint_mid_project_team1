# 기존 데모 코드와 실제 RFP 코드 구분

## 실제 RFP 시스템

다음 경로가 9개 RFP PDF를 사용하는 현재 구현이다.

- `src/ingestion/`
- `src/chunking/structured_chunker.py`
- `src/search/`
- `scripts/run_ingestion.py`
- `scripts/run_ocr.py`
- `scripts/build_search_indexes.py`
- `scripts/run_search.py`
- `scripts/evaluate_retrieval.py`
- `configs/ingestion.yaml`
- `configs/ocr.yaml`
- `configs/search.yaml`

## 기존 학습용 데모

다음 파일은 스마트워치 보증, 환불, 병가 등 하드코딩 예제를 사용하는 이전 데모다.

- `src/loader/load_documents.py`
- `src/dataset.py`
- `src/embeddings/build_embeddings.py`
- `src/retrieval/retriever.py`
- `src/generation/generate_answer.py`
- `src/evaluation/eval_rag.py`
- `scripts/run_indexing.py`
- `scripts/run_rag.py`
- `scripts/run_eval.py`
- `pipeline.py`

이 파일들은 기존 학습 기록과 호환성을 위해 삭제하지 않았다. 실제 RFP 검색 또는 성능 보고에 사용하면 안 된다. 향후 답변 생성 단계는 데모 코드를 수정하기보다 `src/search/`의 결과 스키마를 입력으로 받는 새 모듈로 구현한다.

