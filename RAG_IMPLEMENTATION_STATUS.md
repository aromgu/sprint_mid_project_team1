# RAG 구현 현황

## 현재 상태

실제 9개 RFP PDF에 대한 ingestion과 검색 시스템이 연결되어 있다.

```text
9개 PDF
→ 페이지·표·요구사항 파싱
→ 1,275개 구조화 청크(OCR 청크 8개 포함)
→ BM25 / Dense / RRF Hybrid / Cross-encoder 검색
→ CLI·JSONL export
→ Golden set 검색 평가 인터페이스
```

## 구현 완료

- CSV와 실제 PDF 경로를 연결한 9개 문서 manifest
- 968페이지 구조 보존 파싱
- 표 행과 요구사항 ID 기반 청킹
- 1,275개 실제 청크 loader
- 공통 `SearchChunk`, `SearchResult`, `SearchFilters` 스키마
- 한국어 term + character n-gram BM25 tokenizer
- BM25 영속 캐시와 자동 무효화
- 로컬 `jhgan/ko-sroberta-multitask` Dense 검색
- 정규화 embedding `.npy` 캐시와 자동 무효화
- BM25 + Dense RRF Hybrid 검색
- 로컬 `Dongjin-kr/ko-reranker` Cross-encoder 재정렬
- 최종 검색 결과의 앞뒤 인접 청크 확장
- 문서 ID 및 콘텐츠 유형 필터
- 단일 질의 CLI
- JSONL 배치 검색 및 결과 export
- Golden set 입력 계약
- Recall@1/3/5/10, MRR@10, nDCG@10
- 문서 및 페이지 hit 평가
- 검색 latency 기록
- 실제 문서 smoke test
- 설정 기반 검색 파이프라인과 구성요소별 실행 프리셋
- `korean_ngram` / `regex` / `whitespace` BM25 tokenizer 선택
- RRF / 정규화 weighted-score fusion 선택
- 설정 해시별 BM25 캐시 분리

## 생성된 인덱스

- `data/indexes/bm25_<설정해시>.pkl`
- `data/indexes/bm25_<설정해시>.json`
- `data/indexes/dense_jhgan--ko-sroberta-multitask.npy`
- `data/indexes/dense_jhgan--ko-sroberta-multitask.json`

`chunks.jsonl`의 SHA-256이 바뀌면 캐시는 자동으로 재생성된다.
BM25 tokenizer나 파라미터가 바뀌어도 별도의 설정 해시 캐시를 사용한다.

기본값은 `Hybrid(BM25 + ko-sroberta, RRF)`이다. reranker는 Golden set에서
효과를 확인하기 전까지 기본 비활성화하며, `configs/search/` 프리셋으로
각 구성요소를 교체할 수 있다.

## 기본 실행

```bash
python -m scripts.build_search_indexes

python -m scripts.run_search \
  --query "SFR-007 예약 시스템 기능" \
  --retriever hybrid \
  --top-k 5
```

## Golden set 수령 후 실행

```bash
python -m scripts.evaluate_retrieval \
  --golden data/eval/golden_set.jsonl \
  --retriever all
```

## 현재 기본 모델에 대한 주의

`jhgan/ko-sroberta-multitask`는 CPU에서 1,275개 청크를 약 1분 30초에 최초 인덱싱할 수 있어 기능 완성을 위한 초기 모델로 선택했다. 모델의 최대 입력 길이가 짧으므로 최종 모델로 확정하지 않는다.

`nlpai-lab/KoE5`도 로컬에 캐시되어 있고 설정만 바꾸면 사용할 수 있지만, 현재 CPU 환경에서는 전체 인덱싱 시간이 지나치게 길다. Golden set이 준비되면 별도 실행으로 성능을 비교한다.

## 아직 완료되지 않은 항목

### OCR·멀티모달 추가 검수

OCR 필수 7페이지는 사용자 공간 Tesseract로 처리되어 8개 검색 청크로 반영됐다.

- 표 형태 스캔 페이지는 한국어+영어 OCR 텍스트가 반영됨
- 시스템 구성도 2페이지는 OCR됐지만 연결 관계에 대한 사람 또는 멀티모달 검수 필요
- 선택 대상인 고려대학교 조직도는 아직 검색 청크에 포함하지 않음
- 로컬 Qwen2.5-VL 캐시에는 모델 가중치가 없어 자동 이미지 설명은 미실행

### Golden set 기반 최적화

- BM25 tokenizer 최종 선택
- Dense 모델 최종 선택
- top-k와 RRF 파라미터 튜닝
- 청크 크기 재평가
- 난이도·질문 유형별 성능 비교

### 후속 검색 고도화 및 검증

- query rewriting 또는 multi-query retrieval
- Cross-encoder candidate 수와 지연시간 최적화
- 인접 청크 확장 범위 최적화

위 기능은 Golden set에서 실패 유형과 개선 효과를 확인한 뒤 추가한다.

### 답변 생성

- 근거 제한 답변 프롬프트
- 문서명·페이지·chunk ID 인용
- 답변 불가능 판정
- RAGAS 및 사람 평가

기존 `scripts/run_rag.py`와 `src/dataset.py`는 스마트워치·사내규정 데모이며 실제 RFP 시스템의 일부로 간주하지 않는다.
