# Main Advanced RAG 우선순위별 할 일

> **현행 상태 (2026-07-29)**
> P0·P1·P2 구현과 통합은 완료됐다. 아래 P0/P1의 미체크 항목은 최초 계획 당시의
> 추적 목록으로, 실제 완료 여부는
> [`reports/main_advanced/P1_STATUS.md`](./reports/main_advanced/P1_STATUS.md),
> [`reports/main_advanced/P2_STATUS.md`](./reports/main_advanced/P2_STATUS.md),
> [`reports/main_advanced/TOP10_EVALUATION.md`](./reports/main_advanced/TOP10_EVALUATION.md)를
> 기준으로 판단한다. 현재 남은 항목은 이 문서 끝의 **현행 잔여 작업**만 해당한다.

## 1. 목표

`../sprint_mid_project_team1_main`의 Advanced RAG pipeline만 현재 저장소로 가져와 실행하고, 현재 평가 시스템과 기존 FastAPI/UI에 연결한다.

전체 작업은 다음 3단계로 진행한다.

```text
P0 Advanced pipeline 실행
  → P1 평가 연결
  → P2 UI 연결
```

UI부터 수정하지 않는다. 먼저 Advanced RAG가 현재 저장소에서 단독으로 정상 실행되는 상태를 만든다.

관련 설계 문서:

- [`MAIN_ADVANCED_RAG_EVALUATION_UI_PLAN.md`](./MAIN_ADVANCED_RAG_EVALUATION_UI_PLAN.md)

---

## 2. 범위

### 포함

- Main Advanced preprocessing
- Main Advanced chunking
- Main Advanced dense indexing
- Advanced runtime retriever
- `BidMateRAGSession`
- 현재 Golden set 평가
- RAGAS 및 LLM Judge
- 현재 FastAPI/backend
- 현재 Q&A 및 Workspace UI

### 제외

- Naive 평가
- Naive UI 연결
- Naive/Advanced 비교
- pipeline 선택 UI
- pipeline registry
- 개발/일반/시연 UI 분리
- Advanced BM25/hybrid runtime 정책의 임의 구현

---

# P0 — Main Advanced Pipeline 실행

## P0-1. 코드 및 의존성 이동 범위 확정

목표: Advanced 실행에 필요한 파일과 내부 import 관계를 정확히 파악한다.

우선 조사할 대상:

```text
../sprint_mid_project_team1_main/src/loader/
../sprint_mid_project_team1_main/src/preprocessing/
../sprint_mid_project_team1_main/src/chunking/advanced_chunking.py
../sprint_mid_project_team1_main/src/embeddings/build_advanced_index.py
../sprint_mid_project_team1_main/src/generation/generate_answer.py

../sprint_mid_project_team1_main/scripts/run_advanced_preprocessing.py
../sprint_mid_project_team1_main/scripts/run_advanced_chunking.py
../sprint_mid_project_team1_main/scripts/run_advanced_indexing.py
```

할 일:

- [ ] 각 파일의 내부 import 목록 작성
- [ ] 추가로 필요한 main 내부 파일 확인
- [ ] main `pyproject.toml`과 현재 의존성 비교
- [ ] Advanced 관련 기존 테스트 목록 확인
- [ ] 입력 데이터와 중간 산출물 schema 확인
- [ ] Advanced preprocessing → chunking → indexing의 정확한 파일 흐름 확인
- [ ] Naive 모듈에 의존하는 부분이 있는지 확인

산출물:

```text
Advanced 포팅 대상 파일 목록
필요 의존성 목록
입력 및 출력 파일 목록
내부 import dependency map
```

완료 조건:

- 빠진 파일 없이 포팅 범위를 설명할 수 있다.
- Naive와 공통 코드의 경계를 확인했다.
- Advanced 각 단계의 입출력 schema와 경로를 확인했다.

---

## P0-2. `src/main_rag/`로 코드 포팅

권장 구조:

```text
src/main_rag/
├─ loader/
├─ preprocessing/
├─ chunking/
│  └─ advanced_chunking.py
├─ embeddings/
│  └─ build_advanced_index.py
├─ retrieval/
│  └─ advanced_retriever.py
├─ generation/
│  └─ generate_answer.py
├─ service.py
└─ adapter.py
```

실행 스크립트:

```text
scripts/main_rag/
├─ run_advanced_preprocessing.py
├─ run_advanced_chunking.py
└─ run_advanced_indexing.py
```

할 일:

- [ ] `src/main_rag/` package 생성
- [ ] 필요한 `__init__.py` 생성
- [ ] loader 및 공통 모델 이동
- [ ] Advanced preprocessing 이동
- [ ] Advanced chunking 이동
- [ ] Advanced indexing 이동
- [ ] `BidMateRAGSession` 이동
- [ ] Advanced 실행 스크립트 이동
- [ ] import를 `src.main_rag.*`로 변경
- [ ] 알고리즘 변경 없이 packaging 변경만 했는지 검토

import 변경 예:

```text
src.loader          → src.main_rag.loader
src.preprocessing   → src.main_rag.preprocessing
src.chunking        → src.main_rag.chunking
src.embeddings      → src.main_rag.embeddings
src.generation      → src.main_rag.generation
```

완료 조건:

- 포팅된 모든 모듈을 import할 수 있다.
- 현재 저장소의 기존 module 이름과 충돌하지 않는다.
- Advanced 실행에 Naive module이 불필요하게 포함되지 않는다.

---

## P0-3. 의존성 병합

우선 확인할 의존성:

```text
pdfplumber
rhwp-python
kss
kiwipiepy
```

할 일:

- [ ] main의 정확한 dependency version 확인
- [ ] 현재 `pyproject.toml`과 버전 충돌 확인
- [ ] 필요한 dependency만 현재 `pyproject.toml`에 추가
- [ ] `uv.lock` 갱신
- [ ] 전체 테스트 실행
- [ ] LangChain/OpenAI 관련 호환성 확인

검증:

```bash
uv sync
uv run pytest
```

완료 조건:

- dependency 설치가 성공한다.
- 포팅된 Advanced module을 import할 수 있다.
- 현재 저장소의 기존 테스트가 불필요하게 깨지지 않는다.

---

## P0-4. 경로와 설정 정리

main의 `/home/data/...` 절대 경로를 제거하고 현재 저장소의 설정으로 관리한다.

신규 설정 파일:

```text
configs/main_advanced_rag.yaml
```

예시:

```yaml
paths:
  preprocessing_dir: data/main_advanced/preprocessed
  chunks: data/main_advanced/chunks/chunks_advanced.jsonl.gz
  chroma: data/main_advanced/chroma
  reports: reports/main_advanced

index:
  collection_name: ai11_policy_advanced_v2
  embedding_model: text-embedding-3-small
  batch_size: 100

retrieval:
  mode: dense
  top_k: 5

generation:
  model: gpt-5-nano
```

할 일:

- [ ] 모든 `/home/data` 경로 검색
- [ ] 입력 원본 경로 설정화
- [ ] preprocessing 출력 경로 설정화
- [ ] chunk 출력 경로 설정화
- [ ] Chroma 경로 및 collection 설정화
- [ ] report 경로 설정화
- [ ] embedding/generation model 설정화
- [ ] 기존 산출물 덮어쓰기 방지 정책 유지

완료 조건:

- 포팅 코드에 `/home/data` 하드코딩이 없다.
- 모든 입출력 위치를 YAML에서 확인할 수 있다.
- 현재 workspace 안에서 pipeline을 실행할 수 있다.

---

## P0-5. Advanced 전처리 실행

흐름:

```text
원본 문서
  → Advanced preprocessing
  → documents JSONL
  → blocks JSONL
```

할 일:

- [ ] 입력 원본 경로 확인
- [ ] Advanced preprocessing 실행
- [ ] document 수 확인
- [ ] block 수 확인
- [ ] `source_id` 확인
- [ ] page metadata 확인
- [ ] table 구조 확인
- [ ] 빈 본문 및 실패 문서 확인
- [ ] input/output hash 기록 확인
- [ ] preprocessing report 확인

완료 조건:

- 모든 대상 문서에 대한 전처리 산출물이 생성된다.
- 실패 문서와 원인이 report에 기록된다.
- 생성 결과가 Advanced chunking 입력 schema와 일치한다.

---

## P0-6. Advanced chunking 실행

흐름:

```text
Advanced documents/blocks
  → KSS/Kiwi chunking
  → chunks_advanced.jsonl.gz
```

검증 항목:

- [ ] `chunk_id`
- [ ] `raw_text`
- [ ] `embedding_text`
- [ ] `bm25_tokens`
- [ ] `token_count`
- [ ] page metadata
- [ ] source/document metadata
- [ ] 중복 chunk ID
- [ ] 빈 embedding text
- [ ] token 제한 초과
- [ ] Advanced chunk validation
- [ ] deterministic output

완료 조건:

- Advanced chunk validation을 통과한다.
- chunking report가 생성된다.
- 동일 입력으로 다시 실행했을 때 동일 결과가 생성된다.

---

## P0-7. Advanced Dense Index 생성

이번 범위에서는 BM25 runtime 연결을 보류하고 dense index만 사용한다.

```text
embedding_text
  → text-embedding-3-small
  → Advanced Chroma
```

할 일:

- [ ] Advanced dense indexing 실행
- [ ] collection 이름 확인
- [ ] embedding model 확인
- [ ] embedding dimension 확인
- [ ] 입력 chunk 수 확인
- [ ] Chroma indexed document 수 확인
- [ ] input hash 확인
- [ ] collection metadata 확인
- [ ] 재실행 시 중복 upsert 여부 확인
- [ ] 임의 query similarity search 실행

완료 조건:

- Chroma 문서 수와 입력 chunk 수가 일치한다.
- 임의 query로 similarity search가 가능하다.
- 검색 결과에 문서, 페이지, chunk ID metadata가 존재한다.

---

## P0-8. `AdvancedRetriever` 구현

신규 파일:

```text
src/main_rag/retrieval/advanced_retriever.py
```

책임:

- [ ] Advanced Chroma 지연 초기화
- [ ] query embedding
- [ ] similarity search
- [ ] `top_k` 적용
- [ ] document metadata filter
- [ ] main generation용 `list[dict]` 변환
- [ ] API key 오류 처리
- [ ] index/collection 없음 오류 처리
- [ ] 빈 검색 결과 처리

반환 형식:

```python
{
    "id": chunk_id,
    "chunk_id": chunk_id,
    "text": page_content,
    "file_nm": source_filename,
    "page": page,
    "score": score,
    "metadata": metadata,
}
```

완료 조건:

- 특정 문서만 filter하여 검색할 수 있다.
- top-k가 정상 적용된다.
- `BidMateRAGSession.build_context()`가 결과를 그대로 사용한다.
- 검색 실패를 상위 service가 처리할 수 있다.

---

## P0-9. Generation 연결

신규 파일:

```text
src/main_rag/service.py
```

실행 흐름:

```text
question
  → BidMateRAGSession.rewrite_query()
  → AdvancedRetriever.search_documents()
  → BidMateRAGSession.ask()
  → structured answer
```

할 일:

- [ ] `MainAdvancedRAGService` 구현
- [ ] `AdvancedRetriever` 주입
- [ ] `BidMateRAGSession` 주입
- [ ] query rewrite 연결
- [ ] retrieval 연결
- [ ] generation 연결
- [ ] evidence metadata 확인
- [ ] 단일 질문 CLI 작성
- [ ] 후속 질문 smoke test

완료 조건:

- 터미널에서 질문 한 개에 답변할 수 있다.
- `answer`, `evidence`, `confidence`가 반환된다.
- 후속 질문에서 query rewrite가 동작한다.
- citation이 Advanced chunk metadata를 가리킨다.

## P0 최종 완료 기준

다음 형태의 명령 하나로 Advanced RAG가 실행되어야 한다.

```bash
uv run python scripts/run_main_advanced_rag.py \
  --document-id eval_01 \
  --question "사업 수행 기간은?"
```

출력 항목:

```text
answer
evidence
page
chunk_id
confidence
latency
```

P0가 완료되기 전에는 평가와 UI를 수정하지 않는다.

---

# P1 — 현재 평가 시스템 연결

## P1-1. Retrieval Adapter

목표: Main Advanced 검색 결과를 현재 평가 형식으로 변환한다.

할 일:

- [ ] main metadata → 평가용 chunk ID 변환
- [ ] main metadata → document ID 변환
- [ ] main metadata → page 변환
- [ ] score와 rank 보존
- [ ] metadata에 없는 값은 추측하지 않음
- [ ] page numbering 기준 확인
- [ ] Golden set document ID와 main source ID mapping 확인

완료 조건:

- 현재 retrieval evaluator가 Main Advanced 결과를 읽을 수 있다.
- 원본 chunk/document/page 정보를 잃지 않는다.

---

## P1-2. Retrieval 평가 실행

확인할 지표:

- [ ] Hit@1
- [ ] Hit@3
- [ ] Hit@5
- [ ] 검색 실패 질문
- [ ] 잘못된 문서가 검색된 질문
- [ ] 평균 검색 시간
- [ ] 최대 검색 시간

Naive와 비교하지 않는다. Main Advanced 자체의 정상성과 데모 가능성을 확인한다.

---

## P1-3. Answer Batch 생성

흐름:

```text
Golden sample
  → 새 BidMateRAGSession
  → Advanced retrieval
  → Advanced generation
  → answers.jsonl
```

할 일:

- [ ] sample마다 새 session 생성
- [ ] answer 저장
- [ ] evidence 저장
- [ ] retrieval chunk ID 저장
- [ ] token/latency 저장
- [ ] 실패 sample 오류 저장
- [ ] resume 기능 확인

완료 조건:

- 이전 sample의 `previous_response_id`와 대화 요약이 다음 sample에 영향을 주지 않는다.
- 전체 Golden set answer 결과를 재현 가능하게 저장한다.

---

## P1-4. RAGAS와 LLM Judge 실행

결과 경로:

```text
reports/main_advanced/
├─ retrieval_summary.json
├─ retrieval_details.jsonl
├─ answers.jsonl
├─ answer_summary.json
├─ ragas.json
├─ ragas_details.jsonl
└─ llm_judge.json
```

할 일:

- [ ] 현재 RAGAS 입력 형식으로 변환
- [ ] RAGAS 실행
- [ ] 현재 LLM Judge 입력 형식으로 변환
- [ ] LLM Judge 실행
- [ ] citation 일치 여부 확인
- [ ] latency 결과 정리
- [ ] 실패 원인 분류

---

## P1-5. 평가 결과 검토

최소 확인 항목:

- [ ] 정답 근거 검색 여부
- [ ] 답변 정확성
- [ ] 검색 근거에 대한 충실성
- [ ] citation 일치
- [ ] 빈 답변
- [ ] hallucination
- [ ] 평균/최대 latency
- [ ] API 오류율

## P1 최종 완료 기준

다음 질문에 답할 수 있어야 한다.

```text
Advanced 검색이 정답 근거를 찾는가?
생성 답변이 해당 근거와 일치하는가?
citation이 올바른 문서와 페이지를 가리키는가?
데모 시간 내에 답변이 생성되는가?
```

심각한 문제가 있으면 UI 연결 전에 P0의 chunking/retrieval 설정을 수정한다.

---

# P2 — 기존 UI 연결

## P2-1. Q&A Adapter

Main 결과를 현재 UI response model로 변환한다.

| Main | UI |
|---|---|
| `answer` | 답변 |
| `evidence` | citations |
| `confidence` | confidence |
| `clarification_question` | caveat |
| `conflicts` | 필요 시 추가 정보 |

할 일:

- [x] answer 변환
- [x] evidence → citation 변환
- [x] document/page/chunk ID 변환
- [x] confidence 변환
- [x] clarification/caveat 변환
- [x] 현재 UI response model 검증

주의:

> Main 답변을 현재 `OpenAIRAGService`로 다시 생성하지 않는다.

---

## P2-2. `RAGClient.answer()` 연결

현재:

```text
RAGClient.answer()
  → OpenAIRAGService
  → 현재 SearchService
```

변경 후:

```text
RAGClient.answer()
  → MainAdvancedRAGService
  → AdvancedRetriever
  → BidMateRAGSession
```

데모 generation model 계획:

```text
P1 평가/재현 모델: gpt-5-nano 유지
P2 데모 runtime 모델: gemini-3.5-flash-lite
quota 또는 model access 실패 시: gpt-5-nano fallback
```

전환 이유:

- 동일 질문 partial 비교에서 Flash-Lite 평균 생성시간은 약 3.4초,
  `gpt-5-nano`는 약 29.1초였다.
- 데모에서는 긴 평가 batch 처리보다 사용자가 체감하는 첫 답변 시간이 중요하다.
- 평가 산출물의 모델을 바꾸지 않고 runtime 설정만 분리해 평가 재현성을 유지한다.

할 일:

- [x] `MainAdvancedRAGService`를 backend에 주입
- [x] 기존 Q&A 호출 교체
- [x] 데모 runtime provider/model을 `gemini`/`gemini-3.5-flash-lite`로 설정
- [ ] `GEMINI_API_KEY`, model access, 데모 예상 요청 수와 quota 확인
- [ ] Gemini 1건 smoke test로 schema/citation/한글/latency 확인
- [x] Gemini quota/access 실패 시 `gpt-5-nano` fallback 연결
- [x] P1 평가용 모델 설정과 P2 runtime 모델 설정 분리
- [x] error handling 연결
- [x] cache가 필요하면 conversation 기준으로 단순화
- [x] API response 회귀 테스트

---

## P2-3. 간단한 Conversation Session

부트캠프 데모에서는 memory session만 사용한다.

```text
conversation_id
  → BidMateRAGSession
```

할 일:

- [x] 브라우저별 conversation ID 생성/전달
- [x] session 생성
- [x] 기존 session 재사용
- [x] reset
- [x] 오래된 session 간단 정리

제외:

- Redis
- DB session store
- multi-worker session 공유

---

## P2-4. Workspace 카드 연결

현재 카드 추출 로직은 유지하고 검색기만 변경한다.

```text
RAGClient._retrieve()
  → AdvancedRetriever
```

대상:

- [x] Overview
- [x] Risks
- [x] Eligibility
- [x] Deliverables
- [x] Requirements

완료 조건:

- 카드의 검색 context가 Advanced index에서 나온다.
- 카드 evidence가 Advanced chunk metadata를 가리킨다.
- 다른 문서의 청크가 섞이지 않는다.

---

## P2-5. UI Smoke Test

대표 질문 예:

```text
사업 기간은?
참가 자격은?
필수 제출 서류는?
실격 또는 감점 조건은?
주요 요구사항은?
```

확인 항목:

- [x] 답변 표시
- [x] 문서명 표시
- [x] 페이지 표시
- [x] 인용문 표시
- [x] confidence 표시 여부
- [x] 카드 로딩
- [x] 후속 질문
- [x] session reset
- [x] 오류 메시지
- [x] 응답 시간

## P2 최종 완료 기준

- UI에 pipeline 선택기가 없다.
- UI 질문이 Main Advanced RAG를 사용한다.
- 답변과 근거가 정상적으로 표시된다.
- Workspace 카드가 Advanced 검색 결과를 사용한다.
- 대표 데모 시나리오가 중단 없이 동작한다.

## P2-6. 운영 통합 보강 (2026-07-28)

- [x] 업로드 PDF를 Advanced preprocessing/chunking에 연결
- [x] 업로드 문서 chunk를 기존 Advanced Chroma에 증분 upsert
- [x] 업로드 문서 Q&A에서 legacy fallback 없이 Advanced 검색 사용
- [x] 업로드 후 live chunk/collection count를 Evaluation에 반영
- [x] 내부 Evaluation의 legacy BM25/Hybrid 설명을 Main Advanced 단계로 교체
- [x] P1의 104개 answer/RAGAS와 Advanced retrieval 결과를 Evaluation API/UI에 연결
- [x] 격리 PDF → OpenAI embedding → Chroma → 검색 smoke 통과
- [x] 실제 브라우저 → FastAPI → Advanced Chroma → OpenAI → SSE/reset 통합 테스트 통과
- [x] Q&A와 Golden generation 검색을 Top-10으로 전환
- [x] answered / partially_answered / unanswerable 3단계 판정 적용
- [x] Top-10 Golden 답변 104개와 공식 RAGAS 104개 재평가
- [x] 내부 Evaluation을 Top-10 산출물과 3단계 판정으로 전환

---

## 현행 잔여 작업 (2026-07-29)

### 완료 및 검증

- [x] OpenAI 실제 FastAPI → Advanced RAG → 브라우저 통합 smoke
- [x] Gemini `gemini-3.5-flash-lite` 실제 1건 smoke와 model access/quota 확인
- [x] Overview·Go/NoGo 카드 → 우측 원문 연결 Playwright 회귀 테스트
- [x] Evaluation에 Top-10 104개 결과와 3단계 답변 판정 표시
- [x] Answer Relevancy 전체 평균과 정상/부분/불가 상태별 평균 분리 표시
- [x] Python 전체 테스트, UI 전체 회귀 테스트, production build

### 데모 운영 시 확인

- [ ] 데모 당일 외부 API quota와 네트워크 상태 확인
- [ ] 실제 발표 환경에서 backend/frontend 기동 후 화면 크기·프로젝터 가독성 확인

### 데모 이후 선택 개선

- [ ] 다중 질문 query decomposition으로 필수 사실 coverage 개선
- [ ] 정상 답변의 직접성을 높이는 생성 prompt 실험 후 104개 재평가
- [ ] multi-worker 배포 시 memory session을 Redis/DB로 교체
- [ ] 대용량 PDF indexing을 background job으로 전환하고 실패 재시도 UX 추가

일반 `npm run test:ui`에서는 유료 호출 테스트를 skip한다. 실제 통합 검증은 backend를
실행한 뒤 `RUN_LIVE_RAG_E2E=true npm run test:ui -- --grep '실제 FastAPI'`로 실행한다.

---

# 3. 실제 착수 순서

다음 순서를 그대로 따른다.

1. [x] Advanced 코드의 전체 import 및 파일 의존성 조사
2. [x] `src/main_rag/` scaffold 생성
3. [x] Advanced 코드 포팅
4. [x] 의존성 병합
5. [x] 절대 경로 설정화
6. [x] 기존 Main Advanced 테스트 이식 및 실행
7. [x] Advanced 전처리 실행
8. [x] Advanced chunking 실행
9. [x] Advanced dense index 생성
10. [x] `AdvancedRetriever` 구현
11. [x] `BidMateRAGSession` 연결
12. [x] 단일 질문 CLI smoke test
13. [x] Retrieval 평가 연결
14. [x] Answer batch 생성
15. [x] RAGAS/LLM Judge 연결
16. [x] 평가에서 발견된 문제 수정
17. [x] Q&A UI 연결
18. [x] Workspace 카드 연결
19. [x] 전체 UI smoke test
20. [ ] 데모 시나리오 최종 점검

---

# 4. 단계별 Gate

| Gate | 통과 조건 | 통과 전 금지 사항 |
|---|---|---|
| P0 Gate | CLI에서 Advanced 질문·답변·근거 생성 | 평가/UI 수정 금지 |
| P1 Gate | Golden/RAGAS/Judge 결과와 latency 확인 | UI 연결 금지 |
| P2 Gate | Q&A·카드·근거·후속 질문 smoke test 통과 | 데모 완료 선언 금지 |

---

# 5. 지금 바로 할 일

개발·평가·UI 통합은 완료됐다. 데모 전에는 위 **현행 잔여 작업**의
당일 API quota/네트워크와 발표 환경 가독성만 확인한다. query decomposition,
prompt 실험, 104개 재평가는 데모 완료 후의 선택 개선이다.
