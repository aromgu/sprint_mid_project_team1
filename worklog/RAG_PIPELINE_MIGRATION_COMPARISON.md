# RAG Pipeline 비교 및 이식 가이드

> **상태: 폐기된 초안(Superseded)**
> 이 문서는 현재 저장소의 RAG 장점을 중심으로 통합하는 초기 가정을 바탕으로 작성되었다. 실제 목표는 `../sprint_mid_project_team1_main`의 RAG pipeline을 거의 그대로 유지하면서 현재 저장소의 평가와 UI를 연결하는 것이다. 최신 기준은 [`MAIN_RAG_UI_EVALUATION_INTEGRATION.md`](./MAIN_RAG_UI_EVALUATION_INTEGRATION.md)를 따른다.

## 1. 목적

이 문서는 `../sprint_mid_project_team1_main`의 RAG pipeline과 현재 저장소 `GIT_advan`의 RAG pipeline을 비교하고, 원본 기능을 현재 저장소로 가져오기 위해 바꿔야 할 사항과 구현 우선순위를 정리한다.

핵심 결론은 다음과 같다.

> `sprint_mid_project_team1_main`의 pipeline 전체를 복사하지 않고, 현재 저장소의 검색·평가·API 구조를 유지하면서 **HWP/HWPX 구조 보존 전처리**, **멀티턴 질의 재작성**, **대화 요약 및 누적 필드 관리** 기능만 선별적으로 이식한다.

현재 저장소는 이미 BM25, dense retrieval, hybrid fusion, 선택적 reranking, multi-query, 인용 검증, FastAPI 연동 및 여러 평가 도구를 갖추고 있다. 원본의 Chroma retriever와 터미널 실행 구조를 그대로 가져오면 기능 중복, 경로 하드코딩, 청크 스키마 불일치, 세션 격리 문제가 발생한다.

---

## 2. 전체 Pipeline 비교

| 단계 | `sprint_mid_project_team1_main` | 현재 `GIT_advan` | 이식 방향 |
|---|---|---|---|
| 문서 입력 | PDF, HWP, HWPX | PDF 중심 | HWP/HWPX가 필요할 때 loader/parser만 선별 이식 |
| 중복 처리 | SHA-256 그룹화, 대표 문서 선정, alias 보존 | manifest 기반 파일 관리 | 원본의 해시 중복 로직을 manifest 단계에 통합 |
| PDF 파싱 | `pdfplumber`, 표·블록 구조 보존 전처리 | PyMuPDF, 표 추출, OCR 필요 페이지 판정 | PyMuPDF 유지, 원본의 표 복원 품질 로직만 비교 후 채택 |
| HWP 파싱 | `rhwp-python` 기반 구조 추출 | 미지원 | parser dispatcher와 HWP parser 추가 필요 |
| 청킹 | KSS/Kiwi, 문장·표 구조 기반, token overlap | 페이지·섹션·요구사항 기반 | 공통 청크 계약을 만든 뒤 필요한 advanced chunking 기능 이식 |
| 임베딩 | OpenAI embedding + Chroma | 로컬 sentence-transformers + NumPy index | 현재 방식을 기본으로 유지, Chroma는 선택적 backend로만 검토 |
| Sparse 검색 | advanced 처리에 BM25 토큰이 있으나 메인 검색은 단순함 | BM25 구현 완료 | 현재 구현 유지 |
| Hybrid 검색 | 메인 실행 경로는 Chroma dense similarity 중심 | BM25 + dense + RRF/weighted fusion | 현재 구현 유지 |
| Reranking | 메인 경로에서 실질적으로 사용되지 않음 | cross-encoder 선택 지원 | 현재 구현 유지 및 평가 후 활성화 |
| 질의 처리 | 대화 맥락 기반 standalone query rewrite | 복합 질문 multi-query 분해 | rewrite 후 multi-query를 수행하도록 결합 |
| 생성 | 비동기 OpenAI Responses API, 멀티턴 상태, 엄격한 JSON schema | OpenAI/Gemini, Pydantic 응답, 인용 검증 | 현재 생성기에 세션 기능만 이식 |
| 인용 | 모델이 evidence의 source/page/score까지 생성 | 모델의 source label을 실제 검색 결과와 대조 | 현재 방식 유지 |
| 서비스 | 터미널 기반 `main.py` | FastAPI/backend/frontend | 현재 API 구조 유지 |
| 평가 | 별도 기본 evaluation | golden set, retrieval 평가, RAGAS, LLM judge | 현재 평가 체계 유지 |

---

## 3. 가장 중요한 차이점

### 3.1 검색 구조

원본 저장소의 실제 retriever는 import 시점에 OpenAI embedding과 Chroma를 생성한다.

- embedding model: `text-embedding-3-small`
- collection: `ai11_policy`
- persist path: `/home/data/chroma`
- 검색 방식: dense similarity
- 기본 검색 수: top 5

관련 파일:

- `../sprint_mid_project_team1_main/src/retrieval/retriever.py`
- `../sprint_mid_project_team1_main/src/embeddings/build_embeddings.py`

현재 저장소의 `SearchService`는 설정 파일을 기반으로 다음 기능을 제공한다.

- BM25
- 로컬 dense embedding
- RRF 또는 weighted hybrid fusion
- 선택적 cross-encoder reranking
- document/content type filter
- 이웃 청크 context expansion
- index cache 및 재사용

관련 파일:

- `src/search/service.py`
- `src/search/bm25.py`
- `src/search/dense.py`
- `src/search/hybrid.py`
- `src/search/reranker.py`
- `configs/search.yaml`

따라서 원본의 `search_documents()`와 Chroma 초기화 코드는 복사하지 않는다. 원본의 세션/생성 기능이 현재의 `SearchService.search()` 결과를 사용하도록 변경해야 한다.

### 3.2 청크 스키마

원본 생성기는 대체로 다음 형태를 기대한다.

```text
text
file_nm
score
metadata.page
metadata.chunk_id
```

현재 검색 결과는 `SearchResult`와 `SearchChunk` 객체로 구성된다.

```text
SearchResult.score
SearchResult.context_text
SearchResult.chunk.text
SearchResult.chunk.document_id
SearchResult.chunk.document_title
SearchResult.chunk.page_start
SearchResult.chunk.page_end
SearchResult.chunk.chunk_id
SearchResult.chunk.requirement_ids
```

필드 대응은 다음과 같다.

| 원본 필드 | 현재 필드 |
|---|---|
| `text` | `result.context_text or result.chunk.text` |
| `file_nm` | `result.chunk.document_title` |
| `score` | `result.score` |
| `page` | `result.chunk.page_start` |
| `chunk_id` | `result.chunk.chunk_id` |
| `source_id` | `result.chunk.document_id` |
| 페이지 범위 | `page_start`, `page_end` |
| 요구사항 ID | `requirement_ids` |

단기적으로 adapter를 만들 수 있지만, 장기적으로는 원본의 세션 기능이 `list[dict]` 대신 `list[SearchResult]`를 직접 받도록 변경하는 것이 좋다.

### 3.3 질의 처리 방식

원본은 최근 대화를 보고 짧은 후속 질문을 standalone query로 재작성한다.

```text
이전 질문: 이 사업의 예산은?
후속 질문: 기간은?
재작성 결과: 해당 RFP 사업의 수행 기간은?
```

현재 저장소는 한 질문을 예산, 기간, 요구사항 등의 여러 검색 질문으로 분해한다. 두 기능은 대체 관계가 아니라 순차적으로 결합해야 한다.

권장 흐름:

```text
사용자 질문과 대화 이력
  → standalone query rewrite
  → rewritten query의 multi-query 분해
  → 각 query로 hybrid search
  → 결과 merge 및 boost
  → context 구성
  → 답변 생성과 인용 검증
```

### 3.4 세션 상태

원본의 `BidMateRAGSession`은 객체 내부에 다음 값을 저장한다.

- `previous_response_id`
- `recent_messages`
- `conversation_summary`
- `last_rewritten_query`
- `collected_fields`

터미널 단일 사용자 실행에는 적합하지만, FastAPI 서버의 전역 객체로 사용하면 사용자 간 대화가 섞일 수 있다.

현재 backend는 요청에 포함된 `chat_history`를 생성기로 전달한다. 이 구조를 유지하면서 다음 중 하나를 선택해야 한다.

1. MVP: frontend가 최근 `chat_history`를 계속 전달한다.
2. 확장: `conversation_id`별로 대화 요약과 누적 필드를 서버 저장소에 보관한다.

`previous_response_id`를 하나의 전역 session 객체에 저장해서는 안 된다.

### 3.5 출력과 인용 검증

원본의 응답에는 다음 정보가 포함된다.

- answer
- summary
- fields
- evidence
- confidence
- needs_clarification
- clarification_question
- conflicts

현재 저장소는 Pydantic 모델과 `[S1]` 형태의 source label을 사용한다. 모델이 반환한 label이 실제 검색 결과에 존재할 때만 citation을 생성한다.

현재의 인용 검증 방식이 더 안전하므로 유지한다. 특히 page, chunk ID, document ID, score를 LLM이 직접 생성하게 하지 않고 코드가 `SearchResult`에서 채워야 한다.

원본에서 가져올 항목은 `summary`, `confidence`, clarification, conflicts, extracted fields이며, evidence 생성 방식은 가져오지 않는다.

### 3.6 문서 포맷 지원

원본 loader는 다음 형식을 지원한다.

- PDF
- HWP
- HWPX
- SHA-256 기반 중복 탐지
- 대표 원본 선택
- filename/path alias 보존

현재 `DocumentManifest.resolved_pdf_path()`와 PDF parser는 PDF를 전제로 한다. HWP/HWPX 지원이 필요하면 ingestion 추상화부터 변경해야 한다.

### 3.7 실행 경로

현재 저장소에는 실제 pipeline과 맞지 않는 legacy 파일이 남아 있다.

- `pipeline.py`
- `scripts/run_rag.py`
- `src/generation/generate_answer.py`
- `configs/config.py`

이 파일들은 현재 존재하지 않는 다음 구성 요소를 import한다.

- `src.loader`
- `src.retrieval`
- `build_vector_store`
- `build_retrievers`
- `ProductionRetriever`

실제 운영 경로는 다음이다.

```text
SearchService
  → OpenAIRAGService
  → backend/services/rag_client.py
  → FastAPI router
```

원본 기능을 이식하기 전에 legacy entrypoint를 현재 pipeline 기준으로 정리해야 한다.

---

## 4. 바꿔야 할 것

### 4.1 Legacy entrypoint 교체

대상:

- `pipeline.py`
- `scripts/run_rag.py`
- `src/generation/generate_answer.py`
- `configs/config.py`

변경 내용:

- `src.loader`, `src.retrieval` import 제거
- `SearchService`와 `OpenAIRAGService` 사용
- 더 이상 필요 없는 LangChain 기반 `AdvancedRAGChain` 제거 또는 문서용 legacy 코드로 이동
- CLI와 API가 동일한 generation service를 사용하도록 통일

예상 CLI 구조:

```python
from src.generation.openai_generator import OpenAIRAGService


def main():
    service = OpenAIRAGService()
    question = input("질문을 입력하세요: ").strip()
    result = service.answer(question)
    print(result.answer)
    for citation in result.citations:
        print(citation)
```

### 4.2 Standalone query rewrite 추가

대상:

- `src/generation/openai_generator.py`
- 필요 시 신규 `src/generation/query_rewriter.py`

변경 내용:

- `question`과 최근 `chat_history`를 받아 독립 검색 질의 생성
- 첫 질문이거나 이미 충분히 구체적인 질문이면 LLM 호출 생략
- rewrite 실패 시 원 질문으로 fallback
- rewritten query를 기존 `plan_search_questions()`에 전달
- 원 질문은 최종 답변 생성에 그대로 유지

### 4.3 대화 요약 및 필드 누적

대상:

- `backend/services/state_service.py`
- `backend/services/rag_client.py`
- `src/generation/openai_generator.py`
- backend request/response model

변경 내용:

- `conversation_id` 기준 상태 분리
- 일정 길이를 넘은 과거 대화를 요약
- 최근 N개 메시지와 요약을 함께 rewrite/generation에 전달
- 사업명, 기관명, 예산, 기간, 마감일 등 누적 필드 저장
- reset API 또는 reset action 지원
- 동시 사용자 간 상태가 섞이지 않는 테스트 추가

### 4.4 응답 모델 확장

대상:

- `src/generation/models.py`
- backend response model 및 router
- frontend가 새 필드를 사용할 경우 UI model

추가 후보:

```text
summary
confidence
needs_clarification
clarification_question
conflicts
extracted_fields
rewritten_query
```

주의사항:

- `citations`는 현재 코드 검증 방식을 유지한다.
- 검색 점수는 모델이 만들지 않고 검색 결과에서 가져온다.
- confidence는 검색 점수와 동일한 값으로 취급하지 않는다.

### 4.5 공통 청크 계약 확장

대상:

- `src/ingestion/models.py`
- `src/search/models.py`
- `src/search/loader.py`
- ingestion/chunking script 및 fixture

권장 필드:

```text
필수:
- chunk_id
- document_id
- document_title
- page_start
- page_end
- text
- token_count
- content_type

선택:
- retrieval_text
- section_path
- requirement_ids
- source_filename
- source_sha256
- file_type
- table_id
- block_ids
- quality_flags
- ocr_applied
- bm25_tokens
```

사용 원칙:

- dense embedding: `retrieval_text or text`
- generation 및 citation quote: `text`
- BM25: `bm25_tokens`가 있으면 사용하고, 없으면 현재 tokenizer 사용
- 선택 필드가 없는 기존 JSONL도 읽을 수 있도록 하위 호환 유지

### 4.6 HWP/HWPX ingestion 추가

HWP/HWPX가 제품 범위에 포함될 때만 수행한다.

대상:

- `src/ingestion/models.py`
- 신규 `src/ingestion/hwp_parser.py`
- 신규 또는 기존 parser dispatcher
- `scripts/run_ingestion.py`
- `pyproject.toml`

변경 내용:

- `resolved_pdf_path()`를 `resolved_source_path()`로 일반화
- manifest에 `source_path`, `file_type` 추가
- 기존 `pdf_path` 입력에 대한 하위 호환 제공
- 파일 형식별 parser dispatcher 추가
- `rhwp-python`, 필요 시 `pdfplumber`, `kss`, `kiwipiepy` 의존성 추가
- 표, 이미지 참조, 원본 block provenance를 공통 모델에 보존
- PDF/HWP/HWPX fixture 및 회귀 테스트 추가

### 4.7 Advanced chunking 선별 이식

원본의 다음 기능을 평가 후 선별 이식한다.

- KSS 문장 경계
- token 기준 chunk size와 overlap
- 짧은 마지막 청크 보정
- 표 행/셀 구조 보존
- 실제 PDF page provenance
- `retrieval_text`와 표시용 원문 분리
- Kiwi BM25 token 재사용

현재의 다음 기능은 유지한다.

- section path
- requirement ID 탐지
- content type
- OCR 적용 여부
- page range

새 chunker로 전환하기 전에 동일 문서 집합으로 retrieval evaluation을 비교해야 한다.

### 4.8 설정 외부화

원본의 하드코딩 경로를 가져오지 않는다.

다음 값은 YAML 또는 환경변수로 관리한다.

- embedding backend
- embedding model
- index directory
- Chroma collection name
- top_k
- context character/token budget
- rewrite model 및 활성화 여부
- conversation summary threshold
- reranker 활성화 여부

Chroma를 선택적 backend로 추가할 경우 예시는 다음과 같다.

```yaml
dense:
  backend: sentence_transformers
  model: jhgan/ko-sroberta-multitask

chroma:
  persist_directory: data/indexes/chroma
  collection_name: rfp_chunks
  embedding_model: text-embedding-3-small
```

---

## 5. 구현 우선순위

### P0 — 즉시 처리: pipeline 기준선 정리

목표: 깨진 실행 경로를 제거하고 모든 RAG 호출이 현재 운영 pipeline을 사용하게 한다.

작업:

1. `pipeline.py`를 현재 pipeline 진단 스크립트로 교체하거나 제거한다.
2. `scripts/run_rag.py`를 `OpenAIRAGService` 기반 CLI로 교체한다.
3. `src/generation/generate_answer.py`의 legacy LangChain chain을 제거하거나 명확히 격리한다.
4. `configs/config.py`와 YAML 설정의 중복을 제거한다.
5. CLI, backend, 평가 script가 동일한 `SearchService`와 generation service를 사용하는지 확인한다.

완료 조건:

- 존재하지 않는 `src.loader` 및 `src.retrieval` import가 없다.
- CLI에서 현재 index를 사용해 검색과 답변을 수행할 수 있다.
- 기존 search/generation/backend 테스트가 통과한다.

### P1 — 최우선 기능 이식: 멀티턴 검색 품질

목표: 후속 질문이 대화 문맥을 반영해 올바른 문서를 검색하도록 한다.

작업:

1. standalone query rewriter를 구현한다.
2. rewrite 결과를 기존 multi-query planner 앞에 연결한다.
3. rewrite skip heuristic과 실패 fallback을 구현한다.
4. `rewritten_query`를 로그 또는 응답 metadata에 남긴다.
5. 후속 질문 retrieval 테스트를 추가한다.

완료 조건:

- “그 사업의 기간은?” 같은 질문이 독립 검색 질의로 변환된다.
- 변환된 질의가 multi-query/hybrid search에 사용된다.
- 첫 질문이나 독립적인 질문에는 불필요한 rewrite 호출을 줄인다.
- rewrite 실패 시 원 질문으로 정상 검색한다.

### P2 — 세션 및 응답 강화

목표: 멀티턴 상태를 사용자별로 안전하게 관리하고 원본의 유용한 structured output을 수용한다.

작업:

1. `conversation_id`별 상태 관리 구조를 정의한다.
2. 과거 대화 요약 기능을 추가한다.
3. collected RFP fields 저장 및 병합 규칙을 구현한다.
4. summary, confidence, clarification, conflicts 필드를 응답 모델에 추가한다.
5. reset과 세션 격리 테스트를 추가한다.
6. citation은 현재 source-map 검증 방식을 유지한다.

완료 조건:

- 서로 다른 conversation의 대화와 필드가 섞이지 않는다.
- 긴 대화에서도 검색용 문맥 크기가 제한된다.
- 응답의 citation이 실제 검색 청크와 일치한다.
- 모델이 만든 허위 page/chunk/score가 응답에 사용되지 않는다.

### P3 — 청킹 및 메타데이터 품질 개선

목표: 원본 advanced preprocessing/chunking의 장점을 현재 검색 구조에 반영한다.

작업:

1. 공통 청크 스키마에 선택 필드를 추가한다.
2. `retrieval_text`와 인용용 `text`를 분리한다.
3. 문장 경계 및 표 구조 기반 chunking 실험을 추가한다.
4. requirement ID, section path, page provenance를 보존한다.
5. 기존 chunker와 advanced chunker를 동일 golden set으로 비교한다.

완료 조건:

- 기존 JSONL과 새 JSONL을 모두 읽을 수 있다.
- citation quote에는 원문이 사용된다.
- retrieval hit rate 또는 MRR/NDCG가 기준선보다 개선되거나 최소한 하락하지 않는다.
- 청크 수, index 크기, 검색 latency 증가가 허용 범위 안에 있다.

### P4 — 선택 기능: HWP/HWPX 지원

목표: 제품 범위에 HWP/HWPX가 포함될 때 다중 포맷 ingestion을 제공한다.

작업:

1. manifest와 parser interface를 포맷 중립적으로 변경한다.
2. SHA-256 중복 탐지와 canonical 선택을 통합한다.
3. HWP/HWPX parser를 추가한다.
4. 표, 이미지 reference, block provenance를 보존한다.
5. 포맷별 회귀 테스트를 추가한다.

완료 조건:

- PDF, HWP, HWPX가 동일한 공통 Page/Chunk 계약으로 변환된다.
- 중복 원본이 하나의 문서로 잘못 중복 인덱싱되지 않는다.
- 원본 파일 경로와 출처 메타데이터를 추적할 수 있다.

### P5 — 선택 기능: Chroma/OpenAI embedding backend

목표: 현재 로컬 dense backend와 OpenAI/Chroma backend를 객관적으로 비교한다.

작업:

1. Chroma backend를 설정 기반 선택 기능으로 구현한다.
2. import 시점 초기화와 절대 경로를 제거한다.
3. 현재 dense, Chroma dense, hybrid 조합을 golden set으로 평가한다.
4. 품질, 비용, index 시간, query latency를 비교한다.

완료 조건:

- backend를 설정만으로 전환할 수 있다.
- API key가 없을 때 로컬 backend가 정상 동작한다.
- 품질 또는 운영상 이점이 입증된 경우에만 기본 backend 변경을 검토한다.

---

## 6. 우선순위 요약

| 우선순위 | 작업 | 반드시 필요한가 | 이유 |
|---|---|---|---|
| P0 | legacy pipeline/entrypoint 정리 | 필수 | 현재 구조와 맞지 않는 import 및 중복 실행 경로 제거 |
| P1 | standalone rewrite + multi-query 결합 | 필수 | 원본에서 가져올 가치가 가장 큰 멀티턴 검색 개선 |
| P2 | conversation 상태, 요약, structured response | 중요 | 실제 서비스의 멀티턴 안정성과 사용자 격리 확보 |
| P3 | 공통 청크 계약 및 advanced chunking 실험 | 중요 | 표·문장·출처 보존과 검색 품질 개선 |
| P4 | HWP/HWPX ingestion | 조건부 | 입력 문서 범위에 HWP가 있을 때만 필요 |
| P5 | Chroma/OpenAI embedding | 선택 | 현재 hybrid retrieval을 대체할 근거가 있을 때만 도입 |

---

## 7. 권장 최종 Architecture

```text
PDF/HWP/HWPX source
  → format-neutral manifest
  → SHA-256 duplicate/canonical selection
  → parser dispatcher
      ├─ PyMuPDF PDF parser + OCR
      └─ rhwp HWP/HWPX parser
  → normalized PageRecord/block records
  → section/requirement/table-aware chunker
  → common ChunkRecord
      ├─ display/citation text
      └─ retrieval_text
  → indexes
      ├─ BM25
      └─ dense embedding
  → standalone query rewrite
  → multi-query planning
  → hybrid retrieval
  → optional reranking
  → neighbor context expansion
  → OpenAI/Gemini structured generation
  → source-map citation validation
  → conversation summary/field update
  → FastAPI/backend/frontend
```

---

## 8. 가져오지 말아야 할 것

다음 코드는 그대로 복사하지 않는다.

- `/home/data/chroma` 같은 절대 경로
- `ai11_policy` 같은 고정 collection name
- import 시점 OpenAI client/embedding/vectorstore 초기화
- `list[dict]`에 의존하는 느슨한 검색 결과 계약
- 단일 전역 `BidMateRAGSession`
- LLM이 page, chunk ID, score를 직접 작성하는 evidence
- 현재 BM25/hybrid/reranker를 우회하는 Chroma-only 검색 경로
- 현재 평가 및 backend 구조와 별개인 두 번째 RAG pipeline

---

## 9. 구현 전 확인할 결정 사항

구현 범위를 확정하기 전에 다음을 결정해야 한다.

1. 실제 운영 입력에 HWP/HWPX가 포함되는가?
2. conversation 상태를 frontend가 전달할지, backend가 저장할지?
3. 현재 로컬 dense model을 운영 기본값으로 유지할지?
4. advanced chunker 평가의 성공 기준을 무엇으로 할지?
5. 응답에 누적 필드를 노출할지, 내부 상태로만 사용할지?

권장 기본 결정은 다음과 같다.

- HWP/HWPX는 실제 요구가 확인될 때 P4로 수행한다.
- 초기에는 frontend `chat_history`를 유지하고 backend에는 요약만 선택적으로 저장한다.
- 현재 BM25+dense hybrid를 기본 검색기로 유지한다.
- advanced chunking은 golden set retrieval 점수가 개선될 때만 기본값으로 전환한다.
- citation의 출처 정보는 항상 코드가 검색 결과에서 생성한다.

---

## 10. 최종 권고

현재 저장소의 **검색, reranking, 인용 검증, 평가, FastAPI 구조는 유지**한다. 원본 저장소에서는 다음 기능만 우선적으로 가져온다.

1. 대화 맥락 기반 standalone query rewrite
2. 긴 대화 요약과 RFP 핵심 필드 누적
3. 필요 시 HWP/HWPX 구조 보존 parsing
4. 평가로 효과가 확인된 advanced chunking 기능

가장 먼저 P0에서 현재의 깨진 legacy 실행 경로를 정리하고, P1에서 query rewrite와 기존 multi-query를 결합한다. 이후 P2의 세션 격리와 응답 확장을 수행한다. HWP/HWPX 및 Chroma 도입은 제품 요구와 평가 결과가 확인된 뒤 진행한다.
