# Main RAG Pipeline 유지 및 평가·UI 통합 계획

> **상태: 폐기된 확장안(Superseded)**
> 이 문서는 Naive와 Advanced pipeline을 모두 연결하는 확장안을 설명한다. 최종 범위에서는 Naive를 평가하거나 UI에 연결하지 않고 Main Advanced만 사용한다. 최신 기준은 [`MAIN_ADVANCED_RAG_EVALUATION_UI_PLAN.md`](./MAIN_ADVANCED_RAG_EVALUATION_UI_PLAN.md)를 따른다.

## 1. 목표

이 문서의 목표는 `../sprint_mid_project_team1_main`의 RAG pipeline을 기준 구현으로 거의 그대로 유지하면서, 현재 `GIT_advan` 저장소의 평가 체계와 FastAPI/UI를 연결하는 방법을 정의하는 것이다.

핵심 원칙은 다음과 같다.

> **Main RAG pipeline은 기준 구현으로 보존하고, 현재 저장소는 실행 환경, 평가 harness, API 및 UI shell 역할을 담당한다.**

현재 저장소의 BM25, local dense, hybrid, multi-query 등을 main pipeline에 합치는 것이 목적이 아니다. main pipeline의 검색 속도가 더 느리더라도 부트캠프 팀이 구현한 구조와 수준, 실험 결과 및 동작을 보존하는 것을 우선한다.

---

## 2. 목표 Architecture

```text
../sprint_mid_project_team1_main RAG
  ├─ loader
  ├─ preprocessing
  ├─ advanced chunking
  ├─ OpenAI embedding
  ├─ Chroma
  ├─ retriever
  ├─ query rewrite
  └─ BidMateRAGSession
          ↓
    얇은 Adapter 계층
       ├─ Retrieval Adapter
       ├─ Generation Adapter
       ├─ Evaluation Adapter
       └─ UI Adapter
          ↓
현재 GIT_advan
  ├─ Golden set / Retrieval 평가
  ├─ RAGAS / LLM Judge
  ├─ FastAPI backend
  └─ Frontend UI
```

### 통합 원칙

- main의 RAG 알고리즘은 최대한 수정하지 않는다.
- 현재 `SearchService`로 main retriever를 대체하지 않는다.
- main의 preprocessing, chunking, Chroma index를 그대로 사용한다.
- 평가와 UI가 기대하는 데이터 형식만 adapter에서 변환한다.
- main 코드는 현재 저장소 안의 독립 namespace로 복사한다.
- UI 카드처럼 main에 없는 기능은 application adapter로 처리한다.
- 현재 RAG는 삭제하지 않고 비교 평가용 baseline으로 남긴다.

---

## 3. Main에서 그대로 유지할 Pipeline

### 3.1 Ingestion

```text
PDF/HWP/HWPX
  → SourceDocument loader
  → SHA-256 중복 확인
  → canonical document 선택
  → 구조 보존 preprocessing
```

유지할 기능:

- PDF, HWP, HWPX 입력
- SHA-256 기반 중복 그룹화
- canonical source 선택
- filename/path alias 보존
- 표, 이미지, block provenance 처리

### 3.2 Chunking

```text
구조 보존 block
  → KSS 문장 분리
  → Kiwi 처리
  → token 기반 chunk
  → overlap
  → retrieval_text 생성
```

유지할 기능:

- KSS 문장 경계
- Kiwi 기반 처리
- token 크기 제한
- overlap
- 짧은 tail chunk 보정
- 표 구조와 page provenance 보존
- 검색용 `retrieval_text`와 원문 분리

### 3.3 Indexing

```text
chunks JSONL
  → text-embedding-3-small
  → Chroma collection
```

유지할 기능:

- OpenAI embedding
- main의 chunk metadata contract
- Chroma collection
- indexing audit/report
- main의 advanced index가 실제 사용 가능하면 해당 경로 포함

### 3.4 Retrieval

```text
질문
  → OpenAI query embedding
  → Chroma similarity search
  → top-k 결과
```

main retriever를 현재의 BM25/hybrid retriever로 교체하지 않는다. main에 reranking 경로가 있고 정상 동작한다면 main 구현 범위 안에서 활성화한다.

### 3.5 Generation과 Multi-turn

```text
대화 이력
  → standalone query rewrite
  → main retriever
  → BidMateRAGSession.ask()
  → structured answer/evidence
```

유지할 기능:

- `previous_response_id`
- `recent_messages`
- `conversation_summary`
- `last_rewritten_query`
- `collected_fields`
- token budget
- context deduplication
- clarification/conflict 응답
- main prompt와 JSON schema

---

## 4. 현재 저장소에서 사용할 부분

### 4.1 평가

- Golden set retrieval 평가
- Hit@K, MRR, NDCG 등 retrieval metric
- Answer generation batch 실행
- RAGAS
- LLM Judge
- latency 및 비용 측정
- 결과 report 저장 및 비교

### 4.2 Backend와 UI

- FastAPI application
- backend router와 response model
- frontend Q&A 화면
- Overview, Risks, Eligibility, Deliverables, Requirements 카드
- 문서별 workspace
- UI regression test

### 4.3 비교 Baseline

현재 RAG는 운영 기본 경로에서 제외할 수 있지만 삭제하지 않는다.

```text
main_rag
  = 부트캠프 팀 pipeline
  = 운영/시연 기준 후보

advan_rag
  = 현재 저장소 pipeline
  = 비교 평가 baseline
```

설정으로 backend를 선택할 수 있게 한다.

```yaml
rag:
  backend: main
```

또는:

```bash
RAG_BACKEND=main
```

---

## 5. 현재 운영 RAG 경로에서 빠지는 기능

main을 기본 backend로 선택하면 다음 모듈은 UI Q&A의 운영 검색 경로에서 호출되지 않는다.

```text
src/search/bm25.py
src/search/dense.py
src/search/hybrid.py
src/search/reranker.py
src/search/query_planning.py
src/search/service.py
src/generation/openai_generator.py
```

이 파일들은 다음 목적으로 유지한다.

- 기존 테스트 보존
- main과 현재 pipeline 비교 평가
- fallback 또는 실험용 backend
- 성능 차이 분석

---

## 6. 권장 파일 구조

main 코드를 기존 `src/search`와 `src/generation` 위에 덮어쓰지 않는다. 다음과 같이 별도 namespace로 배치한다.

```text
src/
├─ main_rag/
│  ├─ loader/
│  │  └─ load_documents.py
│  ├─ preprocessing/
│  │  ├─ clean_text.py
│  │  ├─ prepare_advanced.py
│  │  └─ table_formats.py
│  ├─ chunking/
│  │  ├─ split_text.py
│  │  └─ advanced_chunking.py
│  ├─ embeddings/
│  │  ├─ build_embeddings.py
│  │  └─ build_advanced_index.py
│  ├─ retrieval/
│  │  ├─ retriever.py
│  │  └─ reranker.py
│  ├─ generation/
│  │  └─ generate_answer.py
│  ├─ adapters/
│  │  ├─ retrieval_adapter.py
│  │  ├─ generation_adapter.py
│  │  ├─ evaluation_adapter.py
│  │  └─ ui_adapter.py
│  ├─ config.py
│  └─ service.py
├─ search/                    # 현재 baseline 유지
├─ generation/                # 현재 baseline 유지
└─ evaluation/                # 공통 평가
```

복사된 main 모듈의 import 경로만 namespace에 맞게 변경한다.

```text
src.loader          → src.main_rag.loader
src.preprocessing   → src.main_rag.preprocessing
src.chunking        → src.main_rag.chunking
src.embeddings      → src.main_rag.embeddings
src.retrieval       → src.main_rag.retrieval
src.generation      → src.main_rag.generation
```

이는 알고리즘 변경이 아니라 기존 코드와의 이름 충돌을 막는 packaging 변경이다.

---

## 7. 반드시 바꿔야 할 것

## 7.1 절대 경로를 설정으로 이동

main의 다음 절대 경로는 현재 저장소에서 그대로 사용할 수 없다.

```text
/home/data/chunks/chunks_v1.jsonl.gz
/home/data/chroma
/home/data/reports/...
```

현재 저장소 안의 경로로 설정한다.

```yaml
main_rag:
  chunks_path: data/main_rag/chunks/chunks_v1.jsonl.gz
  chroma_path: data/main_rag/chroma
  reports_path: reports/main_rag
  collection_name: ai11_policy
  embedding_model: text-embedding-3-small
  retrieval_top_k: 5
```

`ai11_policy` collection 이름은 실험 재현을 위해 기본값으로 유지할 수 있지만 코드에 하드코딩하지 않는다.

## 7.2 의존성과 Import 경로 병합

main에서 필요한 의존성을 현재 `pyproject.toml`에 추가한다.

주요 후보:

```text
pdfplumber
rhwp-python
kss
kiwipiepy
```

버전은 main의 lock/config를 우선 기준으로 삼되 현재 의존성과 충돌하는지 확인한다. 복사된 코드 내부 import는 `src.main_rag.*`로 변경한다.

## 7.3 Retriever를 지연 초기화 가능한 객체로 감싸기

main retriever는 import 시점에 OpenAI embedding과 Chroma를 초기화한다. FastAPI와 테스트에서 사용할 수 있도록 검색 알고리즘은 유지하고 생성 시점만 변경한다.

```python
class MainRAGRetriever:
    def __init__(
        self,
        persist_directory,
        collection_name,
        embedding_model,
        api_key,
    ):
        self.vectorstore = Chroma(...)

    def search_documents(
        self,
        query: str,
        k: int = 5,
        document_id: str | None = None,
    ) -> list[dict]:
        ...
```

필요한 이유:

- API key가 없는 테스트에서 backend import 실패 방지
- 테스트 collection/client 주입
- index 경로와 collection 전환
- 평가 sample마다 설정 재현
- UI의 문서별 filter 지원

Chroma similarity 계산과 결과 포맷은 main 구현을 유지한다.

## 7.4 문서별 Metadata Filter 추가

현재 UI는 특정 `document_id`의 workspace에서 질문한다. main collection 전체를 검색하면 다른 문서의 청크가 섞일 수 있다.

```python
vectorstore.similarity_search_with_relevance_scores(
    query,
    k=k,
    filter={"source_id": document_id},
)
```

실제 filter key는 main indexing metadata를 확인해 다음 중 정확한 값을 사용한다.

```text
source_id
document_id
bid_notice_id
file_nm
```

이는 검색 알고리즘 변경이 아니라 UI가 지정한 문서 범위를 적용하는 변경이다.

## 7.5 Retrieval Adapter 추가

main retriever 반환 형식:

```python
{
    "id": "...",
    "text": "...",
    "file_nm": "...",
    "score": 0.9,
    "metadata": {...},
}
```

평가 및 UI 공통 형식으로 변환한다.

```python
{
    "chunk_id": metadata["chunk_id"],
    "document_id": metadata["source_id"],
    "document_title": metadata["source_filename"],
    "page_start": metadata["page"],
    "page_end": metadata["page"],
    "text": result["text"],
    "score": result["score"],
    "rank": rank,
    "retriever": "main-chroma",
}
```

adapter는 값을 새로 추론하지 않고 main metadata를 필드 이름만 바꿔 전달한다.

## 7.6 Generation Adapter 추가

main `BidMateRAGSession.ask()` 결과:

```text
answer
summary
fields
evidence
confidence
needs_clarification
clarification_question
conflicts
```

현재 UI 답변 모델의 주요 필드:

```text
question
answer
is_answerable
caveat
citations
retrieved_chunk_ids
retriever
model
latency
```

필드 대응:

| main 결과 | 현재 UI 결과 |
|---|---|
| `answer` | `answer` |
| `summary` | UI 확장 `summary` |
| `fields` | `extracted_fields` 또는 conversation state |
| `evidence` | `citations` |
| `confidence` | UI 확장 `confidence` |
| `needs_clarification` | `is_answerable` 판단에 반영 |
| `clarification_question` | 별도 필드 또는 `caveat` |
| `conflicts` | UI 확장 `conflicts` |

main의 결과를 현재 생성기로 다시 생성하면 안 된다.

```text
잘못된 흐름:
main 검색 → 현재 OpenAIRAGService가 답변 재생성

올바른 흐름:
main 검색 → BidMateRAGSession.ask() → UI adapter가 형식만 변환
```

## 7.7 UI Session Registry 추가

`BidMateRAGSession`은 상태를 가지므로 UI conversation별 객체가 필요하다.

```python
class MainRAGSessionRegistry:
    sessions: dict[str, BidMateRAGSession]

    def get(self, conversation_id: str) -> BidMateRAGSession:
        ...

    def reset(self, conversation_id: str) -> None:
        ...
```

실행 흐름:

```text
UI conversation_id
  → 해당 BidMateRAGSession 조회
  → session.rewrite_query()
  → main retriever 검색
  → session.ask()
  → UI adapter
```

부트캠프 시연 수준에서는 단일 worker의 memory registry로 시작할 수 있다.

제약:

- backend 재시작 시 session 소실
- 여러 worker 사이에 상태가 공유되지 않음
- 오래된 session 정리 정책 필요

여러 worker 또는 장기 운영이 필요하면 Redis나 DB 기반 session store로 교체한다.

---

## 8. UI 기능 연결 방법

## 8.1 Q&A

Q&A는 main generation을 그대로 사용한다.

```text
RAGClient.answer()
  → MainRAGService.answer()
  → conversation session 조회
  → main rewrite_query()
  → main Chroma retrieval
  → BidMateRAGSession.ask()
  → Generation UI Adapter
```

Q&A의 검색과 생성 모두 main pipeline을 거쳐야 한다.

## 8.2 Workspace 카드

현재 UI에는 다음 카드가 있다.

- Overview
- Risks
- Eligibility
- Deliverables
- Requirements

main의 Q&A schema만으로 모든 카드 schema를 직접 만들 수는 없다. 카드 기능은 다음처럼 연결한다.

```text
UI 카드 요청
  → 카드별 현재 고정 query
  → main Chroma retriever
  → 검색 context
  → 현재 카드 Pydantic schema extraction
  → main 검색 결과 기반 evidence 연결
```

현재 `RAGClient`의 카드 extraction 로직은 유지하고 `_retrieve()`만 main retrieval adapter로 교체한다.

```text
현재:
RAGClient._retrieve()
  → SearchService.search()

변경:
RAGClient._retrieve()
  → MainRAGService.search()
```

최종 UI 생성 경로:

```text
Q&A:
main retrieval + main generation

Workspace 카드:
main retrieval + 현재 카드 schema extraction
```

카드 추출은 main에 존재하지 않는 UI application 기능이므로 현재 backend 로직을 사용하는 예외다.

---

## 9. 평가 Pipeline 연결

## 9.1 Retrieval 평가

```text
Golden question
  → main standalone rewrite 사용 여부 설정
  → main Chroma retrieval
  → Evaluation Adapter
  → Hit@K / MRR / NDCG
```

평가 configuration 예:

```text
main-naive
main-advanced
main-advanced-rewrite
main-advanced-reranked
```

현재 pipeline은 비교 baseline으로 남긴다.

```text
advan-bm25
advan-dense
advan-hybrid
advan-reranked
```

## 9.2 Generation 평가

```text
Golden question
  → 새 BidMateRAGSession 생성
  → main query rewrite/retrieval/generation
  → Generation Evaluation Adapter
  → RAGAS / LLM Judge / citation 평가
```

일반 golden set 평가에서는 이전 sample의 대화 상태가 다음 sample을 오염시키지 않도록 질문마다 새 session을 만든다.

```python
for sample in golden_set:
    session = BidMateRAGSession(...)
    result = run_sample(session, sample)
```

멀티턴 평가는 별도의 conversation 단위 dataset으로 실행한다.

## 9.3 평가 Metadata와 결과 경로

각 결과에 pipeline과 모델 정보를 기록한다.

```json
{
  "pipeline": "main-advanced-chroma",
  "embedding_model": "text-embedding-3-small",
  "generation_model": "gpt-5-nano",
  "collection": "ai11_policy",
  "rewrite_enabled": true
}
```

보고서 경로를 분리한다.

```text
reports/main_rag/
reports/advan_baseline/
reports/comparison/
```

---

## 10. 구현 우선순위

## P0 — Main Pipeline 재현

목표: main pipeline을 현재 저장소에서 알고리즘 변경 없이 실행한다.

작업:

1. main 코드를 `src/main_rag/`로 복사한다.
2. import 경로를 `src.main_rag.*`로 변경한다.
3. main 의존성을 현재 `pyproject.toml`에 병합한다.
4. 절대 경로를 `configs/main_rag.yaml`로 이동한다.
5. main preprocessing/chunking/indexing CLI를 현재 저장소에서 실행한다.
6. retriever를 지연 초기화 가능한 객체로 감싼다.
7. 원본과 포팅본의 retrieval parity test를 추가한다.

완료 조건:

- 같은 입력, 설정, 질문에서 원본과 동일한 chunk ID가 검색된다.
- main의 기존 테스트가 현재 저장소에서도 통과한다.
- main 실행 중 현재 `SearchService`를 호출하지 않는다.
- 절대 경로 없이 현재 workspace의 설정으로 실행된다.

## P1 — 평가 Adapter

목표: 현재 평가 도구가 main retrieval/generation을 평가하게 한다.

작업:

1. main retrieval 결과를 공통 평가 형식으로 변환한다.
2. main generation 결과를 answer 평가 형식으로 변환한다.
3. golden set retrieval 평가에 `main` backend를 추가한다.
4. RAGAS 및 LLM Judge에 main answer runner를 연결한다.
5. pipeline/config metadata를 결과에 저장한다.
6. 질문마다 새 session을 생성해 평가 독립성을 보장한다.

완료 조건:

- main pipeline으로 현재 retrieval 평가를 실행할 수 있다.
- main pipeline으로 현재 answer/RAGAS/LLM Judge 평가를 실행할 수 있다.
- 결과가 main pipeline ID와 별도 report 경로로 저장된다.
- sample 사이에 대화 상태가 공유되지 않는다.

## P2 — UI Q&A 연결

목표: 현재 UI의 질문이 main retrieval과 main generation을 사용하게 한다.

작업:

1. `MainRAGService` facade를 구현한다.
2. conversation별 `BidMateRAGSession` registry를 구현한다.
3. `RAGClient.answer()`를 main service에 연결한다.
4. main evidence를 UI citation으로 변환한다.
5. `conversation_id` 전달과 reset 경로를 추가한다.
6. API key, index 없음, 검색 결과 없음 등의 오류 처리를 추가한다.

완료 조건:

- UI 질문이 main retriever와 `BidMateRAGSession`을 거친다.
- 후속 질문에 main query rewrite가 적용된다.
- 서로 다른 conversation의 상태가 섞이지 않는다.
- citation이 main retrieval metadata와 일치한다.

## P3 — UI Workspace 카드 연결

목표: UI의 모든 카드가 main retrieval 결과를 사용하게 한다.

작업:

1. `RAGClient._retrieve()`를 main retrieval adapter로 교체한다.
2. Overview, Risks, Eligibility, Deliverables, Requirements를 확인한다.
3. main metadata의 document/page/chunk를 카드 evidence로 변환한다.
4. document metadata filter를 적용한다.
5. 카드별 회귀 테스트를 추가한다.

완료 조건:

- 모든 카드가 main Chroma 검색 결과를 사용한다.
- 카드 evidence가 실제 main chunk를 가리킨다.
- 현재 frontend 변경이 최소화된다.
- 다른 문서의 청크가 workspace 카드에 섞이지 않는다.

## P4 — 안정화 및 비교

목표: 재현성과 시연 안정성을 확보하고 현재 baseline과 비교한다.

작업:

1. memory session TTL/cleanup을 추가한다.
2. index와 API key 상태 점검을 추가한다.
3. single-worker/multi-worker 운영 조건을 문서화한다.
4. latency와 API 비용을 측정한다.
5. 원본 main과 포팅본의 parity test를 정기 실행한다.
6. main과 현재 baseline의 비교 평가 보고서를 작성한다.

완료 조건:

- backend 재시작과 session 제약이 문서화된다.
- 실패 시 사용자에게 이해 가능한 오류가 반환된다.
- 원본 대비 포팅본의 결과 차이가 설명 가능하다.
- main과 baseline의 품질·속도·비용 비교 결과가 저장된다.

---

## 11. 우선순위 요약

| 우선순위 | 작업 | 중요도 | 핵심 결과 |
|---|---|---|---|
| P0 | main pipeline 복사·재현 | 필수 | main 알고리즘을 현재 저장소에서 동일하게 실행 |
| P1 | 평가 adapter | 필수 | 현재 Golden/RAGAS/LLM Judge로 main 평가 |
| P2 | UI Q&A 연결 | 필수 | UI 채팅에서 main retrieval/generation 사용 |
| P3 | Workspace 카드 연결 | 중요 | UI 카드가 main retrieval 결과와 evidence 사용 |
| P4 | 안정화 및 baseline 비교 | 중요 | 재현성, session 관리, latency 및 비교 보고서 확보 |

---

## 12. 최종 Pipeline

```text
                        ┌──────────────────────────────┐
                        │ main preprocessing/chunking │
                        └──────────────┬───────────────┘
                                       ↓
                         OpenAI embedding + Chroma
                                       ↓
                    ┌──────────────────┴──────────────────┐
                    ↓                                     ↓
             UI Q&A 요청                           UI 카드 요청
                    ↓                                     ↓
          main query rewrite                    카드별 고정 query
                    ↓                                     ↓
          main Chroma retrieval                main Chroma retrieval
                    ↓                                     ↓
          BidMateRAGSession.ask                카드 schema extraction
                    ↓                                     ↓
          Generation UI Adapter                  Evidence Adapter
                    └──────────────────┬──────────────────┘
                                       ↓
                              현재 FastAPI + UI

별도 평가:
Golden set → main RAG adapters → 현재 평가/RAGAS/LLM Judge
```

---

## 13. 하지 않을 것

- main retriever를 현재 `SearchService`로 교체하지 않는다.
- main Chroma 검색을 현재 BM25/hybrid와 강제로 합치지 않는다.
- main의 chunking을 현재 chunk schema에 맞추기 위해 알고리즘 수준에서 재작성하지 않는다.
- main 답변을 현재 `OpenAIRAGService`로 다시 생성하지 않는다.
- main 코드를 기존 `src/search` 또는 `src/generation` 위에 덮어쓰지 않는다.
- 현재 baseline RAG를 바로 삭제하지 않는다.
- UI conversation 전체가 하나의 전역 `BidMateRAGSession`을 공유하지 않는다.

---

## 14. 최종 권고

구현은 다음 순서로 진행한다.

```text
main pipeline 포팅 및 parity 확인
  → 평가 adapter 연결
  → UI Q&A 연결
  → Workspace 카드 연결
  → session/latency 안정화
  → 현재 baseline과 비교 보고
```

이 구조에서는 main pipeline이 뒤로 밀리거나 현재 pipeline에 흡수되지 않는다. main이 RAG의 중심이며 현재 저장소는 main을 실행하고 평가하고 화면에 표시하는 플랫폼 역할을 한다.
