# Main Advanced RAG 평가 및 UI 연결 계획

## 1. 목표

`../sprint_mid_project_team1_main`의 **Advanced RAG pipeline만** 현재 `GIT_advan` 저장소로 가져와 평가하고 기존 FastAPI/UI에 연결한다.

Naive pipeline은 다음 범위에서 제외한다.

- 평가하지 않는다.
- UI에 연결하지 않는다.
- Naive/Advanced 비교 화면을 만들지 않는다.
- 실행 중 pipeline을 선택하는 기능을 만들지 않는다.

핵심 원칙은 다음과 같다.

> Main Advanced pipeline을 기준 RAG로 사용하고, 현재 저장소는 평가와 FastAPI/UI 연결에 사용한다.

---

## 2. 최종 구조

```text
Main Advanced preprocessing
  → Advanced chunking
  → Advanced indexing
  → Advanced runtime retrieval
  → BidMateRAGSession
  → 현재 평가 도구
  → 현재 FastAPI/UI
```

최종 서비스 흐름:

```text
Frontend
  → FastAPI
  → RAGClient
  → MainAdvancedRAGService
  → AdvancedRetriever
  → BidMateRAGSession
```

필요하지 않은 구성:

- Naive 평가
- Naive/Advanced 비교
- pipeline selector
- pipeline registry
- pipeline 선택 환경변수
- pipeline별 cache/session
- 비교 결과 UI
- 개발/일반/시연용 UI 분리

---

## 3. Main Pipeline의 현재 상태

Main 저장소에는 Naive와 Advanced 오프라인 pipeline이 모두 존재한다.

### Naive

```text
Naive recursive chunking
  → OpenAI embedding
  → Naive Chroma
  → 현재 main.py의 retriever
  → BidMateRAGSession
```

현재 main의 실제 Q&A는 Naive Chroma retriever에 연결되어 있다. 하지만 이번 작업에서는 Naive를 사용하지 않는다.

### Advanced

```text
Advanced preprocessing
  → KSS/Kiwi Advanced chunking
  → embedding_text + bm25_tokens
  ├─ OpenAI embedding → Advanced Chroma
  └─ BM25 pickle index
```

Advanced는 preprocessing, chunking, dense indexing, BM25 indexing까지 구현되어 있다. 반면 현재 `main.py`의 Q&A는 Advanced index에 연결되어 있지 않다.

따라서 이번 작업의 핵심 추가 구현은 다음 연결이다.

```text
Advanced index
  → Advanced runtime retriever
  → BidMateRAGSession
```

---

## 4. 가져올 범위

Main 저장소에서 다음 코드를 가져온다.

```text
src/loader/
src/preprocessing/
src/chunking/advanced_chunking.py
src/embeddings/build_advanced_index.py
src/generation/generate_answer.py

scripts/run_advanced_preprocessing.py
scripts/run_advanced_chunking.py
scripts/run_advanced_indexing.py
```

공통 전처리 과정에서 필요한 파일도 함께 가져온다.

예:

```text
src/preprocessing/table_formats.py
src/preprocessing/prepare_advanced.py
```

현재 main의 `src/retrieval/retriever.py`는 Naive Chroma에 연결되어 있으므로 그대로 사용하지 않는다. 반환 형식은 유지하되 Advanced index를 사용하는 `AdvancedRetriever`를 새로 만든다.

---

## 5. 가져오지 않을 범위

다음 Naive 관련 코드는 이번 평가 및 UI 연결 범위에서 제외한다.

```text
src/chunking/split_text.py
src/embeddings/build_embeddings.py
scripts/run_chunking.py
scripts/run_indexing.py
Naive Chroma index
Naive retriever 실행 경로
```

현재 저장소의 다음 RAG pipeline도 운영 경로에는 사용하지 않는다.

```text
src/search/bm25.py
src/search/dense.py
src/search/hybrid.py
src/search/reranker.py
src/search/query_planning.py
src/search/service.py
src/generation/openai_generator.py
```

기존 테스트나 참고를 위해 파일을 삭제할 필요는 없지만, Main Advanced의 Q&A 및 UI 경로에서는 호출하지 않는다.

---

## 6. 권장 파일 구조

Naive/Advanced 선택 기능이 없으므로 별도의 registry나 여러 backend 구조가 필요 없다.

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

복사한 코드의 import 경로만 `src.main_rag.*`에 맞게 변경한다.

```text
src.loader          → src.main_rag.loader
src.preprocessing   → src.main_rag.preprocessing
src.chunking        → src.main_rag.chunking
src.embeddings      → src.main_rag.embeddings
src.generation      → src.main_rag.generation
```

이는 알고리즘 변경이 아니라 현재 코드와 이름이 충돌하지 않게 하기 위한 packaging 변경이다.

---

## 7. Advanced Retrieval 구현

## 7.1 가장 단순한 권장안

먼저 Advanced Chroma dense retrieval만 Q&A에 연결한다.

```text
사용자 질문
  → BidMateRAGSession.rewrite_query()
  → Advanced Chroma similarity search
  → top 5
  → BidMateRAGSession.ask()
```

기본 index 정보:

```text
persist directory: chroma_advanced_v2
collection: ai11_policy_advanced_v2
embedding model: text-embedding-3-small
```

이 구성의 명칭은 정확하게 다음과 같이 사용한다.

```text
Main Advanced Chunking + Dense Retrieval
```

Advanced chunking을 사용하더라도 BM25/fusion을 연결하지 않았다면 `Advanced Hybrid`라고 부르지 않는다.

## 7.2 Dense를 우선하는 이유

- 기존 Naive retriever와 코드 구조가 거의 같다.
- main의 검색 결과 형식을 그대로 유지할 수 있다.
- fusion 정책을 새로 정의하지 않아도 된다.
- `BidMateRAGSession`을 거의 수정하지 않아도 된다.
- 평가와 UI까지 빠르게 연결할 수 있다.
- Main Advanced의 원본 산출물과 알고리즘을 덜 변경한다.

## 7.3 BM25/Hybrid에 대한 원칙

Main Advanced에는 BM25 index 생성 기능이 있지만 runtime retrieval과 fusion 정책은 명확히 연결되어 있지 않다.

다음을 확인하기 전에는 임의로 hybrid를 추가하지 않는다.

- dense와 BM25 결과를 어떤 방식으로 결합하려 했는가?
- score normalization 정책이 있는가?
- weighted sum인가, RRF인가?
- reranking은 계획인가, 실제 구현인가?

정책이 main 산출물에서 확인되고 데모 전에 구현할 여유가 있을 때만 후속 범위로 추가한다.

## 7.4 AdvancedRetriever 예시

```python
class AdvancedRetriever:
    def __init__(
        self,
        persist_directory,
        collection_name,
        embedding_model,
        api_key,
    ):
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
        )
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )

    def search_documents(
        self,
        query: str,
        k: int = 5,
        document_id: str | None = None,
    ) -> list[dict]:
        kwargs = {"k": k}
        if document_id:
            kwargs["filter"] = {"source_id": document_id}

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            **kwargs,
        )

        return [
            {
                "id": doc.metadata.get("chunk_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "text": doc.page_content,
                "file_nm": doc.metadata.get("source_filename"),
                "page": doc.metadata.get("page"),
                "score": float(score),
                "metadata": doc.metadata,
            }
            for doc, score in results
        ]
```

실제 metadata filter key와 페이지 필드는 Advanced index의 metadata contract에 맞춰 확정해야 한다.

---

## 8. MainAdvancedRAGService

평가와 UI가 공통으로 사용할 service 하나를 만든다.

```python
class MainAdvancedRAGService:
    def __init__(self, retriever, session):
        self.retriever = retriever
        self.session = session

    def search(self, query, document_id=None, top_k=5):
        return self.retriever.search_documents(
            query,
            k=top_k,
            document_id=document_id,
        )

    async def answer(self, question, document_id=None):
        rewritten_query = await self.session.rewrite_query(question)

        docs = self.search(
            rewritten_query,
            document_id=document_id,
            top_k=5,
        )

        return await self.session.ask(
            query=question,
            retrieved_docs=docs,
            rewritten_query=rewritten_query,
        )
```

평가와 UI가 retriever 및 session을 각자 다시 구현하지 않고 동일한 service를 사용하게 한다.

---

## 9. 설정 변경

Main의 절대 경로를 그대로 사용하지 않는다.

기존 예:

```text
/home/data/advanced/...
/home/data/chroma_advanced_v2
/home/data/bm25_advanced_v2
/home/data/reports/...
```

현재 저장소 안의 상대 경로로 설정한다.

```yaml
main_advanced_rag:
  preprocessing_dir: data/main_advanced/preprocessed
  chunks_path: data/main_advanced/chunks/chunks_advanced.jsonl.gz
  chroma_path: data/main_advanced/chroma
  reports_path: reports/main_advanced
  collection_name: ai11_policy_advanced_v2
  embedding_model: text-embedding-3-small
  retrieval_top_k: 5
  retrieval_mode: dense
```

pipeline 선택 설정은 필요하지 않다. Main Advanced가 유일한 운영 RAG다.

---

## 10. 평가 연결

Naive와 비교하지 않고 Main Advanced 자체가 정상 동작하는지 평가한다.

```text
Golden set
  → MainAdvancedRAGService
  → Advanced retrieval 결과
  → Advanced generation 결과
  → 현재 평가 도구
```

평가 ID:

```text
main-advanced-dense
```

결과 경로:

```text
reports/main_advanced/
├─ retrieval_summary.json
├─ retrieval_details.jsonl
├─ answers.jsonl
├─ ragas.json
└─ llm_judge.json
```

## 10.1 평가 항목

- Retrieval Hit@K
- 정답 근거 청크 포함 여부
- 답변 생성 성공 여부
- RAGAS
- LLM Judge
- citation의 문서/page/chunk 일치 여부
- retrieval/generation latency
- API 오류 및 빈 검색 결과 처리

## 10.2 평가 목적

Naive 대비 개선을 증명하는 것이 목적이 아니다. 다음 질문에 답하는 것이 목적이다.

```text
Advanced pipeline이 정상 동작하는가?
검색 결과가 정답 근거를 포함하는가?
답변이 검색 근거와 일치하는가?
UI 데모가 가능한 속도인가?
```

## 10.3 Session 격리

Golden set 평가에서는 이전 sample의 상태가 다음 질문에 영향을 주지 않도록 질문마다 새로운 `BidMateRAGSession`을 사용한다.

```python
for sample in golden_set:
    session = BidMateRAGSession(...)
    service = MainAdvancedRAGService(retriever, session)
    result = await service.answer(sample.question, sample.document_id)
```

---

## 11. UI 연결

UI에는 pipeline 정보나 선택 기능을 표시하지 않는다.

```text
사용자 질문
  → FastAPI
  → MainAdvancedRAGService
  → 답변과 evidence adapter
  → 기존 UI
```

UI에 표시할 항목:

- 답변
- 근거 문서
- 페이지
- 인용문
- 필요하면 confidence

표시하지 않을 항목:

- Naive/Advanced 선택기
- Dense/BM25/Hybrid 선택기
- pipeline ID
- embedding model
- collection 이름
- 개발/시연 mode
- pipeline 비교 결과

### 11.1 Q&A

```text
RAGClient.answer()
  → MainAdvancedRAGService.answer()
  → AdvancedRetriever
  → BidMateRAGSession
  → UI answer adapter
```

main의 답변을 현재 `OpenAIRAGService`로 다시 생성하지 않는다.

### 11.2 Conversation

부트캠프 데모 수준에서는 간단한 memory session으로 충분하다.

```text
conversation_id
  → BidMateRAGSession
```

브라우저당 `conversation_id` 하나를 사용한다. Redis나 DB 기반 session store는 이번 범위에서 제외한다.

### 11.3 Workspace 카드

현재 UI의 카드도 Advanced retriever를 사용한다.

```text
Overview query       ─┐
Risks query           │
Eligibility query     ├→ AdvancedRetriever → 현재 카드 schema 추출
Deliverables query    │
Requirements query   ─┘
```

현재 `RAGClient`의 카드 추출 로직은 유지하고 `_retrieve()` 내부만 Advanced retrieval adapter로 교체한다.

```text
Q&A:
Advanced retrieval + BidMateRAGSession

Workspace 카드:
Advanced retrieval + 현재 카드 Pydantic extraction
```

---

## 12. Adapter 역할

Adapter는 알고리즘을 변경하지 않고 Main의 결과를 현재 평가/UI 형식으로 변환한다.

Main answer:

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

UI answer:

```text
answer
citations
confidence
caveat
```

간단한 예:

```python
def to_ui_response(result):
    return {
        "answer": result["answer"],
        "citations": result["evidence"],
        "confidence": result["confidence"],
        "caveat": result.get("clarification_question"),
    }
```

실제 구현에서는 evidence의 source/page/chunk ID가 UI model과 일치하도록 명시적으로 변환한다.

---

## 13. 구현 우선순위

## P0 — Advanced Pipeline 실행

작업:

1. Advanced 관련 main 코드를 `src/main_rag/`로 가져온다.
2. import 경로를 `src.main_rag.*`로 변경한다.
3. 필요한 의존성을 현재 `pyproject.toml`에 병합한다.
4. 절대 경로를 설정 파일로 이동한다.
5. Advanced preprocessing을 실행한다.
6. Advanced chunking을 실행한다.
7. Advanced dense index를 생성한다.
8. `AdvancedRetriever`를 구현한다.
9. `BidMateRAGSession`과 연결한다.

완료 조건:

- 현재 저장소에서 Advanced 전처리부터 Q&A까지 실행된다.
- Advanced Chroma collection에서 결과를 검색한다.
- Naive Chroma와 현재 `SearchService`를 호출하지 않는다.
- main의 answer/evidence schema가 유지된다.

## P1 — 평가 연결

작업:

1. retrieval 결과 adapter를 구현한다.
2. answer 결과 adapter를 구현한다.
3. 현재 Golden set으로 retrieval 평가를 실행한다.
4. answer batch를 생성한다.
5. RAGAS 및 LLM Judge를 실행한다.
6. citation과 latency를 확인한다.

완료 조건:

- `main-advanced-dense` 결과가 별도 report 경로에 저장된다.
- 검색 결과가 정답 근거를 포함하는지 확인할 수 있다.
- 답변과 citation이 실제 검색 결과와 일치한다.
- 데모 가능한 latency인지 판단할 수 있다.

## P2 — UI 연결

작업:

1. `RAGClient.answer()`를 `MainAdvancedRAGService`에 연결한다.
2. 카드의 `_retrieve()`를 Advanced retrieval adapter로 교체한다.
3. evidence를 현재 UI citation 형식으로 변환한다.
4. memory conversation session을 연결한다.
5. 대표 데모 질문으로 smoke test를 실행한다.

완료 조건:

- UI 질문이 Advanced retrieval과 `BidMateRAGSession`을 거친다.
- UI에 답변, 문서, 페이지 및 인용문이 표시된다.
- Workspace 카드가 Advanced 검색 결과를 사용한다.
- pipeline 선택이나 비교 기능 없이 데모가 가능하다.

---

## 14. 우선순위 요약

| 우선순위 | 작업 | 결과 |
|---|---|---|
| P0 | Advanced pipeline 포팅 및 Q&A 연결 | Advanced 전처리부터 답변까지 실행 |
| P1 | 현재 평가 도구 연결 | Golden/RAGAS/LLM Judge 및 latency 확인 |
| P2 | 기존 UI 연결 | 답변·근거·카드를 기존 화면에 표시 |

---

## 15. 데모 방식

발표에서는 pipeline 내부 선택이나 비교를 설명하지 않는다.

간단한 설명:

> 문서 구조와 한국어 문장 경계를 반영한 Advanced RAG pipeline을 구축했고, 사전에 평가한 뒤 기존 RFP 분석 UI에 연결했습니다.

데모 순서:

```text
RFP 문서 선택
  → Overview/Requirements 카드 확인
  → 대표 질문 입력
  → 답변과 근거 페이지 확인
```

평가 결과가 필요하면 한 장의 표만 사용한다.

| 항목 | 결과 |
|---|---:|
| Hit@5 | 평가 결과 |
| Answer score | 평가 결과 |
| 평균 응답 시간 | 평가 결과 |

Naive 결과와 비교하지 않는다.

---

## 16. 하지 않을 것

- Naive를 평가하지 않는다.
- Naive를 UI에 연결하지 않는다.
- Naive/Advanced 비교 기능을 만들지 않는다.
- pipeline selector를 만들지 않는다.
- pipeline registry를 만들지 않는다.
- 별도의 개발·시연 UI를 만들지 않는다.
- Advanced BM25/hybrid 정책을 임의로 정의하지 않는다.
- main 답변을 현재 generator로 재생성하지 않는다.
- 현재 RAG와 Advanced RAG를 하나의 검색 pipeline으로 합치지 않는다.

---

## 17. 최종 결론

최종 구현은 다음 한 경로로 고정한다.

```text
Main Advanced만 포팅
  → Advanced Chroma dense retriever 연결
  → BidMateRAGSession 연결
  → 현재 평가 실행
  → 기존 UI에 연결
```

가장 중요한 추가 구현은 Advanced index를 실제 Q&A에 연결하는 `AdvancedRetriever`다. 이후 평가와 UI는 동일한 `MainAdvancedRAGService`를 사용한다.
