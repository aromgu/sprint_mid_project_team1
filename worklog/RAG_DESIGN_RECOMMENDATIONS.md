# RAG 시스템 설계 의견서

## 1. 결론

9개 RFP 문서를 난이도에 따라 서로 다른 시스템으로 구현하지 않고, 하나의 통합 RAG 파이프라인으로 처리하는 것을 권장한다.

문서 난이도는 어떤 검색 알고리즘을 적용할지 결정하는 기준이 아니라 평가 결과를 분석하기 위한 분류값으로 사용한다. BM25, Dense Retrieval, Hybrid Retrieval, Re-ranking을 동일한 9개 문서와 동일한 Golden set에서 비교한 뒤 실제 성능 개선이 확인된 구성만 최종 시스템에 포함한다.

권장 기본 구조는 다음과 같다.

```text
PDF 수집 및 경로 확정
→ 레이아웃·페이지·섹션 기반 파싱
→ 구조 기반 청킹 및 메타데이터 생성
→ BM25 검색 + Dense 검색
→ RRF 기반 결과 결합
→ Cross-encoder Re-ranking
→ 근거 및 페이지 인용을 포함한 답변 생성
→ 검색 지표 + RAGAS + 사람 평가
```

Pinecone, LLM 기반 쿼리 확장, LLM 기반 Re-ranking, 전체 페이지 OCR은 최초 구현에 포함하지 않는다. 기본 파이프라인의 평가 결과에서 필요성이 확인될 때 선택적으로 추가한다.

## 2. 대상 데이터

- 문서 목록: `RAG_검증_샘플_파일명_정리_2.csv`
- PDF 위치: `/home/hyojinkim/work/9_Etc_Labs/pdf`
- 대상 문서: 상·중·하 난이도별 3개, 총 9개 PDF
- Golden set 구축 절차: `GOLDEN_SET_CREATION_GUIDE.md`

CSV의 파일명과 실제 PDF 파일명은 공백, 괄호, 가운뎃점 등이 다를 수 있다. 최초 실행 시 실제 파일을 확인해 manifest를 만들고, 이후에는 문서 ID와 확정된 경로를 사용한다.

CSV에 기재된 난이도와 목표 정확도는 검증 결과가 아니라 사전 분류 및 목표값으로 취급한다.

## 3. 권장하지 않는 설계

### 3.1 난이도별 별도 시스템

다음과 같이 문서 난이도마다 알고리즘을 고정하지 않는다.

```text
하 난이도 문서 → BM25
중 난이도 문서 → Hybrid
상 난이도 문서 → Pinecone + LLM
```

이 방식은 문서 난이도와 알고리즘 차이가 동시에 바뀌기 때문에 성능 개선의 원인을 판단할 수 없다. 쉬운 문서에도 Dense Retrieval이 도움이 될 수 있고, 어려운 문서에서도 정확한 요구사항 ID나 기술 용어에는 BM25가 더 유리할 수 있다.

### 3.2 단계별 상속 구조

`AdvancedRetriever(HybridRetriever)`와 같이 단계가 올라갈수록 상속하는 구조는 권장하지 않는다. 검색 방식과 저장소, 결합 방식, reranker가 강하게 결합되어 기능을 독립적으로 비교하거나 교체하기 어렵다.

상속 대신 작은 구성요소를 조합하는 구조를 권장한다.

```text
DocumentLoader
Chunker
SparseRetriever
DenseRetriever
FusionStrategy
Reranker
AnswerGenerator
Evaluator
```

### 3.3 고정 점수 가중합

BM25 30%, Dense 70%와 같은 고정 가중합을 기본값으로 사용하지 않는다. 두 검색기의 점수 범위와 분포가 달라 정규화 방식에 따라 결과가 크게 변한다.

최초 구현에서는 순위 기반 RRF(Reciprocal Rank Fusion)를 사용한다. 이후 Golden set에서 가중합이 더 좋은지 별도 실험한다.

### 3.4 LLM이 출력한 신뢰도

LLM 또는 reranker가 출력한 `0.8` 같은 값을 정답 확률이나 confidence로 간주하지 않는다. 보정되지 않은 자기평가 점수는 실제 정확도를 의미하지 않는다.

운영 시에는 다음 신호를 조합해 답변 가능 여부를 판단한다.

- 상위 검색 결과의 관련성
- 정답 후보 청크의 수
- 상위 결과 사이의 점수 차이
- 서로 다른 검색 방식의 결과 일치 여부
- 답변을 지지하는 독립적인 근거 수
- RAGAS 및 Golden set에서 정한 임계값

출력 상태는 숫자 confidence보다 다음처럼 구분하는 것이 안전하다.

```text
answerable
uncertain
insufficient_evidence
```

## 4. 문서 수집과 manifest

문서마다 변하지 않는 ID를 부여한다.

```json
{
  "document_id": "상_1",
  "difficulty": "high",
  "organization": "국방과학연구소",
  "title": "기록관리시스템 통합 활용 및 보안 환경 구축",
  "pdf_path": "/home/hyojinkim/work/9_Etc_Labs/pdf/실제 파일명.pdf",
  "sha256": "파일 해시",
  "page_count": 69,
  "parser_version": "1.0"
}
```

파일 해시와 parser 버전을 기록하면 PDF 또는 파싱 로직이 변경됐을 때 인덱스와 Golden set 근거를 다시 생성해야 하는지 판단할 수 있다.

## 5. PDF 파싱

### 5.1 기본 선택

PDF 파싱의 기본 도구로 PyMuPDF를 권장한다. RFP 문서에서는 텍스트뿐 아니라 페이지, 블록 위치, 표 및 문서 구조가 중요하다.

다음 정보를 보존한다.

- 문서 ID와 제목
- 실제 PDF 페이지와 표시 페이지
- 장·절·하위 섹션
- 요구사항 ID
- 표 제목, 열 이름 및 행
- 텍스트 블록의 읽기 순서
- OCR 사용 여부

### 5.2 OCR

모든 페이지에 OCR을 수행하지 않는다. 텍스트 추출 결과가 없거나 현저히 적은 스캔 페이지에만 적용한다.

OCR 적용 여부와 품질을 메타데이터에 남긴다.

```json
{
  "ocr_applied": true,
  "ocr_engine": "tesseract",
  "ocr_language": "kor+eng"
}
```

### 5.3 표 처리

RFP 요구사항 표의 한 행은 하나의 의미 단위로 보존한다. 열 제목을 각 행의 텍스트에 포함해 검색 시 열의 의미가 사라지지 않게 한다.

예를 들어 값만 연결하지 않고 다음처럼 직렬화한다.

```text
요구사항 ID: SER-004
요구사항명: 사용자 접근통제
분류: 보안 요구사항
세부 내용: ...
```

## 6. 청킹 전략

고정된 300단어 단위의 공백 기반 청킹은 권장하지 않는다. 한국어 RFP는 표, 조사, 줄바꿈, 반복 머리글의 영향을 크게 받는다.

권장 우선순위는 다음과 같다.

1. 요구사항 표의 한 행
2. 제목과 본문으로 구성된 하나의 섹션
3. 의미적으로 완결된 문단
4. 너무 긴 섹션만 토큰 단위로 추가 분할

초기 실험값은 다음 범위에서 시작한다.

- 목표 크기: 500~800 tokens
- overlap: 청크 크기의 10~15%
- 검색 결과에 포함할 인접 청크: 필요 시 앞뒤 1개

청크는 문자열이 아닌 구조화된 객체로 관리한다.

```json
{
  "chunk_id": "상_1_p032_SER-004",
  "document_id": "상_1",
  "document_title": "기록관리시스템 통합 활용 및 보안 환경 구축",
  "page_start": 32,
  "page_end": 33,
  "section_path": ["제안요청 내용", "보안 요구사항"],
  "requirement_id": "SER-004",
  "content_type": "requirement_table_row",
  "text": "검색 및 생성에 사용할 본문"
}
```

## 7. 검색 설계

### 7.1 BM25

BM25는 제거하지 않고 기본 검색기로 유지한다. RFP의 요구사항 ID, 제품명, 약어, 기관명, 규격 및 보안 용어처럼 정확한 문자열에 강하다.

한국어 검색 품질을 위해 단순 공백 분할과 형태소 또는 subword 기반 토큰화를 Golden set에서 비교한다.

### 7.2 Dense Retrieval

Dense Retriever에는 한국어와 기술 문서를 처리할 수 있는 embedding 모델을 사용한다. 모델은 이름이나 공개 벤치마크만으로 선택하지 않고 9개 PDF의 Golden set에서 비교한다.

평가 시 다음 조건을 동일하게 유지한다.

- 동일한 청크
- 동일한 Golden set
- 동일한 top-k
- 동일한 메타데이터 필터
- 동일한 하드웨어 조건

### 7.3 Hybrid Retrieval

BM25와 Dense Retriever가 각각 상위 후보를 반환하고 RRF로 결합한다.

초기 권장값은 다음과 같다.

```text
BM25 top_k: 20
Dense top_k: 20
RRF 후보 통합: 최대 40
RRF k 상수: 60부터 실험
```

중복 청크는 `chunk_id`를 기준으로 병합한다. 검색 방식별 원래 순위와 점수도 결과에 보존해 오류 분석에 사용한다.

### 7.4 메타데이터 필터

사용자가 특정 문서, 기관, 요구사항 종류 또는 페이지 범위를 지정한 경우 검색 전에 필터링한다.

명시적 문서 조건이 없는 일반 질문에는 난이도 정보를 검색 필터로 사용하지 않는다.

## 8. Re-ranking

최초 reranker로 생성형 LLM보다 multilingual 또는 한국어 cross-encoder를 권장한다.

장점은 다음과 같다.

- 반복 평가가 상대적으로 일관된다.
- 지연시간과 비용이 낮다.
- JSON 파싱 및 응답 형식 오류가 없다.
- 후보별 관련성 점수를 직접 비교하기 쉽다.

초기 설정은 다음과 같이 시작한다.

```text
RRF 후보: 30~40개
Cross-encoder 입력: 질문 + 후보 청크
최종 검색 결과: 5~8개
```

LLM reranking은 다중 조건 질의처럼 cross-encoder가 반복적으로 실패하는 사례가 확인된 경우에만 선택적으로 적용한다.

## 9. 쿼리 처리

초기 구현에서는 수작업 동의어 사전 50개 또는 100개를 문서마다 만들지 않는다. 잘못된 동의어 확장은 관련 없는 결과를 늘리고 관리 비용도 크다.

우선 원본 질문으로 Hybrid Retrieval을 수행한다. Golden set 실패 사례를 분석한 후 필요한 기능만 추가한다.

추가 후보는 다음과 같다.

- 기관 및 사업별 약어 사전
- RFP 고유 요구사항 용어 정규화
- 긴 질문의 검색용 질의 축약
- multi-query retrieval
- query rewriting
- HyDE

동의어 및 query rewriting의 효과는 원본 질의 결과와 비교하는 ablation test로 검증한다.

## 10. 벡터 저장소

9개 PDF 규모에서는 Pinecone을 최초 선택으로 권장하지 않는다. 관리형 벡터 DB 사용 자체가 검색 정확도를 높이지 않으며, 초기 실험에는 외부 서비스 비용과 운영 복잡도만 증가할 수 있다.

권장 선택 기준은 다음과 같다.

| 상황 | 권장 저장 방식 |
|---|---|
| 초기 실험 및 소규모 데이터 | NumPy 또는 FAISS |
| 로컬/서버 운영과 메타데이터 필터 | Qdrant |
| 기존 PostgreSQL 활용 | pgvector |
| 대규모 운영과 관리형 서비스 필요 | Pinecone 검토 |

저장소는 검색 알고리즘과 분리된 인터페이스로 구현해 나중에 교체할 수 있게 한다.

## 11. 답변 생성

검색 성능이 먼저 검증된 후 답변 생성을 연결한다. 검색 실패와 생성 실패를 한 지표로 섞지 않는다.

답변 생성 프롬프트는 다음 원칙을 따른다.

- 제공된 근거만 사용한다.
- 근거가 부족하면 부족하다고 명시한다.
- 문서명과 PDF 페이지를 인용한다.
- 여러 근거가 충돌하면 충돌 사실을 밝힌다.
- 일반 지식을 RFP 요구사항처럼 표현하지 않는다.
- 답변과 출처를 구조화된 형식으로 반환한다.

권장 출력 구조는 다음과 같다.

```json
{
  "status": "answerable",
  "answer": "근거 기반 답변",
  "citations": [
    {
      "document_id": "상_1",
      "page": 32,
      "chunk_id": "상_1_p032_SER-004"
    }
  ]
}
```

## 12. 평가 설계

### 12.1 Golden set

먼저 문서당 10문항, 총 90문항의 Pilot Golden set을 구축한다. 자세한 생성 및 검수 절차는 `GOLDEN_SET_CREATION_GUIDE.md`를 따른다.

Golden set에는 최소한 다음 항목이 있어야 한다.

- 질문
- 기준 답변
- 핵심 사실
- 정답 문서
- 정답 페이지
- 정답 청크 ID
- 답변 가능 여부
- 질문 유형과 난이도

### 12.2 검색 지표

검색 성능의 주 지표로 단일 `accuracy`를 사용하지 않는다.

- Recall@5
- MRR@10
- nDCG@10
- ID 기반 Context Precision/Recall
- RAGAS Context Precision/Recall
- 검색 응답시간 p50/p95

문서 단위 적중과 청크 단위 적중을 구분한다. 올바른 문서를 찾았지만 정답 페이지를 찾지 못한 경우를 성공으로 처리하지 않는다.

### 12.3 답변 지표

- RAGAS Faithfulness
- RAGAS Response Relevancy
- RAGAS Answer Correctness 또는 Factual Correctness
- 핵심 사실 충족률
- 답변 불가능 문항 거절 정확도
- 문서 및 페이지 인용 정확도
- 응답시간과 호출 비용
- 사람 평가 표본

Faithfulness가 높아도 잘못 검색한 문맥을 충실하게 요약했을 수 있다. 따라서 반드시 Context Recall과 정답 청크 ID 평가를 함께 본다.

### 12.4 난이도별 분석

모든 검색기를 동일한 9개 문서에 적용한 뒤 결과를 다음처럼 집계한다.

- 전체 평균
- 상 난이도 3개 평균
- 중 난이도 3개 평균
- 하 난이도 3개 평균
- 질문 유형별 평균
- 답변 가능/불가능 문항별 평균

이 방식으로 알고리즘이 실제로 어느 문서와 질문 유형에 도움이 되는지 판단한다.

## 13. 실험 순서

각 단계에서 이전 단계와 동일한 Golden set을 사용한다.

### Experiment 0: 데이터 품질

- 9개 PDF 경로 확정
- 텍스트 추출률 측정
- 스캔 페이지 탐지
- 표와 요구사항 ID 추출 품질 확인
- 반복 머리글 및 바닥글 제거

### Experiment 1: BM25 baseline

- 구조 기반 청킹
- BM25 인덱싱
- 토큰화 방식 비교
- Recall@5, MRR@10, nDCG@10 측정

### Experiment 2: Dense Retrieval

- 2~3개 embedding 모델 비교
- 동일한 청크와 top-k 사용
- 문서 및 질문 유형별 실패 사례 분석

### Experiment 3: Hybrid Retrieval

- BM25와 Dense 결과를 RRF로 결합
- BM25, Dense, Hybrid의 paired comparison
- 성능 개선의 통계적·실용적 크기 확인

### Experiment 4: Re-ranking

- Hybrid 상위 30~40개 후보 재정렬
- 최종 top 5~8의 Recall과 nDCG 비교
- 지연시간 증가와 성능 개선을 함께 평가

### Experiment 5: 답변 생성

- 근거 기반 답변과 페이지 인용 생성
- RAGAS와 사람 평가 수행
- 답변 불가능 문항의 거절 성능 확인

### Experiment 6: 선택적 고도화

실패 사례가 충분히 누적된 뒤 다음 기능을 개별적으로 실험한다.

- 선택적 OCR
- 표 전용 파서
- query rewriting
- multi-query retrieval
- LLM reranking
- 외부 벡터 DB

한 번에 여러 기능을 추가하지 않고 각 기능의 효과를 ablation test로 확인한다.

## 14. 권장 프로젝트 구조

```text
rag_system/
├── configs/
│   ├── base.yaml
│   └── experiments/
├── data/
│   ├── manifests/
│   ├── processed/
│   └── golden_set/
├── ingestion/
│   ├── pdf_parser.py
│   ├── layout_parser.py
│   └── ocr.py
├── chunking/
│   ├── base.py
│   ├── section_chunker.py
│   └── requirement_chunker.py
├── retrieval/
│   ├── sparse.py
│   ├── dense.py
│   ├── fusion.py
│   └── reranker.py
├── generation/
│   ├── answer_generator.py
│   └── citations.py
├── evaluation/
│   ├── retrieval_metrics.py
│   ├── ragas_metrics.py
│   ├── benchmark.py
│   └── reports.py
├── schemas/
│   ├── document.py
│   ├── chunk.py
│   └── evaluation.py
├── tests/
└── main.py
```

Phase별 구현을 별도 클래스 계층으로 만들지 않고 설정 파일로 실험 구성을 선택한다.

```yaml
retrieval:
  sparse:
    enabled: true
    type: bm25
    top_k: 20
  dense:
    enabled: true
    model: "selected-after-benchmark"
    top_k: 20
  fusion:
    type: rrf
    rrf_k: 60
  reranker:
    enabled: true
    type: cross_encoder
    candidate_k: 40
    final_k: 6
```

## 15. 재현성과 추적성

각 실험 결과에는 다음 정보를 함께 저장한다.

- Git commit 또는 코드 버전
- PDF 파일 해시
- parser 및 chunker 버전
- embedding과 reranker의 정확한 모델 버전
- 청크 크기와 overlap
- 검색기 파라미터
- Golden set 버전 및 split
- 평가 LLM 모델과 설정
- 실행 시간과 하드웨어
- API 비용

LLM 기반 RAGAS 지표는 평가 모델과 실행 시점에 따라 변동할 수 있으므로 모델 버전을 고정하고, 핵심 결과에는 반복 실행 또는 사람 평가 표본을 포함한다.

## 16. 구현 우선순위

### 반드시 먼저 구현

1. 9개 PDF manifest
2. 페이지 및 요구사항 구조를 보존하는 파서
3. 구조 기반 청킹과 안정적인 chunk ID
4. 90문항 Pilot Golden set
5. BM25 baseline
6. Dense 및 RRF Hybrid 검색
7. ID 기반 검색 평가
8. Cross-encoder reranker
9. 출처 페이지를 포함한 답변 생성
10. RAGAS 평가

### 평가 후 결정

- 한국어 형태소 분석기
- 동의어 사전
- query rewriting
- multi-query retrieval
- OCR 범위 확대
- LLM reranking
- Pinecone 또는 다른 관리형 벡터 DB
- 멀티모달 RAG

## 17. 최종 기술 선택안

| 영역 | 권장 선택 |
|---|---|
| PDF 파싱 | PyMuPDF 중심 |
| 표 처리 | PyMuPDF 우선, 필요 시 표 전용 파서 추가 |
| OCR | 스캔 페이지에만 선택 적용 |
| 청킹 | 요구사항 행·섹션 기반, 긴 구간만 토큰 분할 |
| Sparse 검색 | BM25 |
| Dense 검색 | Golden set에서 선정한 한국어/다국어 embedding |
| 검색 결합 | RRF |
| Re-ranking | Cross-encoder 우선 |
| 벡터 저장소 | 초기 FAISS 또는 Qdrant |
| 답변 생성 | 근거 제한 및 페이지 인용이 가능한 LLM |
| 검색 평가 | Recall@5, MRR@10, nDCG@10, ID 기반 지표 |
| 답변 평가 | RAGAS + 핵심 사실 + 인용 정확도 + 사람 표본 |

## 18. 최종 의견

이 프로젝트의 핵심은 고급 기술을 많이 연결하는 것이 아니라, 9개 PDF에서 어떤 구성요소가 실제로 성능을 개선하는지 재현 가능하게 증명하는 것이다.

따라서 다음 원칙을 최종 설계 기준으로 삼는다.

1. 모든 문서에 동일한 통합 파이프라인을 적용한다.
2. 문서 난이도는 알고리즘 선택이 아니라 결과 분석에 사용한다.
3. 검색 성능을 먼저 검증한 후 답변 생성을 평가한다.
4. Golden set의 정답 청크 ID를 검색 평가의 중심으로 사용한다.
5. RAGAS는 검색 지표와 사람 평가를 보완하는 도구로 사용한다.
6. Pinecone, LLM reranking, query expansion은 실패 사례가 필요성을 증명할 때 추가한다.
7. 각 기능은 독립적인 ablation test를 통과한 경우에만 최종 시스템에 포함한다.

