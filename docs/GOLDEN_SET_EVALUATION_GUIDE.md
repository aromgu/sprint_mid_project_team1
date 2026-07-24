# Golden Set 기반 RAG 파이프라인 평가 가이드

## 1. 기본 평가 방식

PDF를 하나씩 업로드하며 테스트하기보다, 9개 PDF 전체를 한 번에 전처리·인덱싱한 뒤 하나의 Golden Set으로 반복 평가한다.

```text
9개 PDF 전체
  ↓ 한 번만 처리
Parsing → Chunking → BM25/Dense Index
  ↓
Golden Set 1개(문서별 질문 포함)
  ↓
Chunk/Index → Retriever → Generation → E2E 평가
```

평가할 때 PDF를 다시 업로드할 필요는 없다. 다음 데이터가 준비되어 있으면 된다.

```text
data/processed/pages.jsonl
data/processed/chunks.jsonl
data/indexes/
data/eval/golden_set.jsonl
```

실제 UI는 사용자가 RFP 하나를 선택한 뒤 질문하므로, 주 평가는 해당 문서 안에서 검색하는 **문서 한정 평가**로 진행한다.

## 2. Golden Set 구성

9개 문서마다 10~15개 질문을 만들고 하나의 JSONL 파일로 합치는 방식을 권장한다.

```text
9개 PDF × 10개 질문 = 90개
```

각 항목에는 평가 대상 문서를 식별할 수 있도록 `reference_document_ids`를 반드시 포함한다.

```json
{
  "question_id": "하_2_q001",
  "question": "총 사업예산과 무상유지보수기간은?",
  "reference_context_ids": ["하_2_p002_c0003"],
  "reference_document_ids": ["하_2"],
  "reference_pages": [4],
  "reference_answer": "사업예산은 11,270,000,000원이고 무상유지보수기간은 사업종료일로부터 12개월이다.",
  "required_facts": [
    "사업예산은 11,270,000,000원이다",
    "VAT가 포함된다",
    "무상유지보수기간은 사업종료일로부터 12개월이다"
  ],
  "question_type": "multi_fact",
  "difficulty": "medium",
  "answerable": true,
  "review_status": "approved"
}
```

### 권장 질문 구성

| 질문 유형 | 권장 비율 |
|---|---:|
| 단일 청크 질문 | 40% |
| 여러 청크·페이지 종합 질문 | 30% |
| 조건·예외 질문 | 20% |
| 답변 불가능 질문 | 10% |

## 3. 검색 범위 설정

### 문서 한정 검색

실제 UI 품질을 평가하는 주 지표다.

```text
질문 + reference_document_ids
→ 해당 문서 안에서만 검색
```

### 전체 문서 검색

문서 선택 없이 전체 RFP를 대상으로 질문하는 확장 기능 평가에 사용한다.

```text
질문
→ 9개 문서 전체에서 검색
```

현재 제품은 사용자가 문서를 먼저 선택하므로 문서 한정 평가를 기본값으로 사용한다.

> 주의: 현재 Retrieval 평가 코드의 `evaluate_queries()`는 `reference_document_ids`를 검색 필터로 사용하지 않고 전체 문서를 검색한다. 실제 UI와 동일한 평가를 위해 문서 한정 검색 옵션을 추가해야 한다.

## 4. Chunk / Index 성능 평가

이 단계에서는 LLM을 사용하지 않는다.

### 확인 항목

- 원문 문장 유실 여부
- 페이지 번호 보존 여부
- Heading 보존율
- 요구사항 ID 보존율
- 너무 짧거나 긴 청크 비율
- 중복 `chunk_id`
- 임베딩 누락
- 청크 수와 Dense index 행 개수 일치 여부

### 권장 기준

| 지표 | 권장 기준 |
|---|---:|
| 원문 핵심 사실 보존율 | 99% 이상 |
| 페이지 번호 정확도 | 100% |
| 중복 chunk ID | 0건 |
| 임베딩 누락 | 0건 |
| 최대 토큰 초과 | 0건 |
| 너무 짧은 청크 | 5% 이하 |

예산, 기간, 수량, 담당 주체, 예외 조건처럼 답변에 필요한 핵심 정보가 청킹 과정에서 사라지지 않았는지 Golden Set의 근거 문장으로 검사해야 한다.

## 5. Retriever 성능 평가

먼저 OpenAI 비용이 발생하지 않는 검색 평가를 수행한다.

```bash
uv run python -m scripts.evaluate_retrieval \
  --golden data/eval/golden_set.jsonl \
  --retriever all \
  --top-k 10
```

### 비교 대상

- BM25
- Dense
- Hybrid
- 필요하면 Reranker

### 주요 지표

| 지표 | 의미 |
|---|---|
| Recall@1 | 첫 번째 결과에 정답 청크가 있는 비율 |
| Recall@3 | 상위 3개 안에 정답 청크가 있는 비율 |
| Recall@5 | 상위 5개 안에 정답 청크가 있는 비율 |
| MRR@10 | 정답 청크가 얼마나 앞에 나오는지 |
| nDCG@10 | 여러 정답 청크의 순위 품질 |
| Page Hit@5 | 정답 페이지를 찾았는지 |
| Latency | 검색 소요 시간 |

### 초기 목표값 예시

```text
Recall@5 ≥ 0.85
MRR@10 ≥ 0.70
Page Hit@5 ≥ 0.90
P95 검색시간 ≤ 1초
```

목표값은 문서 종류와 질문 난이도에 따라 조정한다.

## 6. Generation / RAGAS 평가

Retriever 성능이 안정된 후 실행한다. 검색 결과가 틀린 상태에서 LLM만 평가하면 실패 원인을 구분하기 어렵다.

```bash
uv run python -m scripts.evaluate_ragas \
  --golden data/eval/golden_set.jsonl
```

이 평가는 OpenAI API를 사용하므로 비용이 발생한다.

### RAGAS 지표

| 지표 | 의미 |
|---|---|
| Faithfulness | 답변이 검색 근거에 충실한가 |
| Answer Relevancy | 질문에 직접 답했는가 |
| Context Precision | 검색 근거 중 불필요한 내용이 적은가 |
| Context Recall | 정답에 필요한 근거를 충분히 가져왔는가 |

### 프로젝트 특화 지표

- `required_facts` 포함률
- 숫자·날짜·기간 정확도
- 답변 불가능 질문의 보류 정확도
- 존재하지 않는 출처 인용 비율
- 인용 페이지 정확도

RFP에서는 일반적인 문장 유사도보다 금액, 기간, 수량, 담당 주체와 예외 조건이 정확한지가 더 중요하다.

## 7. E2E / Operations 평가

최종적으로 실제 UI와 동일한 흐름을 테스트한다.

```text
문서 선택
→ 질문 입력
→ Hybrid 검색
→ LLM 답변
→ 인용 페이지 확인
```

### 확인 항목

- 질문에 직접 답했는가
- 필수 사실이 모두 포함됐는가
- 숫자와 기간이 정확한가
- 근거 페이지가 맞는가
- 근거 없는 내용을 만들지 않았는가
- 전체 응답시간
- 입력·출력 토큰과 API 비용
- 답변 불가능 질문을 적절히 거절했는가

### 권장 E2E 목표

```text
필수 사실 포함률 ≥ 90%
숫자·날짜 정확도 ≥ 95%
근거 페이지 정확도 ≥ 95%
근거 없는 주장 ≤ 2%
답변 불가능 질문 거절 정확도 ≥ 90%
```

## 8. 권장 실행 순서

1. 9개 PDF 전체 전처리 및 인덱싱
2. Golden Set 90개 작성
3. 사람이 근거 청크·페이지·정답 검수
4. Chunk / Index 무결성 검사
5. BM25, Dense, Hybrid 검색 결과 비교
6. 가장 좋은 Retriever 선택
7. RAGAS 및 필수 사실 기반 답변 평가
8. 실제 UI 기반 E2E 평가
9. 실패 질문을 유형별로 분류
10. 수정 후 같은 Golden Set으로 회귀 테스트

## 9. 실패 원인 분류

평가에 실패한 질문은 다음 단계 중 어디에서 문제가 발생했는지 구분한다.

| 실패 유형 | 주요 원인 |
|---|---|
| Chunk 실패 | 원문 유실, 잘못된 분할, 페이지 정보 오류 |
| Index 실패 | 임베딩 누락, 오래된 인덱스, 청크와 인덱스 불일치 |
| Retrieval 실패 | 정답 청크가 Top-k에 포함되지 않음 |
| Generation 실패 | 근거는 검색됐지만 답변이 핵심 사실을 누락함 |
| Citation 실패 | 답변은 맞지만 잘못된 청크나 페이지를 인용함 |
| E2E 실패 | API, 캐시, UI 표시 또는 응답시간 문제 |

동일한 Golden Set을 계속 사용해야 수정 전후 결과를 정확히 비교할 수 있다.
