# Claude용 RFP Golden Set 생성 프롬프트

## 사용 방법

Claude에 다음 자료를 함께 첨부한다.

1. 평가 대상 RFP PDF 9개
2. `data/processed/chunks.jsonl`
3. 아래의 프롬프트 본문

정확한 `reference_context_ids`를 생성하려면 PDF뿐 아니라 `chunks.jsonl`도 반드시 제공하는 것이 좋다. PDF만 제공하면 Claude가 프로젝트의 실제 청크 ID를 알 수 없으므로, 청크 ID를 임의로 생성하지 않도록 해야 한다.

한 번에 90개를 생성하는 것보다 문서별로 10개씩 나누어 생성하고 사람이 검수하는 방식을 권장한다.

---

## 프롬프트 본문

당신은 RFP 문서 기반 RAG 시스템의 Golden Set을 설계하는 전문 평가 데이터 구축자다.

첨부한 9개 PDF는 실제 RFP 원문이며, `chunks.jsonl`은 해당 PDF를 파싱·분할한 검색 청크 데이터다.

목표는 BM25, Dense Vector, Hybrid Retriever와 LLM 답변 품질을 평가할 수 있는 고품질 Golden Set을 만드는 것이다.

### 중요 원칙

1. 모든 질문과 정답은 첨부된 PDF 원문에서 확인할 수 있어야 한다.
2. 원문에 없는 내용을 추측하거나 일반 지식으로 보충하지 않는다.
3. 질문은 실제 입찰 담당자가 문서를 검토하며 물어볼 법한 자연스러운 한국어로 작성한다.
4. `reference_answer`는 PDF 근거만 사용하여 간결하고 명확하게 작성한다.
5. `reference_context_ids`에는 `chunks.jsonl`에 실제 존재하는 `chunk_id`만 넣는다.
6. `chunk_id`를 확실히 찾을 수 없다면 임의로 생성하지 말고 빈 배열로 두고 `review_status`를 `needs_chunk_mapping`으로 표시한다.
7. `reference_pages`에는 PDF에 표시된 실제 페이지 번호를 넣는다.
8. 파일 뷰어상의 순번과 문서에 인쇄된 페이지 번호가 다르면 `chunks.jsonl`의 `page_start`, `page_end` 기준을 우선한다.
9. 질문과 정답이 애매하거나 근거가 부족하면 `answerable`을 `false`로 설정한다.
10. 최종 결과에는 설명이나 Markdown 코드 블록 없이 JSONL 데이터만 출력한다.

### 평가 데이터 구성 목표

- 총 90개 질문을 작성한다.
- 9개 PDF마다 정확히 10개 질문을 작성한다.
- 특정 앞부분에 질문이 몰리지 않도록 문서 전체 구간을 고르게 사용한다.
- 동일 사실을 표현만 바꿔 중복 질문으로 만들지 않는다.
- 같은 정답 청크에 지나치게 많은 질문을 배정하지 않는다.
- 한 문서의 특정 분야에 질문이 편중되지 않도록 한다.

### 문서별 질문 유형

각 PDF에서 다음 유형을 가능한 한 고르게 포함한다.

1. 참가 자격 및 제한 조건
2. 실격 조건 및 감점 조건
3. 제출 서류와 필수 산출물
4. 제출 기한, 질의 기한, 사업 일정
5. 기능 요구사항
6. 성능 및 품질 요구사항
7. 보안 및 개인정보보호 요구사항
8. 운영·유지보수·교육 요구사항
9. 인력 및 조직 요구사항
10. 계약, 검수, 보고 또는 대금 지급 조건

문서에 해당 유형의 내용이 없다면 억지로 만들지 말고 다른 실제 유형으로 대체한다.

### 난이도 구성

각 PDF의 10개 질문을 가급적 다음처럼 구성한다.

- `easy`: 3개
  - 한 개 청크에서 직접 답을 찾을 수 있는 질문
- `medium`: 4개
  - 조건, 예외, 수량 또는 여러 문장을 함께 해석해야 하는 질문
- `hard`: 3개
  - 두 개 이상의 청크나 페이지를 종합해야 하는 질문

### 질문 유형 분류

`question_type`은 다음 중 하나를 사용한다.

- `single_context`: 한 청크로 답변 가능
- `multi_context`: 여러 청크 또는 페이지를 종합해야 함
- `condition`: 조건, 예외 또는 제한사항을 확인하는 질문
- `list`: 여러 항목을 빠짐없이 찾아야 하는 질문
- `comparison`: 둘 이상의 조건이나 요구사항 비교
- `unanswerable`: 원문만으로 답변할 수 없음

### 답변 가능 여부

- 답변 가능한 질문을 전체의 약 90%로 구성한다.
- 약 10%는 의도적으로 `answerable=false`인 질문으로 구성한다.
- 답변 불가능 질문도 해당 RFP와 관련 있는 자연스러운 질문이어야 한다.
- `answerable=false`이면 `reference_answer`에 다음과 같이 작성한다.

  `제공된 RFP 원문에서 해당 내용을 확인할 수 없다.`

- 이 경우 `reference_context_ids`와 `reference_pages`는 빈 배열로 둔다.
- 문서에 실제 답이 있는데 찾기 어렵다는 이유만으로 `unanswerable`로 지정하지 않는다.

### 근거 청크 선택 방법

1. PDF에서 정답 근거가 있는 문장과 페이지를 찾는다.
2. `chunks.jsonl`에서 동일 `document_id`와 해당 페이지를 가진 청크를 찾는다.
3. 정답 사실이 실제로 들어 있는지 청크의 `text`를 확인한다.
4. 직접 근거가 포함된 최소한의 청크만 `reference_context_ids`에 넣는다.
5. 단순히 같은 페이지에 있다는 이유로 관련 없는 청크를 넣지 않는다.
6. `multi_context` 질문은 답변에 필요한 모든 핵심 청크를 포함한다.
7. `reference_context_ids`의 각 ID는 `chunks.jsonl`의 실제 `chunk_id`와 문자 단위로 동일해야 한다.
8. `reference_document_ids`에는 `chunks.jsonl`의 실제 `document_id`를 사용한다.

### `required_facts` 작성 기준

- 정답에서 반드시 언급되어야 하는 독립적인 핵심 사실을 배열로 작성한다.
- 날짜, 수량, 담당 주체, 조건, 예외를 각각 별도 사실로 분리한다.
- 포괄적인 문장 하나로 모든 사실을 합치지 않는다.
- `answerable=false`이면 빈 배열로 둔다.

### 출력 스키마

한 줄에 하나의 JSON 객체를 출력한다.

```json
{
  "question_id": "중_3_q001",
  "question": "예약 가능 기간은 누가 설정하며 어떤 방식으로 관리해야 하는가?",
  "reference_context_ids": [
    "중_3_p001_c0001"
  ],
  "reference_document_ids": [
    "중_3"
  ],
  "reference_pages": [
    1
  ],
  "reference_answer": "관리자가 예약 가능 기간을 설정하고 운영 정책에 따라 관리해야 한다.",
  "required_facts": [
    "관리자가 예약 가능 기간을 설정한다",
    "운영 정책에 따라 예약 기간을 관리한다"
  ],
  "question_type": "single_context",
  "category": "functional",
  "difficulty": "easy",
  "answerable": true,
  "evidence_quotes": [
    "원문에서 확인한 짧은 직접 근거 문장"
  ],
  "review_status": "draft"
}
```

### 필드 작성 규칙

#### `question_id`

- `{document_id}_q001` 형식을 사용한다.
- 문서별로 `q001`부터 `q010`까지 부여한다.

#### `category`

다음 중 하나를 사용한다.

- `eligibility`
- `disqualification`
- `deliverable`
- `schedule`
- `functional`
- `performance`
- `security`
- `operation`
- `personnel`
- `contract`

#### `difficulty`

다음 중 하나를 사용한다.

- `easy`
- `medium`
- `hard`

#### `evidence_quotes`

- PDF 또는 `chunks.jsonl`에서 찾은 근거 문장을 짧게 기록한다.
- 긴 문단 전체를 복사하지 않는다.

#### `review_status`

- 정확한 청크까지 매핑했으면 `draft`로 설정한다.
- 청크 ID를 확인하지 못했으면 `needs_chunk_mapping`으로 설정한다.

### 자체 검증

최종 출력 전에 각 항목을 내부적으로 검증한다.

1. `question_id`가 전체 데이터에서 중복되지 않는가?
2. PDF마다 정확히 10개 질문이 있는가?
3. 총 90개 질문인가?
4. 모든 `reference_context_ids`가 `chunks.jsonl`에 실제 존재하는가?
5. 청크의 `document_id`가 `reference_document_ids`와 일치하는가?
6. 청크의 페이지 범위가 `reference_pages`를 포함하는가?
7. `reference_answer`의 모든 핵심 사실이 근거 청크에 존재하는가?
8. `required_facts`가 `reference_answer`와 일치하는가?
9. `answerable=false` 항목에 근거 ID가 잘못 포함되지 않았는가?
10. 질문이 서로 중복되거나 지나치게 유사하지 않은가?
11. 정답이 질문보다 더 넓은 내용을 임의로 추가하지 않았는가?
12. JSON 문법과 JSONL 형식이 올바른가?

### 품질 우선순위

질문 수를 채우는 것보다 근거 정확성을 우선한다. 확실하지 않은 사실이나 청크 ID를 만들어내지 않는다.

먼저 첨부된 9개 PDF와 `chunks.jsonl`의 문서 ID 대응 관계를 내부적으로 파악한 뒤 작업한다. 최종 응답에는 검토 과정, 요약, 표, Markdown 코드 펜스 또는 설명을 넣지 말고 JSONL 90줄만 출력한다.

---

## 문서별 생성 권장 프롬프트

한 번에 전체 90개를 만들지 않고 PDF별로 작업하려면 프롬프트 본문의 평가 데이터 구성 목표를 다음 내용으로 교체한다.

```text
이번 작업에서는 document_id가 "중_3"인 PDF 하나만 처리한다.
해당 문서에 대해 q001부터 q010까지 총 10개의 JSONL만 출력한다.
다른 문서의 질문은 생성하지 않는다.
```

`중_3`은 실제 처리할 `document_id`로 교체한다.

## 결과 저장 위치

Claude가 생성한 JSONL 결과를 사람이 검수한 후 다음 파일로 저장한다.

```text
data/eval/golden_set.jsonl
```

Retrieval 평가 실행:

```bash
uv run python -m scripts.evaluate_retrieval \
  --golden data/eval/golden_set.jsonl \
  --retriever all \
  --top-k 10
```

RAGAS 평가 실행:

```bash
uv run python -m scripts.evaluate_ragas \
  --golden data/eval/golden_set.jsonl
```

`evidence_quotes`, `category`, `difficulty`, `review_status`는 검수용 추가 필드다. 현재 Retrieval 평가기는 필요한 필드만 읽으므로 추가 필드가 포함되어 있어도 평가 실행에 영향을 주지 않는다.
