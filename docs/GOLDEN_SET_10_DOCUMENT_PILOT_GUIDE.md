# 임의의 RFP 10개로 시작하는 Golden Set Pilot 가이드

## 1. 목적

이 가이드는 전체 98개 RFP를 바로 처리하지 않고, 먼저 대표성 있는 10개 문서를 선정하여 문서당 10문항, 총 100문항의 Pilot Golden Set을 만드는 절차를 정의한다.

Pilot의 목적은 문항 수를 빠르게 늘리는 것이 아니다. 실제 생성·검수·평가를 끝까지 수행하면서 다음 항목을 학습하고, 그 결과로 Golden Set의 품질 기준을 개선하는 것이다.

- 어떤 문서와 페이지에서 파싱 오류가 발생하는가
- 어떤 질문 유형에서 근거 누락과 잘못된 답변 거절이 발생하는가
- 단일·복합 질문의 난이도 기준이 적절한가
- 섹션과 `required_facts`가 검색·답변 평가에 충분한가
- LLM 검증과 자동 검사에서 잡지 못하고 사람이 발견하는 오류는 무엇인가

## 2. 변하지 않는 원칙

1. 질문보다 정답 근거를 먼저 고른다.
2. 질문과 정답은 해당 RFP 원문만으로 검증할 수 있어야 한다.
3. LLM 출력은 항상 후보 데이터이며, 사람 승인 전에는 Golden으로 취급하지 않는다.
4. 정답 문장뿐 아니라 문서, 페이지, 섹션, 핵심 사실을 기록한다.
5. 청크 ID는 파서 변경에 따라 바뀔 수 있으므로 문서·섹션·사실을 정답의 안정적인 기준으로 사용한다.
6. 검색 평가와 답변 생성 평가를 구분한다.
7. 문서 전체에 답이 없는 자연스러운 질문도 포함하여 환각과 답변 거절 능력을 평가한다.
8. Pilot의 최종 테스트 문항은 검색기, 청킹 또는 프롬프트 튜닝에 사용하지 않는다.

## 3. 10개 문서 선정

단순히 목록 앞의 10개를 선택하지 않는다. 재현 가능한 무작위 선택을 기본으로 하되, 전체 98개의 특성을 가능한 한 포함하도록 층화한다.

### 권장 구성

| 특성 | 권장 문서 수 | 확인 목적 |
|---|---:|---|
| 텍스트 추출이 양호한 일반 문서 | 3 | 기본 검색·답변 성능 |
| 요구사항 표가 많은 문서 | 2 | 표와 요구사항 ID 보존 |
| 장문 또는 섹션 구조가 복잡한 문서 | 2 | 복합 검색과 문맥 연결 |
| OCR 또는 파싱 품질이 낮은 문서 | 1 | 전처리 실패 대응 |
| 참가 자격·계약 조건이 복잡한 문서 | 1 | 조건·예외 질문 |
| 보안·성능 요구사항이 많은 문서 | 1 | 전문 용어 검색 |

한 문서가 여러 특성을 동시에 만족해도 된다. 최종 선정 결과에는 선택 사유와 난수 seed를 기록한다.

```json
{
  "pilot_version": "pilot-01",
  "selection_seed": 20260727,
  "documents": [
    {
      "document_id": "실제 ID",
      "source_document": "실제 PDF 파일명",
      "selection_tags": ["table_heavy", "complex_contract"],
      "selection_reason": "요구사항 표와 참가 제한 조건이 모두 포함됨"
    }
  ]
}
```

## 4. Pilot 문항 구성

각 문서에서 10문항을 작성한다. 문서에 존재하지 않는 유형을 억지로 채우지 않는다.

### 내용 범주

- 참가 자격 및 제한 조건
- 실격 및 감점 조건
- 제출 서류와 필수 산출물
- 제출·질의 기한과 사업 일정
- 기능 요구사항
- 성능·품질 요구사항
- 보안·개인정보보호 요구사항
- 운영·유지보수·교육 요구사항
- 인력·조직 요구사항
- 계약·검수·보고·대금 지급 조건

### 문서별 권장 난이도

- `easy` 3개: 한 섹션 또는 한 근거에서 직접 답변
- `medium` 4개: 조건, 예외, 수량 또는 목록 해석 필요
- `hard` 3개: 서로 다른 섹션이나 페이지의 사실을 종합

난이도는 질문 문장의 길이가 아니라 필요한 근거의 수와 추론 부담으로 결정한다.

### 답변 가능 여부

- 문서당 9개: `answerable=true`
- 문서당 1개: `answerable=false`

답변 불가능 문항은 관련 RFP에 대해 실제 사용자가 물을 법해야 하며, PDF 전체 검색 후에도 답이 없음을 확인해야 한다.

## 5. 권장 출력 스키마

한 줄에 하나의 JSON 객체를 기록한다.

```json
{
  "question_id": "DOC_ID_q001",
  "question": "실제 사용자가 물을 자연스러운 질문",
  "query_type": "keyword",
  "category": "eligibility",
  "difficulty": "easy",
  "hop_count": 1,
  "gold_sections": ["상위절_하위절"],
  "gold_sections_optional": [],
  "reference_context_ids": ["실제_chunk_id"],
  "reference_pages": [12],
  "ground_truth": "RFP 원문에만 근거한 간결하고 완전한 기준 답변",
  "required_facts": ["반드시 포함할 독립 사실 1"],
  "evidence_quotes": ["검수에 필요한 짧은 원문 근거"],
  "answerable": true,
  "source_document": "실제 PDF 파일명",
  "document_id": "DOC_ID",
  "review_status": "draft",
  "review_notes": "",
  "generator_model": "사용 모델과 버전",
  "validator_model": ""
}
```

### 허용값

`query_type`:

- `keyword`: 원문 용어와 직접 대응
- `semantic`: 동의어 또는 자연스러운 표현 변형
- `multihop`: 여러 근거를 함께 회수해야 함
- `list`: 여러 항목을 빠짐없이 찾아야 함
- `comparison`: 둘 이상의 조건 비교
- `unanswerable`: 문서 전체에 답이 없음

`category`:

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

청크 ID를 확정하지 못한 경우 임의로 만들지 않는다. `reference_context_ids`를 빈 배열로 두고 `review_status`를 `needs_chunk_mapping`으로 기록한다.

## 6. 전체 작업 순서

```text
10개 문서 선정 및 manifest 고정
→ PDF·청크·섹션 매핑 점검
→ 문서별 근거 후보 선정
→ 문서별 10문항 초안 생성
→ 별도 세션 또는 별도 모델로 독립 검증
→ JSONL 자동 품질 검사
→ 사람이 PDF 원문과 대조
→ 승인·수정·반려 상태 기록
→ 개발·검증·테스트 split 고정
→ Retriever 및 답변 생성 평가
→ 실패 유형과 규칙 변경 기록
→ 수정된 규칙으로 두 번째 Pilot 또는 전체 확장 결정
```

한 번에 10개 문서의 100문항을 생성하지 않는다. 문서 하나씩 생성하고 검수한 뒤 다음 문서로 이동한다. 첫 2개 문서를 완료한 시점에 오류 유형을 검토하고, 생성 규칙을 한 차례 보정한다.

## 7. 문서별 Golden Set 생성 프롬프트

아래 프롬프트에 한 개 RFP의 PDF 원문 또는 페이지별 추출문, 해당 문서의 청크 데이터, 문서 메타데이터를 함께 제공한다.

```text
당신은 공공 RFP 기반 RAG 시스템의 Golden Set을 설계하는 전문 평가 데이터 구축자다.

이번 작업에서는 아래에 제공된 RFP 문서 한 개만 처리한다.

[입력]
- document_id: {{DOCUMENT_ID}}
- source_document: {{PDF_FILENAME}}
- RFP 원문 또는 페이지별 추출문
- 해당 문서의 chunks.jsonl 레코드
- 필요한 경우 문서 manifest와 섹션 목록

[목표]
이 문서에 대해 q001부터 q010까지 정확히 10개의 Golden Set 후보를 만든다. 결과는 BM25, Dense, Hybrid Retriever의 검색 품질과 LLM 답변 품질을 모두 평가할 수 있어야 한다.

[절대 원칙]
1. 질문을 만들기 전에 객관적인 정답을 만들 수 있는 원문 근거부터 선택한다.
2. 질문, ground_truth, required_facts는 제공된 RFP 원문만으로 검증 가능해야 한다.
3. 외부 지식, 업계 관행, 법률에 대한 일반 지식으로 내용을 보충하지 않는다.
4. 질문은 실제 입찰·제안 담당자가 물을 법한 자연스러운 한국어로 작성한다.
5. ground_truth는 질문이 요구한 모든 핵심 사실을 포함하되 불필요하게 길게 쓰지 않는다.
6. required_facts는 날짜, 수량, 담당 주체, 조건, 예외를 독립적인 사실로 분리한다.
7. gold_sections에는 정답 사실이 실제로 존재하는 최소한의 섹션만 기록한다.
8. reference_context_ids에는 제공된 chunks.jsonl에 문자 단위로 동일하게 존재하고 정답을 직접 뒷받침하는 chunk_id만 기록한다.
9. 청크 ID를 확인할 수 없으면 만들지 말고 빈 배열과 review_status=needs_chunk_mapping을 사용한다.
10. reference_pages는 chunks.jsonl의 page_start/page_end와 일치해야 한다.
11. evidence_quotes는 원문 대조에 필요한 짧은 문장만 사용하며 긴 문단을 복사하지 않는다.
12. 같은 사실을 표현만 바꾸어 중복 문항으로 만들지 않는다.
13. 문서 앞부분이나 같은 섹션에 문항을 집중시키지 않는다.
14. LLM이 만든 결과는 승인된 정답이 아니라 검수 전 후보이므로 review_status=draft로 기록한다.

[문항 구성]
- answerable=true 9개, answerable=false 1개를 목표로 한다.
- easy 3개, medium 4개, hard 3개를 목표로 한다.
- hard 문항은 가능한 경우 서로 다른 페이지 또는 섹션의 두 개 이상 근거를 요구해야 한다.
- keyword, semantic, multihop, list, comparison 유형을 문서 내용에 맞게 섞는다.
- 참가 자격, 실격·감점, 제출물, 일정, 기능, 성능, 보안, 운영, 인력, 계약 범주를 고르게 검토한다.
- 문서에 없는 범주를 억지로 질문하지 말고 실제로 중요한 다른 범주로 대체한다.
- answerable=false 문항은 PDF 전체에 답이 없는지 확인한 뒤 작성한다. 단순히 찾기 어렵다는 이유로 답변 불가능으로 지정하지 않는다.

[난이도 판정]
- easy: 한 섹션 또는 한 청크의 직접 사실
- medium: 조건·예외·수량·목록을 함께 해석
- hard: 서로 다른 섹션이나 페이지의 사실을 종합

[출력 필드]
question_id, question, query_type, category, difficulty, hop_count,
gold_sections, gold_sections_optional, reference_context_ids,
reference_pages, ground_truth, required_facts, evidence_quotes,
answerable, source_document, document_id, review_status,
review_notes, generator_model, validator_model

[답변 불가능 문항 규칙]
- query_type은 unanswerable로 기록한다.
- ground_truth는 "제공된 RFP 원문에서 해당 내용을 확인할 수 없다."로 기록한다.
- gold_sections, reference_context_ids, reference_pages, required_facts, evidence_quotes는 빈 배열로 둔다.

[출력 전 내부 검사]
1. question_id가 {{DOCUMENT_ID}}_q001부터 q010까지 중복 없이 존재하는가?
2. 모든 정답의 각 문장이 원문 근거로 직접 검증되는가?
3. required_facts의 모든 항목이 ground_truth 및 근거와 일치하는가?
4. 모든 chunk_id가 입력 청크에 실제 존재하는가?
5. 페이지와 섹션이 근거 청크 및 PDF와 일치하는가?
6. multi-hop 문항의 모든 필수 사실에 대응하는 근거가 포함됐는가?
7. answerable=false 문항에 근거 배열이 포함되지 않았는가?
8. 질문 자체에 정답을 노출하지 않았는가?
9. 동일하거나 지나치게 유사한 문항이 없는가?
10. JSON 문법이 유효한가?

최종 응답에는 설명, 요약, 표, Markdown 코드 펜스를 넣지 않는다. 한 줄에 하나의 JSON 객체만 기록한 유효한 JSONL 10줄을 출력한다.
```

## 8. 독립 검증 프롬프트

생성에 사용한 대화 맥락을 공유하지 않는 별도 세션 또는 별도 모델에서 실행한다. 원문, 청크 데이터와 생성된 JSONL 10줄을 함께 제공한다.

```text
당신은 Golden Set 생성자가 아니라 독립 품질 검수자다. 제공된 RFP 원문과 chunks.jsonl만을 증거로 사용하여 Golden Set 후보를 엄격하게 검증하라.

[입력]
- document_id: {{DOCUMENT_ID}}
- source_document: {{PDF_FILENAME}}
- RFP 원문 또는 페이지별 추출문
- 해당 문서의 chunks.jsonl 레코드
- 검증 대상 Golden Set JSONL

[검증 항목]
1. question_id와 문서 ID가 정확한가?
2. 질문이 자연스럽고 실제 RFP 검토 업무와 관련 있는가?
3. ground_truth의 모든 문장이 원문에 직접 근거하는가?
4. 질문에서 요구한 핵심 사실이 ground_truth에 빠짐없이 포함됐는가?
5. required_facts가 독립적이고 자동 채점 가능한 단위인가?
6. gold_sections가 실제 정답 섹션이며 누락되거나 과도하지 않은가?
7. reference_context_ids가 실제 존재하고 정답을 직접 포함하는가?
8. reference_pages와 evidence_quotes가 PDF·청크와 일치하는가?
9. 난이도와 hop_count가 실제 근거 수 및 추론 부담에 맞는가?
10. answerable=false인 경우 문서 전체에 정말 답이 없는가?
11. 문항 간 의미 중복 또는 동일 근거 편중이 있는가?
12. 외부 지식, 추측, 일반화 또는 원문보다 강한 표현이 섞였는가?

[판정]
- approved_candidate: 자동·사람 검수로 넘길 수 있음
- revise: 수정하면 사용할 수 있음
- reject: 근거 오류 또는 문항 설계 오류로 다시 생성해야 함

[출력]
설명이나 Markdown 없이 JSONL만 출력한다. 입력 문항마다 다음 형식의 검증 결과 한 줄을 출력한다.

{
  "question_id": "{{DOCUMENT_ID}}_q001",
  "verdict": "approved_candidate",
  "errors": [],
  "missing_facts": [],
  "unsupported_claims": [],
  "invalid_context_ids": [],
  "suggested_question": "",
  "suggested_ground_truth": "",
  "review_notes": ""
}

오류가 없다는 확신이 없으면 approved_candidate로 판정하지 않는다. 원문에서 확인할 수 없는 내용은 추정하여 교정하지 말고 오류로 보고한다.
```

## 9. 수정 프롬프트

검증 결과가 `revise`인 항목만 수정할 때 사용한다.

```text
제공된 RFP 원문, chunks.jsonl, 기존 Golden Set 항목과 독립 검증 결과를 사용하여 revise 판정 항목만 수정하라.

검증자가 지적하지 않은 사실도 원문과 다시 대조한다. 기존 문항 수와 question_id는 유지한다. 근거가 불충분하면 추측으로 채우지 말고 문항 자체를 다른 근거 기반 문항으로 교체한다. 존재하지 않는 chunk_id를 만들지 않는다.

최종 응답에는 설명이나 Markdown 없이 수정된 JSONL 항목만 출력한다.
```

## 10. 자동 품질 검사

LLM 검증 후 최소한 다음 사항을 코드로 검사한다.

- JSONL 파싱 가능 여부와 필수 필드 누락
- `question_id` 중복과 문서별 문항 수
- `document_id`와 `source_document`의 manifest 존재 여부
- `reference_context_ids`의 실제 청크 존재 여부
- 청크의 문서 ID 및 페이지 범위 일치 여부
- `answerable=true`인데 근거·정답·필수 사실이 비어 있는 항목
- `answerable=false`인데 근거 또는 필수 사실이 들어 있는 항목
- 동일·유사 질문과 동일 근거의 과도한 재사용
- 허용되지 않은 category, query_type, difficulty, review_status
- split 간 동일 근거 또는 유사 문항 누출

자동 검사를 통과했다는 사실은 정답의 의미적 정확성을 보장하지 않는다.

## 11. 사람 검수 체크리스트

검수자는 청크만 보지 말고 실제 PDF 페이지를 함께 확인한다.

- [ ] 질문에 대한 답이 실제 원문에 존재한다.
- [ ] 정답 문서, 페이지, 섹션과 청크가 정확하다.
- [ ] `ground_truth`에 추측 또는 외부 지식이 없다.
- [ ] 질문이 요구한 핵심 사실을 모두 답한다.
- [ ] `required_facts`가 빠짐없이 독립적인 단위로 나뉘었다.
- [ ] 숫자, 날짜, 단위, 비율, 예외 조건이 정확하다.
- [ ] 질문이 원문 문장을 그대로 복사한 검색 힌트가 아니다.
- [ ] 실제 사용자가 물을 수 있는 자연스러운 표현이다.
- [ ] 같은 의미의 다른 문항과 중복되지 않는다.
- [ ] 답변 불가능 문항은 PDF 전체에 실제 답이 없다.

승인 시 다음 값을 기록한다.

```json
{
  "review_status": "approved",
  "reviewer": "검수자 식별자",
  "review_notes": "검수 또는 수정 내용"
}
```

## 12. Split 권장안

100문항을 다음처럼 나눈다.

| 구분 | 전체 문항 | 용도 |
|---|---:|---|
| 개발셋 | 60 | 청킹·검색·프롬프트 개선 |
| 검증셋 | 20 | 설정과 모델 선택 |
| 최종 테스트셋 | 20 | 확정 구성의 최종 측정 |

동일 섹션, 동일 요구사항 또는 사실상 같은 근거에서 파생된 문항은 하나의 그룹으로 묶어 같은 split에 넣는다. 최종 테스트셋의 질문과 정답을 보고 시스템을 튜닝하지 않는다.

## 13. Pilot 학습 기록

각 실패는 다음 형식으로 기록한다.

```json
{
  "question_id": "DOC_ID_q001",
  "stage": "generation|validation|retrieval|answer",
  "error_type": "wrong_page|missing_fact|false_unanswerable|duplicate|parse_error|other",
  "cause": "관찰된 직접 원인",
  "action": "프롬프트·파서·검색기·검수 규칙 중 변경할 항목",
  "regression_test": "재발 여부를 확인할 방법"
}
```

오류 한 건만 보고 전체 생성 규칙을 즉시 바꾸지 않는다. 같은 유형이 반복되는지 확인하고, 규칙 변경 전후를 동일한 개발·검증 문항으로 비교한다.

## 14. 전체 98개 문서로 확장하는 조건

다음 조건을 모두 만족한 뒤 확장한다.

- 10개 문서 100문항 생성 완료
- 모든 문항이 자동 품질 검사 통과
- 모든 문항의 사람 검수 상태 확정
- 답변 가능 문항의 근거 페이지·섹션 정확도 100%
- 존재하지 않는 청크 ID 0건
- 답변 불가능 문항의 오분류 0건
- 의미 중복 문항이 허용 기준 이하
- 검색·답변 실패 유형이 분류되고 개선 우선순위가 정해짐
- 프롬프트와 스키마 버전이 고정됨
- 개발·검증·테스트 split이 고정됨

확장 시에도 98개를 한 번에 생성하지 않는다. 5~10개 문서 단위 배치로 생성·검수하며 배치마다 품질 보고서를 남긴다.
