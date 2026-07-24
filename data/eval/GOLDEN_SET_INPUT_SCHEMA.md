# 검색 평가용 Golden set 입력 계약

검색 평가기는 JSONL을 입력으로 사용한다. 한 줄이 하나의 평가 질문이다.

## 필수 필드

```json
{
  "question_id": "상_1_q001",
  "question": "전자기록관 DB 정비 및 마이그레이션 요구사항은 무엇인가?",
  "reference_context_ids": ["상_1_p018_c0024"]
}
```

- `question_id`: 평가셋 전체에서 유일한 ID
- `question`: 검색기에 전달할 질문
- `reference_context_ids`: 정답 근거 청크 ID 목록

## 권장 필드

```json
{
  "question_id": "상_1_q001",
  "question": "전자기록관 DB 정비 및 마이그레이션 요구사항은 무엇인가?",
  "reference_context_ids": ["상_1_p018_c0024", "상_1_p019_c0025"],
  "reference_document_ids": ["상_1"],
  "reference_pages": [18, 19],
  "reference_answer": "사람이 승인한 기준 답변",
  "required_facts": ["이용자 DB 정비", "관리자 DB 정비"],
  "question_type": "multi_context",
  "difficulty": "high",
  "answerable": true,
  "review_status": "approved"
}
```

현재 검색 평가기는 context ID, document ID, page를 사용한다. 나머지 필드는 이후 답변 생성 및 RAGAS 평가에서 사용한다.

## 실행

```bash
python -m scripts.evaluate_retrieval \
  --golden data/eval/golden_set.jsonl \
  --retriever all
```

결과는 기본적으로 `reports/retrieval/`에 저장된다.

