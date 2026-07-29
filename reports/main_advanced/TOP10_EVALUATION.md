# Main Advanced Top-10 재평가

## 실행 조건

- 답변 생성: `gpt-5-nano`, Top-10, 4 workers, 104문항
- 답변 판정: `answered / partially_answered / unanswerable`
- 공식 RAGAS: `gpt-4o-mini`, 8 workers, batch 16
- 기존 Top-5 산출물은 보존하고 Top-10 산출물을 별도 생성

## 3단계 판정 결과

| 상태 | 문항 수 |
|---|---:|
| answered | 88 |
| partially_answered | 14 |
| unanswerable | 2 |

Golden 기준 답변 가능 문항을 잘못 거절한 수는 기존 17건에서 1건으로 감소했다.

## Top-5 대비 Top-10

| 지표 | Top-5 | Top-10 | 변화 |
|---|---:|---:|---:|
| Answerability accuracy | 0.7885 | 0.9135 | +0.1250 |
| Required fact coverage | 0.6079 | 0.5991 | -0.0088 |
| Numeric recall | 0.7003 | 0.7085 | +0.0082 |
| Numeric precision | 0.5893 | 0.6265 | +0.0372 |
| Ground-truth token coverage | 0.3761 | 0.4067 | +0.0305 |
| RAGAS Faithfulness | 0.8142 | 0.8479 | +0.0337 |
| RAGAS Answer Relevancy | 0.3364 | 0.3742 | +0.0378 |

## 해석

Top-10과 3단계 판정으로 부분 답변의 전체 거절 문제가 크게 줄었고, faithfulness,
relevancy, 숫자 precision과 ground-truth coverage가 개선됐다. Required fact coverage는
0.88%p 하락했으므로 Top-10 자체가 multi-fact 완전성을 자동 보장하지는 않는다.
다음 개선은 hard/multihop 질문의 query decomposition 또는 fact별 retrieval이다.

## 산출물

- `answers_top10.jsonl`
- `answer_summary_top10.json`
- `ragas_top10.json`
- `ragas_details_top10.jsonl`
