# W&B 설치 및 RAG 연결 가이드

이 가이드의 목표는 간단한 W&B 연결임.

1. W&B와 Weave 설치
2. 현재 RAG 모델 연결
3. 질문 1회 실행 및 W&B Run/Weave Trace 확인

처음 연결하는 데 필요한 최소 코드만 사용함.

## 1. W&B 설치

프로젝트 루트에서 실행함.

```bash
uv add wandb weave
```

설치 확인:

```bash
uv run wandb --version
```

## 2. W&B 로그인

[W&B](https://wandb.ai/) 가입 후 API Key 발급.

터미널 로그인:

```bash
uv run wandb login
```

안내 화면에서 발급받은 W&B API Key 입력.

> API Key를 소스 코드나 Git 저장소에 저장하지 않음.

## 3. RAG 실행 결과를 W&B에 기록

저장소에 포함된 실행 파일:

```text
scripts/run_answer_wandb.py
```

`.env.example`을 `.env.local`로 복사하고 `WANDB_API_KEY`, `WANDB_PROJECT`, 필요 시
`WANDB_ENTITY`를 설정함. 같은 계정의 다른 프로젝트와 키를 공유해도 프로젝트 이름으로
실험이 분리됨. 키는 저장소에 커밋하지 않음.

## 4. 실행

OpenAI API Key 설정 필요. 프로젝트 루트의 `.env` 파일에 다음 내용 저장:

```env
OPENAI_API_KEY=발급받은_OpenAI_API_Key
```

질문 1회 실행:

```bash
uv run python -m scripts.run_answer_wandb \
  --query "SFR-007 예약 시스템 기능은 무엇입니까?"
```

정상 실행 시 터미널에 다음 두 항목 출력됨.

```text
답변: ...
W&B: https://wandb.ai/...
```

같은 프로젝트의 `Weave > Traces`에서 `rag.answer` 상위 호출과 OpenAI 하위
호출의 입력·출력, 지연시간, 토큰 및 비용을 확인할 수 있음.

출력된 W&B 주소를 브라우저에서 열어 실행 결과 확인.

## 5. W&B에서 확인할 내용

처음에는 아래 항목만 확인하면 충분함.

- 질문 및 답변 기록 여부
- 문서 근거 기반 답변 가능 여부
- 검색 및 답변 생성 소요 시간
- 토큰 사용량 및 예상 비용

위 항목 확인 시 W&B 기본 연결 완료임.

## 문제 해결

### `wandb: command not found`

`uv run`을 붙여 실행함.

```bash
uv run wandb login
```

### W&B 로그인을 다시 해야 하는 경우

```bash
uv run wandb login --relogin
```

### OpenAI API Key 오류

프로젝트 루트의 `.env` 파일 존재 여부 확인.

```env
OPENAI_API_KEY=...
```

`.env` 파일은 Git에 올리지 않음.

## 다음 단계

기본 연결 확인 후 필요에 따라 다음 기능 추가 가능함.

- 여러 질문의 결과 비교
- BM25, Dense, Hybrid 검색 방식 비교
- Recall, MRR 같은 검색 평가 지표 기록
- W&B Weave를 이용한 상세 호출 추적

처음부터 모든 기능을 설정할 필요 없음. 우선 질문 1회 결과의 W&B 표시 여부만 확인함.
