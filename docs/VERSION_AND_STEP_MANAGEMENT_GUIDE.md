# RAG MVP 버전·단계 관리 가이드

RAG·LLM 프로젝트는 요구사항이 애매하거나 프롬프트·검색 설정 실험이 반복되므로, 코드 버전과 기능 단계 버전을 분리해 관리한다.

## 1. 기능 단계별 Git checkpoint

각 단계가 정상 동작할 때 태그를 만든다.

```text
v0.1-data-pipeline
v0.2-search
v0.3-rag-generation
v0.4-handover-backend
v0.5-handover-frontend
v0.6-usability-fixes
```

예:

```bash
git tag v0.6-usability-fixes
```

## 2. 작업 단위별 작은 커밋

하나의 커밋에는 하나의 목적만 둔다.

```text
fix: overview empty action items display
fix: eligibility empty-result fallback
perf: prewarm dense embedding model
ui: clear document search after page selection
```

작은 커밋은 특정 변경만 검토하거나 되돌리기 쉽게 한다.

## 3. 실험 코드는 별도 브랜치

모델, 프롬프트, 검색 알고리즘을 시험할 때는 별도 브랜치를 사용한다.

```bash
git switch -c experiment/overview-prompt
```

실험 결과가 검증된 경우에만 기본 브랜치에 반영한다.

## 4. 설정은 코드와 분리

검색기, reranker, 모델, context 크기 등은 YAML 설정으로 관리한다.

```yaml
retriever: hybrid
reranker: null
model: gpt-5-nano
max_context_chars: 12000
```

실험이 실패하면 코드 수정 없이 설정만 이전 값으로 되돌릴 수 있다.

## 5. 애매한 요구사항은 결정 로그로 남김

`DECISIONS.md`에 결정 사항과 이유를 기록한다.

```markdown
## D-003 Overview 마감일 표시

- 결정: RFP 본문에 없으면 "입찰공고 참조" 표시
- 이유: 추측 날짜를 표시하지 않기 위해
- 적용일: 2026-07-22
```

이후 구현 방향이 바뀌어도 왜 해당 동작을 선택했는지 추적할 수 있다.

## 6. 문제 발생 시 안전한 rollback

최근 커밋과 태그를 먼저 확인한다.

```bash
git log --oneline --decorate -15
```

안정 버전을 임시로 확인한다.

```bash
git switch --detach v0.5-handover-frontend
```

현재 작업과 안정 버전의 차이를 확인한다.

```bash
git diff v0.5-handover-frontend..HEAD
```

`git reset --hard`는 작업물을 잃을 수 있으므로 먼저 태그·브랜치로 안정 상태를 보존한 뒤 사용한다. 일반적으로는 새 revert 커밋을 만드는 방식이 더 안전하다.

## 7. 권장 운영 순서

```text
요구사항 결정
    ↓
DECISIONS.md 기록
    ↓
작은 기능 구현
    ↓
자동 테스트 실행
    ↓
작은 커밋 생성
    ↓
안정 상태에서 Git tag 생성
    ↓
다음 단계 진행
```

이 구조를 사용하면 애매한 요구사항으로 잘못된 방향으로 진행해도 마지막 정상 단계로 빠르게 돌아갈 수 있다.
