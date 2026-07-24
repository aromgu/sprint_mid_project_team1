# Claude Code Codex Prompt 사용 가이드
## RAG 시스템 자동 구현

---

## 📋 개요

이 가이드는 **Claude Code (agentic coding)**를 사용하여 RAG 시스템을 자동으로 구현하는 방법을 설명합니다.

**3가지 파일 제공:**
1. `rag_codex_prompt.py` - 모든 프롬프트 포함
2. `rag_prompt_manager.py` - 대화형 프롬프트 매니저
3. 이 가이드 (`CLAUDE_CODE_USAGE_GUIDE.md`)

---

## 🚀 빠른 시작

### 방법 1: 자동 프롬프트 복사 (권장)

```bash
# 모든 프롬프트 보기
python rag_prompt_manager.py

# 특정 프롬프트 복사 (클립보드)
python rag_prompt_manager.py copy phase1_implementation

# 특정 프롬프트 확인
python rag_prompt_manager.py phase1_implementation
```

### 방법 2: 수동 복사

1. 아래에서 필요한 프롬프트 복사
2. Claude Code에서 실행
3. 결과 확인

### 방법 3: 직접 실행

```bash
claude code
```

그리고 다음 프롬프트 중 하나 입력:

```
Phase 1을 구현해줄래?
```

---

## 📊 구현 순서 (권장)

### 1단계: 프로젝트 초기화 (5분)

```bash
python rag_prompt_manager.py copy initialize_project
```

**Claude Code 실행:**
```
프로젝트를 초기화해줄래?

**작업 내용:**
1. 프로젝트 폴더 구조 생성:
   rag_system/
   ├── main.py
   ├── config.yaml
   ├── core/
   ├── retrievers/
   ├── utils/
   └── tests/

2. .gitignore 생성 (Python용)
3. README.md 생성 (기본 설명)
4. 모든 __init__.py 파일 생성
```

**결과:** 폴더 구조와 기본 파일 생성

---

### 2단계: 개발 환경 설정 (10분)

```bash
python rag_prompt_manager.py copy setup_environment
```

**결과:**
- requirements.txt 생성
- .env.example 생성
- setup.py 생성
- 가상 환경 설정

---

### 3단계: PHASE 1 구현 (1일)

```bash
python rag_prompt_manager.py copy phase1_implementation
```

**구현 파일:**
- `core/pdf_processor.py` - PDF 처리
- `core/base_retriever.py` - 기본 인터페이스
- `retrievers/bm25_retriever.py` - BM25 검색
- `main.py` - 진입점
- `config_phase1.yaml` - 설정
- `tests/test_bm25.py` - 테스트

**테스트:**
```bash
pytest tests/test_bm25.py -v
```

**기대 결과:**
- 정확도 > 70%
- 응답 시간 < 1초

---

### 4단계: PHASE 2 구현 (1일)

```bash
python rag_prompt_manager.py copy phase2_implementation
```

**추가 구현 파일:**
- `utils/embeddings.py` - 임베딩 모델
- `utils/synonyms.py` - 동의어 사전
- `retrievers/hybrid_retriever.py` - 하이브리드 검색
- `tests/test_hybrid.py` - 테스트

**테스트:**
```bash
pytest tests/test_hybrid.py -v
```

**기대 결과:**
- 정확도 > 75%
- 응답 시간 < 2초

---

### 5단계: PHASE 3 구현 (2일)

```bash
python rag_prompt_manager.py copy phase3_implementation
```

**추가 구현 파일:**
- `utils/reranker.py` - LLM Re-ranking
- `utils/query_expansion.py` - 쿼리 변환
- `retrievers/advanced_retriever.py` - 고도화 RAG
- `tests/test_advanced.py` - 테스트

**환경 변수 설정:**
```bash
export ANTHROPIC_API_KEY=sk-...
export PINECONE_API_KEY=...
```

**테스트:**
```bash
pytest tests/test_advanced.py -v
```

**기대 결과:**
- 정확도 > 80%
- 신뢰도 > 0.8
- 응답 시간 < 3초

---

## 💻 Claude Code 사용법

### 시작하기

```bash
claude code
```

이 명령어는 VS Code 또는 Terminal에서 Claude Code 인터페이스를 엽니다.

### 프롬프트 입력

복사한 프롬프트를 Claude Code에 붙여넣으면:

1. Claude가 코드를 자동으로 생성
2. 파일을 자동으로 생성/수정
3. 필요시 Python 코드 실행
4. 테스트 실행
5. 결과 보고

### 예시 인터랙션

**You:**
```
Phase 1을 구현해줄래?

**목표:**
- 정확도 > 70%
- 응답 시간 < 1초
- 3개 문서 테스트

**구현 파일:**
1. core/pdf_processor.py
2. core/base_retriever.py
3. retrievers/bm25_retriever.py
4. main.py
5. config_phase1.yaml
6. tests/test_bm25.py

각 파일을 순서대로 생성해줄래.
```

**Claude:**
```
✅ 파일 생성을 시작하겠습니다.

1️⃣ core/pdf_processor.py 생성 중...
   - extract_text() 구현
   - chunk_text() 구현
   ✓ 완료

2️⃣ core/base_retriever.py 생성 중...
   - BaseRetriever 추상 클래스
   - index() 메서드
   - retrieve() 메서드
   ✓ 완료

3️⃣ retrievers/bm25_retriever.py 생성 중...
   - BM25Retriever 클래스
   - BM25 인덱싱
   - 검색 구현
   ✓ 완료

4️⃣ tests/test_bm25.py 생성 중...
   - test_pdf_extraction()
   - test_chunking()
   - test_bm25_indexing()
   - test_search_accuracy()
   ✓ 완료

5️⃣ 테스트 실행 중...
   pytest tests/test_bm25.py -v
   ✓ 모든 테스트 통과!

PHASE 1 구현 완료! 🎉
```

---

## 📁 파일 구조

구현 완료 후 폴더 구조:

```
rag_system/
├── main.py                              # 진입점
├── config.yaml                          # 기본 설정
├── config_phase1.yaml                   # Phase 1 설정
├── config_phase2.yaml                   # Phase 2 설정
├── config_phase3.yaml                   # Phase 3 설정
├── requirements.txt                     # 의존성
├── .env.example                         # 환경 변수 예시
├── .gitignore                           # Git 제외 파일
├── README.md                            # 프로젝트 설명
│
├── core/                                # 공통 모듈
│   ├── __init__.py
│   ├── pdf_processor.py                 # PDF 처리
│   ├── chunker.py                       # 청킹
│   └── base_retriever.py                # 기본 인터페이스
│
├── retrievers/                          # 리트리버 구현
│   ├── __init__.py
│   ├── bm25_retriever.py                # Phase 1
│   ├── hybrid_retriever.py              # Phase 2
│   └── advanced_retriever.py            # Phase 3
│
├── utils/                               # 유틸리티
│   ├── __init__.py
│   ├── embeddings.py                    # 임베딩
│   ├── synonyms.py                      # 동의어
│   ├── query_expansion.py               # 쿼리 변환
│   ├── reranker.py                      # Re-ranking
│   └── metrics.py                       # 평가 메트릭
│
└── tests/                               # 테스트
    ├── __init__.py
    ├── test_bm25.py                     # Phase 1 테스트
    ├── test_hybrid.py                   # Phase 2 테스트
    └── test_advanced.py                 # Phase 3 테스트
```

---

## 🧪 테스트 실행

각 Phase 완료 후:

```bash
# Phase 1 테스트
pytest tests/test_bm25.py -v

# Phase 2 테스트
pytest tests/test_hybrid.py -v

# Phase 3 테스트
pytest tests/test_advanced.py -v

# 모든 테스트
pytest tests/ -v
```

---

## 📈 성능 검증

### 평가 메트릭

```python
from utils.metrics import evaluate_rag_system

results = evaluate_rag_system(
    test_queries=test_data,
    retriever=retriever,
    ground_truth=answers
)

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Precision@3: {results['precision@3']:.2%}")
print(f"Response Time: {results['avg_time']:.2f}s")
print(f"Confidence: {results['confidence']:.2f}")
```

### 단계별 성능 목표

| Phase | 정확도 | 응답시간 | 신뢰도 | 상태 |
|-------|--------|---------|--------|------|
| 1 | > 70% | < 1s | N/A | ✓ |
| 2 | > 75% | < 2s | N/A | ✓ |
| 3 | > 80% | < 3s | > 0.8 | ✓ |

---

## 🔧 트러블슈팅

### 문제 1: "ModuleNotFoundError"

**원인:** 의존성 설치 안 됨

**해결:**
```bash
pip install -r requirements.txt
```

### 문제 2: "PDF 파일을 찾을 수 없음"

**원인:** 파일 경로 오류

**해결:**
```bash
# 파일 경로 확인
ls /path/to/pdf/files/

# 코드에서 경로 수정
text = processor.extract_text("절대경로/파일명.pdf")
```

### 문제 3: "임베딩 모델 다운로드 실패"

**원인:** 네트워크 오류 또는 저장소 접근 불가

**해결:**
```bash
# 모델 수동 다운로드
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
"
```

### 문제 4: "Pinecone 연결 실패"

**원인:** API 키 오류 또는 환경 변수 미설정

**해결:**
```bash
# 환경 변수 확인
echo $PINECONE_API_KEY

# .env 파일 생성 (.env.example 참고)
cp .env.example .env
# 파일 편집후
source .env
```

---

## 💡 팁 & 트릭

### 1. 점진적 구현

한 번에 전체를 구현하기보다 한 단계씩 구현하고 테스트하세요.

```bash
# Good
claude code
# "Phase 1을 구현해줄래?"
pytest tests/test_bm25.py -v
# "Phase 2를 구현해줄래?"
pytest tests/test_hybrid.py -v

# Bad
claude code
# "전체를 구현해줄래?"  # 너무 많음!
```

### 2. 명확한 요구사항

프롬프트를 명확하게 작성하세요.

```
❌ 나쁜 예:
"RAG를 구현해줄래?"

✅ 좋은 예:
"PHASE 1 (BM25 검색)을 구현해줄래?
**구현 파일:**
1. core/pdf_processor.py
2. retrievers/bm25_retriever.py
3. tests/test_bm25.py

**테스트 쿼리:**
- '농산물 가격'
- '보안 모듈'

각 파일을 순서대로 생성해줄래."
```

### 3. 파일별 분리 요청

큰 파일은 여러 개의 작은 프롬프트로 나누세요.

```
1. "core/pdf_processor.py를 생성해줄래?"
2. 테스트 실행
3. "core/base_retriever.py를 생성해줄래?"
4. 테스트 실행
```

### 4. 피드백 반영

Claude는 피드백에 반응합니다.

```
"테스트 결과를 보니 성능이 60%입니다.
동의어 사전을 확장해줄래?
다음 동의어를 추가해줄래:
'수강신청': ['신청', '등록', '신청 절차']
'학점': ['점수', '성적']
..."
```

---

## 📝 주요 프롬프트 목록

### 필수 프롬프트

```
1. initialize_project
   → 폴더 구조 생성

2. setup_environment
   → 개발 환경 설정

3. phase1_implementation
   → BM25 검색 구현

4. phase2_implementation
   → 하이브리드 검색 구현

5. phase3_implementation
   → 고도화 RAG 구현
```

### 선택 프롬프트

```
6. generate_test_dataset
   → 테스트 데이터 생성

7. performance_optimization
   → 성능 최적화

8. deploy_to_production
   → 프로덕션 배포
```

---

## 🎯 완료 체크리스트

- [ ] Phase 1 구현 및 테스트 통과 (정확도 > 70%)
- [ ] Phase 2 구현 및 테스트 통과 (정확도 > 75%)
- [ ] Phase 3 구현 및 테스트 통과 (정확도 > 80%)
- [ ] 통합 테스트 완료
- [ ] 성능 최적화 완료
- [ ] 문서화 완료
- [ ] 프로덕션 배포 준비

---

## 📚 참고 자료

- **Prompt Engineering Guide**: `RAG_IMPLEMENTATION_ROADMAP.md`
- **System Architecture**: `RAG_SYSTEM_ARCHITECTURE.md`
- **API Documentation**: 구현 완료 후 생성

---

## 🤝 지원

문제가 발생하면:

1. Claude Code에서 직접 물어보기
   ```
   "왜 이 테스트가 실패했어?"
   ```

2. 에러 메시지와 함께 다시 요청
   ```
   "ModuleNotFoundError: No module named 'rank_bm25'
    어떻게 해야 해?"
   ```

3. 프롬프트 매니저에서 도움말 보기
   ```bash
   python rag_prompt_manager.py --help
   ```

---

**Happy Coding! 🚀**
