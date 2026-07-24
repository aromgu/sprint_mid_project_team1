#!/usr/bin/env python3
"""
RAG 시스템 구현 Codex Prompt
===========================
Claude Code (agentic coding)를 위한 자동 코드 생성 프롬프트

사용법:
1. Claude Code 터미널에서 실행:
   claude code

2. 다음 프롬프트 중 하나를 사용:
   - Phase 1 구현
   - Phase 2 구현
   - Phase 3 구현
   - 전체 구현
"""

CODEX_PROMPTS = {
    "initialize_project": """
프로젝트를 초기화해줄래?

**작업 내용:**
1. 프로젝트 폴더 구조 생성:
   ```
   rag_system/
   ├── main.py
   ├── config.yaml
   ├── config_phase1.yaml
   ├── config_phase2.yaml
   ├── config_phase3.yaml
   ├── requirements.txt
   ├── core/
   │   ├── __init__.py
   │   ├── pdf_processor.py
   │   ├── chunker.py
   │   └── base_retriever.py
   ├── retrievers/
   │   ├── __init__.py
   │   ├── bm25_retriever.py
   │   ├── hybrid_retriever.py
   │   └── advanced_retriever.py
   ├── utils/
   │   ├── __init__.py
   │   ├── embeddings.py
   │   ├── synonyms.py
   │   ├── reranker.py
   │   └── metrics.py
   └── tests/
       ├── __init__.py
       ├── test_bm25.py
       ├── test_hybrid.py
       └── test_advanced.py
   ```

2. .gitignore 생성 (Python용)

3. README.md 생성 (기본 설명)

4. 모든 __init__.py 파일 생성

**기술 스택:**
- Python 3.9+
- PyPDF2: PDF 처리
- rank-bm25: BM25 검색
- sentence-transformers: 임베딩
- pinecone-client: 벡터 DB (선택)
- anthropic: Claude API (선택)
""",

    "phase1_implementation": """
PHASE 1 (난이도 하 - BM25 검색)을 구현해줄래?

**목표:**
- 정확도 > 70%
- 응답 시간 < 1초
- 3개 문서 테스트

**구현 파일:**

1. core/pdf_processor.py
   - extract_text(pdf_path): PDF에서 텍스트 추출
   - chunk_text(text, chunk_size=300, overlap=50): 텍스트 청킹
   
2. core/base_retriever.py
   - BaseRetriever 추상 클래스
   - index(chunks): 청크 인덱싱
   - retrieve(query, top_k=3): 검색

3. retrievers/bm25_retriever.py
   - BM25Retriever(BaseRetriever)
   - BM25 인덱싱
   - 기본 검색 (Top-3 반환)

4. main.py
   - 설정 로드
   - PDF 처리
   - 검색 실행

5. config_phase1.yaml
   - retriever_type: "bm25"
   - BM25 설정 (chunk_size, overlap)

6. requirements.txt
   - PyPDF2
   - rank-bm25
   - pyyaml

7. tests/test_bm25.py
   - test_pdf_extraction()
   - test_chunking()
   - test_bm25_indexing()
   - test_search_accuracy()

**테스트 데이터:**
- 한국연구재단_2024년_대학산학협력활동_실태조사_시스템UICC_기능개선.pdf
- 한국생산기술연구원 고압가스 안전
- 한국전기안전공사 전기안전 관제

**테스트 쿼리:**
- "시스템 기능 요구사항"
- "안전 관리 기준"
- "보안 모듈 스펙"

각 파일을 순서대로 생성해줄래.
""",

    "phase2_implementation": """
PHASE 2 (난이도 중 - 하이브리드 검색)을 구현해줄래?

**목표:**
- 정확도 > 75%
- 응답 시간 < 2초
- 3개 문서 테스트

**추가 구현 파일:**

1. utils/embeddings.py
   - get_embeddings(model_name): 임베딩 모델 로드
   - encode(texts): 텍스트 벡터화
   - 사용 모델: sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens

2. utils/synonyms.py
   - load_synonyms(): 동의어 사전 로드
   - expand_query(query): 쿼리 확장
   - 동의어 사전 작성 (각 문서별 50+ 항목)
   
   예시:
   {
     "수강신청": ["신청", "등록", "수강", "신청 절차"],
     "학점": ["점수", "성적", "학위", "학점 인정"],
     "도시계획": ["도시", "계획", "규제", "도시 설계"]
   }

3. retrievers/hybrid_retriever.py
   - HybridRetriever(BM25Retriever) 상속
   - BM25 + 벡터 검색 결합
   - 점수 통합 (BM25 30%, 벡터 70%)
   - 동의어 쿼리 확장

4. config_phase2.yaml
   - retriever_type: "hybrid"
   - embedding.enabled: true
   - embedding.model: "sentence-transformers/..."
   - synonyms.enabled: true

5. tests/test_hybrid.py
   - test_embedding_model()
   - test_synonym_expansion()
   - test_hybrid_search()
   - test_semantic_similarity()
   - test_search_accuracy()

**테스트 데이터:**
- 한영대학_한영대학교_특성화_맞춤형_교육환경_구축__트랙운영_학사정보.pdf
- 인천광역시_도시계획위원회_통합관리시스템_구축용역.pdf
- 스포츠윤리센터 LMS

**테스트 쿼리:**
- "학생 수강 등록 절차"
- "도시 규제 및 제한"
- "학습 콘텐츠 관리"

동의어 사전을 풍부하게 작성하고, 하이브리드 검색이 제대로 작동하는지 확인해줄래.
""",

    "phase3_implementation": """
PHASE 3 (난이도 상 - 고도화)을 구현해줄래?

**목표:**
- 정확도 > 80%
- 신뢰도 > 0.8
- 응답 시간 < 3초
- 3개 문서 테스트

**추가 구현 파일:**

1. utils/reranker.py
   - LLMReranker 클래스
   - rerank(query, candidates, top_k=3): LLM으로 재정렬
   - Claude API 사용:
     model: "claude-3-5-sonnet-20241022"
     prompt: 쿼리와 후보의 관련성 평가

2. retrievers/advanced_retriever.py
   - AdvancedRetriever(HybridRetriever) 상속
   - Pinecone 벡터 DB 연동
   - 고급 쿼리 변환:
     - 동의어 확장
     - 문법적 변형
     - 엔티티 조합
   - LLM 기반 Re-ranking
   - 신뢰도 점수 계산

3. utils/query_expansion.py
   - advanced_query_expansion(query, domain_dict)
   - generate_grammatical_variants(query)
   - extract_domain_entities(query)
   - 도메인별 용어 사전 구축

4. config_phase3.yaml
   - retriever_type: "advanced"
   - advanced.enabled: true
   - advanced.pinecone_api_key: "${PINECONE_API_KEY}"
   - advanced.use_reranking: true
   - advanced.use_image_ocr: true
   - advanced.llm_model: "claude-3-5-sonnet-20241022"

5. tests/test_advanced.py
   - test_pinecone_upload()
   - test_query_expansion()
   - test_reranking()
   - test_confidence_score()
   - test_search_accuracy()
   - test_response_time()

**테스트 데이터:**
- 한국농수산식품유통공사_농산물가격안정기금_정부예산회계연계시스템_.pdf
- 한국수자원공사_건설통합시스템CMS_고도화.pdf
- 국가과학기술지식정보서비스_통합정보시스템_고도화_용역.pdf

**테스트 쿼리:**
- "농산물 가격 결산 처리 절차"
- "건설 프로젝트 통합 관리 프로세스"
- "학제간 정보 표준화 방안"

**구현 순서:**
1. Pinecone 설정 및 테스트
2. 고급 쿼리 변환 구현
3. Re-ranker 구현
4. AdvancedRetriever 완성
5. 통합 테스트

환경 변수 설정:
- ANTHROPIC_API_KEY
- PINECONE_API_KEY
- PINECONE_ENVIRONMENT

다 해줄래?
""",

    "setup_environment": """
RAG 시스템을 위한 개발 환경을 설정해줄래?

**작업:**

1. requirements.txt 생성:
   ```
   PyPDF2==4.0.1
   rank-bm25==0.2.2
   sentence-transformers==2.2.2
   numpy==1.24.3
   scikit-learn==1.3.0
   pyyaml==6.0
   python-dotenv==1.0.0
   pytest==7.4.0
   anthropic>=0.7.0  # PHASE 3용
   pinecone-client>=2.2.0  # PHASE 3용
   pymupdf==1.23.8  # PHASE 3용 (이미지 처리)
   ```

2. .env.example 생성:
   ```
   # Claude API
   ANTHROPIC_API_KEY=sk-...
   
   # Pinecone (PHASE 3용)
   PINECONE_API_KEY=...
   PINECONE_ENVIRONMENT=us-west4-gcp
   PINECONE_INDEX=rag-system
   
   # 모델 설정
   EMBEDDING_MODEL=sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens
   LLM_MODEL=claude-3-5-sonnet-20241022
   ```

3. setup.py 생성 (패키지 설치용)

4. .gitignore 업데이트:
   - .env (환경 변수)
   - __pycache__/
   - *.pyc
   - .pytest_cache/
   - venv/
   - embeddings/
   - indexes/

5. Docker 설정 (선택):
   - Dockerfile
   - docker-compose.yml

6. 가상 환경 설정:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

7. 환경 변수 설정 테스트

8. 모든 모듈 import 테스트

다 설정해줄래?
""",

    "full_implementation": """
RAG 시스템 전체를 구현해줄래? (Phase 1 → 2 → 3)

**작업 순서:**

1. 프로젝트 초기화
   - 폴더 구조 생성
   - requirements.txt 작성
   - 환경 변수 설정

2. PHASE 1 완성 (BM25)
   - core/ 모듈 구현
   - BM25Retriever 구현
   - 테스트 및 검증

3. PHASE 2 완성 (하이브리드)
   - 임베딩 모델 통합
   - 동의어 사전 구축
   - HybridRetriever 구현
   - 테스트 및 검증

4. PHASE 3 완성 (고도화)
   - Pinecone 통합
   - 고급 쿼리 변환
   - LLM Re-ranking
   - AdvancedRetriever 구현
   - 테스트 및 검증

5. 통합 테스트
   - 3가지 리트리버 모두 테스트
   - 성능 비교
   - 최적화

6. 문서화
   - API 문서
   - 사용 예제
   - 성능 리포트

**목표:**
- PHASE 1: 정확도 > 70%
- PHASE 2: 정확도 > 75%
- PHASE 3: 정확도 > 80%, 신뢰도 > 0.8

각 단계별로 진행하면서 매번 테스트를 통과해야 다음 단계로 넘어가게 될거야.
시작할래?
""",

    "generate_test_dataset": """
RAG 시스템 테스트를 위한 테스트 데이터셋을 준비해줄래?

**작업:**

1. test_queries.json 생성
   ```json
   {
     "phase1": [
       {
         "query": "시스템 기능 요구사항",
         "document": "한국연구재단_UICC",
         "expected_answer": "...",
         "difficulty": "easy"
       },
       ...
     ],
     "phase2": [...],
     "phase3": [...]
   }
   ```

2. ground_truth.json 생성 (정답 데이터셋)
   - 각 쿼리에 대한 올바른 답변
   - 여러 정답 변형 포함
   - 신뢰도 점수

3. test_documents.json 생성
   - 각 PDF에서 추출한 주요 청크
   - 섹션 정보
   - 페이지 정보

4. metrics.py 업데이트
   - accuracy: 정답 포함 여부
   - precision@k: Top-K 정확도
   - recall: 재현율
   - f1_score: F1 점수
   - mrr: Mean Reciprocal Rank
   - ndcg: Normalized DCG

5. benchmark.py 생성
   - 각 단계별 성능 측정
   - 성능 비교 리포트
   - 시각화 (matplotlib)

6. test_runner.py 생성
   - 자동 테스트 실행
   - 결과 리포트 생성
   - CI/CD 연동

다 준비해줄래?
""",

    "performance_optimization": """
RAG 시스템의 성능을 최적화해줄래?

**최적화 항목:**

1. 검색 성능 최적화
   - BM25 파라미터 튜닝
   - 청킹 크기 최적화
   - 인덱싱 효율 개선

2. 임베딩 성능 최적화
   - 배치 처리
   - 캐싱
   - GPU 사용 (CUDA)

3. Re-ranking 최적화
   - 캐시 활용
   - 병렬 처리
   - API 호출 최소화

4. 메모리 최적화
   - 불필요한 데이터 제거
   - 메모리 누수 체크
   - 모델 양자화

5. 응답 시간 최적화
   - 비동기 처리
   - 프리페칭
   - 병렬 검색

**성능 목표:**
- PHASE 1: < 1초
- PHASE 2: < 2초
- PHASE 3: < 3초

프로파일링과 성능 측정을 통해 병목 지점을 찾고 최적화해줄래?
""",

    "deploy_to_production": """
RAG 시스템을 프로덕션에 배포해줄래?

**배포 작업:**

1. Docker 이미지 생성
   - Dockerfile 작성
   - 이미지 빌드 및 테스트
   - 레지스트리에 푸시

2. API 서버 구축
   - FastAPI/Flask 서버
   - REST 엔드포인트
   - 에러 처리
   - 로깅

3. 배포 설정
   - docker-compose.yml
   - kubernetes 설정 (선택)
   - 헬스 체크

4. 모니터링 설정
   - 성능 메트릭
   - 에러 로깅
   - 알림

5. CI/CD 파이프라인
   - GitHub Actions
   - 자동 테스트
   - 자동 배포

6. 문서화
   - API 문서
   - 배포 가이드
   - 트러블슈팅

배포 준비 다 해줄래?
"""
}

def print_prompts():
    """사용 가능한 프롬프트 목록 출력"""
    print("=" * 80)
    print("RAG 시스템 구현 - Claude Code Codex Prompts")
    print("=" * 80)
    print("\n사용 가능한 프롬프트:\n")
    
    for i, (key, prompt) in enumerate(CODEX_PROMPTS.items(), 1):
        title = key.replace('_', ' ').title()
        first_line = prompt.split('\n')[0]
        print(f"{i}. {title}")
        print(f"   {first_line}")
        print()
    
    print("=" * 80)
    print("사용 방법:")
    print("=" * 80)
    print("""
1. Claude Code 열기:
   claude code

2. 다음 중 하나를 복사해서 붙여넣기:

   # 초기 설정 먼저
   - initialize_project
   - setup_environment

   # 단계별 구현
   - phase1_implementation
   - phase2_implementation
   - phase3_implementation

   # 또는 전체 한번에
   - full_implementation

   # 추가 작업
   - generate_test_dataset
   - performance_optimization
   - deploy_to_production

3. 프롬프트 실행
   각 프롬프트를 Claude Code에 붙여넣으면
   자동으로 코드 생성 및 파일 작성이 진행됩니다.
""")

if __name__ == "__main__":
    print_prompts()
    
    # 개별 프롬프트 출력 옵션
    import sys
    if len(sys.argv) > 1:
        key = sys.argv[1]
        if key in CODEX_PROMPTS:
            print("\n" + "=" * 80)
            print(f"Prompt: {key.upper()}")
            print("=" * 80)
            print(CODEX_PROMPTS[key])
        else:
            print(f"Unknown prompt: {key}")
            print(f"Available prompts: {', '.join(CODEX_PROMPTS.keys())}")
