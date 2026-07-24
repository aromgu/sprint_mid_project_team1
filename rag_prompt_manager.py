#!/usr/bin/env python3
"""
Claude Code RAG 시스템 프롬프트 매니저
=====================================

사용법:
  python rag_prompt_manager.py              # 대화형 모드
  python rag_prompt_manager.py phase1       # 특정 프롬프트 실행
  python rag_prompt_manager.py copy phase1  # 프롬프트 복사 (클립보드)
"""

import sys
import pyperclip
from typing import Dict, List, Tuple
from enum import Enum

class Phase(Enum):
    INITIALIZE = "initialize_project"
    SETUP = "setup_environment"
    PHASE1 = "phase1_implementation"
    PHASE2 = "phase2_implementation"
    PHASE3 = "phase3_implementation"
    FULL = "full_implementation"
    TESTS = "generate_test_dataset"
    OPTIMIZE = "performance_optimization"
    DEPLOY = "deploy_to_production"

PROMPTS: Dict[str, Dict] = {
    "initialize_project": {
        "title": "프로젝트 초기화",
        "description": "RAG 시스템 폴더 구조 생성",
        "duration": "5분",
        "dependencies": []
    },
    
    "setup_environment": {
        "title": "개발 환경 설정",
        "description": "requirements.txt, .env 파일 설정",
        "duration": "10분",
        "dependencies": ["initialize_project"]
    },
    
    "phase1_implementation": {
        "title": "PHASE 1: BM25 검색 구현",
        "description": "난이도 하 - 기본 키워드 검색 (정확도 > 70%)",
        "duration": "1일",
        "dependencies": ["initialize_project", "setup_environment"],
        "target_accuracy": "70%",
        "target_time": "< 1초"
    },
    
    "phase2_implementation": {
        "title": "PHASE 2: 하이브리드 검색 구현",
        "description": "난이도 중 - BM25 + 벡터 검색 (정확도 > 75%)",
        "duration": "1일",
        "dependencies": ["phase1_implementation"],
        "target_accuracy": "75%",
        "target_time": "< 2초"
    },
    
    "phase3_implementation": {
        "title": "PHASE 3: 고도화 RAG 구현",
        "description": "난이도 상 - Pinecone + Re-ranking (정확도 > 80%)",
        "duration": "2일",
        "dependencies": ["phase2_implementation"],
        "target_accuracy": "80%",
        "target_confidence": "> 0.8"
    },
    
    "full_implementation": {
        "title": "전체 RAG 시스템 구현",
        "description": "Phase 1, 2, 3을 순차적으로 모두 구현",
        "duration": "3.5주",
        "dependencies": [],
        "note": "각 단계별 테스트를 통과해야 다음 단계로 진행"
    },
    
    "generate_test_dataset": {
        "title": "테스트 데이터셋 생성",
        "description": "테스트 쿼리, 정답 데이터, 성능 메트릭",
        "duration": "1일",
        "dependencies": ["initialize_project"]
    },
    
    "performance_optimization": {
        "title": "성능 최적화",
        "description": "검색 속도, 정확도, 메모리 최적화",
        "duration": "2일",
        "dependencies": ["phase1_implementation", "phase2_implementation", "phase3_implementation"]
    },
    
    "deploy_to_production": {
        "title": "프로덕션 배포",
        "description": "Docker, API 서버, CI/CD 설정",
        "duration": "2일",
        "dependencies": ["phase1_implementation", "phase2_implementation", "phase3_implementation"]
    }
}

PROMPTS_TEXT = {
    "initialize_project": """프로젝트를 초기화해줄래?

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

4. 모든 __init__.py 파일 생성""",
    
    "setup_environment": """RAG 시스템을 위한 개발 환경을 설정해줄래?

**작업:**

1. requirements.txt 생성:
   - PyPDF2==4.0.1
   - rank-bm25==0.2.2
   - sentence-transformers==2.2.2
   - numpy==1.24.3
   - scikit-learn==1.3.0
   - pyyaml==6.0
   - python-dotenv==1.0.0
   - pytest==7.4.0
   - anthropic>=0.7.0
   - pinecone-client>=2.2.0

2. .env.example 생성
3. setup.py 생성
4. .gitignore 업데이트
5. 가상 환경 설정
6. 환경 변수 설정 테스트
7. 모든 모듈 import 테스트

다 설정해줄래?""",

    "phase1_implementation": """PHASE 1 (난이도 하 - BM25 검색)을 구현해줄래?

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
   - BM25 설정

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

각 파일을 순서대로 생성해줄래.""",

    "phase2_implementation": """PHASE 2 (난이도 중 - 하이브리드 검색)을 구현해줄래?

**목표:**
- 정확도 > 75%
- 응답 시간 < 2초
- 3개 문서 테스트

**추가 구현 파일:**

1. utils/embeddings.py
   - get_embeddings(model_name): 임베딩 모델 로드
   - encode(texts): 텍스트 벡터화

2. utils/synonyms.py
   - load_synonyms(): 동의어 사전 로드
   - expand_query(query): 쿼리 확장
   - 동의어 사전 작성 (각 문서별 50+ 항목)

3. retrievers/hybrid_retriever.py
   - HybridRetriever(BM25Retriever) 상속
   - BM25 + 벡터 검색 결합
   - 점수 통합 (BM25 30%, 벡터 70%)
   - 동의어 쿼리 확장

4. config_phase2.yaml
   - retriever_type: "hybrid"
   - embedding.enabled: true

5. tests/test_hybrid.py
   - test_embedding_model()
   - test_synonym_expansion()
   - test_hybrid_search()
   - test_search_accuracy()

**테스트 데이터:**
- 한영대학 학사정보
- 인천광역시 도시계획위원회
- 스포츠윤리센터 LMS

동의어 사전을 풍부하게 작성하고, 하이브리드 검색이 제대로 작동하는지 확인해줄래.""",

    "phase3_implementation": """PHASE 3 (난이도 상 - 고도화)을 구현해줄래?

**목표:**
- 정확도 > 80%
- 신뢰도 > 0.8
- 응답 시간 < 3초
- 3개 문서 테스트

**추가 구현 파일:**

1. utils/reranker.py
   - LLMReranker 클래스
   - rerank(query, candidates, top_k=3)
   - Claude API 사용

2. retrievers/advanced_retriever.py
   - AdvancedRetriever(HybridRetriever) 상속
   - Pinecone 벡터 DB 연동
   - 고급 쿼리 변환
   - LLM 기반 Re-ranking
   - 신뢰도 점수 계산

3. utils/query_expansion.py
   - advanced_query_expansion()
   - generate_grammatical_variants()
   - extract_domain_entities()

4. config_phase3.yaml
   - retriever_type: "advanced"
   - advanced.enabled: true
   - 환경 변수 설정

5. tests/test_advanced.py
   - test_pinecone_upload()
   - test_query_expansion()
   - test_reranking()
   - test_confidence_score()
   - test_search_accuracy()

**테스트 데이터:**
- 한국농수산 농산물기금
- 한국수자원공사 CMS
- 국가과학기술 통합정보시스템

환경 변수 설정:
- ANTHROPIC_API_KEY
- PINECONE_API_KEY

다 해줄래?""",

    "full_implementation": """RAG 시스템 전체를 구현해줄래? (Phase 1 → 2 → 3)

**작업 순서:**

1. 프로젝트 초기화
2. PHASE 1 완성 (BM25)
3. PHASE 2 완성 (하이브리드)
4. PHASE 3 완성 (고도화)
5. 통합 테스트
6. 문서화

**목표:**
- PHASE 1: 정확도 > 70%
- PHASE 2: 정확도 > 75%
- PHASE 3: 정확도 > 80%, 신뢰도 > 0.8

각 단계별로 진행하면서 매번 테스트를 통과해야 다음 단계로 넘어가게 될거야.
시작할래?""",

    "generate_test_dataset": """RAG 시스템 테스트를 위한 테스트 데이터셋을 준비해줄래?

**작업:**

1. test_queries.json 생성
   - 각 phase별 쿼리 셋
   - 정답 데이터

2. ground_truth.json 생성
   - 각 쿼리에 대한 올바른 답변
   - 신뢰도 점수

3. test_documents.json 생성
   - 주요 청크
   - 섹션 정보

4. metrics.py 업데이트
   - accuracy, precision, recall
   - f1_score, mrr, ndcg

5. benchmark.py 생성
   - 성능 측정
   - 리포트 생성

6. test_runner.py 생성
   - 자동 테스트 실행
   - CI/CD 연동

다 준비해줄래?""",

    "performance_optimization": """RAG 시스템의 성능을 최적화해줄래?

**최적화 항목:**

1. 검색 성능 최적화
   - BM25 파라미터 튜닝
   - 청킹 크기 최적화

2. 임베딩 성능 최적화
   - 배치 처리
   - 캐싱
   - GPU 사용

3. Re-ranking 최적화
   - 캐시 활용
   - 병렬 처리

4. 메모리 최적화
   - 메모리 누수 체크
   - 모델 양자화

5. 응답 시간 최적화
   - 비동기 처리
   - 병렬 검색

**성능 목표:**
- PHASE 1: < 1초
- PHASE 2: < 2초
- PHASE 3: < 3초

프로파일링과 성능 측정을 통해 최적화해줄래?""",

    "deploy_to_production": """RAG 시스템을 프로덕션에 배포해줄래?

**배포 작업:**

1. Docker 이미지 생성
2. API 서버 구축 (FastAPI)
3. 배포 설정 (docker-compose)
4. 모니터링 설정
5. CI/CD 파이프라인
6. 문서화

배포 준비 다 해줄래?"""
}

def print_header():
    """헤더 출력"""
    print("\n" + "=" * 80)
    print("🚀 Claude Code RAG 시스템 프롬프트 매니저")
    print("=" * 80 + "\n")

def print_menu():
    """메뉴 출력"""
    print("📋 사용 가능한 프롬프트:\n")
    
    categories = {
        "🔧 초기 설정": ["initialize_project", "setup_environment"],
        "📊 단계별 구현": ["phase1_implementation", "phase2_implementation", "phase3_implementation"],
        "🎯 전체": ["full_implementation"],
        "📈 추가 작업": ["generate_test_dataset", "performance_optimization", "deploy_to_production"]
    }
    
    for category, prompts in categories.items():
        print(f"\n{category}")
        print("-" * 80)
        for i, prompt_key in enumerate(prompts, 1):
            prompt_info = PROMPTS.get(prompt_key, {})
            title = prompt_info.get("title", prompt_key)
            duration = prompt_info.get("duration", "")
            print(f"  {i}. {title:<50} ({duration})")
            desc = prompt_info.get("description", "")
            if desc:
                print(f"     → {desc}")

def show_details(prompt_key: str):
    """프롬프트 상세 정보 표시"""
    if prompt_key not in PROMPTS:
        print(f"❌ 알 수 없는 프롬프트: {prompt_key}")
        return False
    
    info = PROMPTS[prompt_key]
    
    print("\n" + "=" * 80)
    print(f"📌 {info['title']}")
    print("=" * 80)
    print(f"\n📝 설명: {info['description']}")
    print(f"⏱️  예상 시간: {info['duration']}")
    
    if "target_accuracy" in info:
        print(f"🎯 목표 정확도: {info['target_accuracy']}")
    if "target_time" in info:
        print(f"⚡ 목표 응답시간: {info['target_time']}")
    if "target_confidence" in info:
        print(f"💯 목표 신뢰도: {info['target_confidence']}")
    
    if info.get("dependencies"):
        print(f"\n📦 의존성:")
        for dep in info["dependencies"]:
            print(f"   - {PROMPTS.get(dep, {}).get('title', dep)}")
    
    if "note" in info:
        print(f"\n⚠️  주의: {info['note']}")

def copy_to_clipboard(text: str) -> bool:
    """텍스트를 클립보드에 복사"""
    try:
        pyperclip.copy(text)
        return True
    except:
        return False

def main():
    """메인 함수"""
    print_header()
    
    if len(sys.argv) > 1:
        # 커맨드라인 인자 처리
        if sys.argv[1] == "copy" and len(sys.argv) > 2:
            prompt_key = sys.argv[2]
            if prompt_key in PROMPTS_TEXT:
                text = PROMPTS_TEXT[prompt_key]
                if copy_to_clipboard(text):
                    print(f"✅ '{prompt_key}' 프롬프트가 클립보드에 복사되었습니다!")
                    print("\nClaude Code에 다음 명령어 실행:")
                    print("  claude code")
                    print("\n그리고 Ctrl+V (Cmd+V on Mac)로 프롬프트를 붙여넣으세요.")
                else:
                    print(f"❌ 클립보드 복사 실패")
            else:
                print(f"❌ 알 수 없는 프롬프트: {prompt_key}")
        
        elif sys.argv[1] in PROMPTS_TEXT:
            # 프롬프트 표시
            prompt_key = sys.argv[1]
            show_details(prompt_key)
            print("\n" + "-" * 80)
            print("\n📄 프롬프트:\n")
            print(PROMPTS_TEXT[prompt_key])
            print("\n" + "-" * 80)
            print(f"\n💡 사용 방법:")
            print(f"  1. claude code 실행")
            print(f"  2. 위의 프롬프트를 복사")
            print(f"  3. Claude Code에 붙여넣기")
            
            # 자동 복사 제안
            if copy_to_clipboard(PROMPTS_TEXT[prompt_key]):
                print(f"\n✅ 프롬프트가 클립보드에 자동 복사되었습니다!")
        else:
            print("❌ 사용법:")
            print(f"  python {sys.argv[0]} <prompt_name>")
            print(f"  python {sys.argv[0]} copy <prompt_name>")
            print(f"\n사용 가능한 프롬프트:")
            for key in PROMPTS_TEXT.keys():
                print(f"  - {key}")
    else:
        # 대화형 모드
        print_menu()
        print("\n" + "=" * 80)
        print("사용 방법:")
        print("=" * 80)
        print("\n옵션 1: 프롬프트 복사")
        print("  python rag_prompt_manager.py copy phase1_implementation")
        print("\n옵션 2: 프롬프트 확인")
        print("  python rag_prompt_manager.py phase1_implementation")
        print("\n옵션 3: Claude Code에서 직접 실행")
        print("  1. 'claude code' 실행")
        print("  2. 아래 프롬프트 중 하나 선택해서 복사")
        print("  3. Claude Code에 붙여넣기")
        print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 종료합니다.")
        sys.exit(0)
